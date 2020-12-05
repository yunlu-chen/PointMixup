from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import pointnet2.train.etw_pytorch_utils as pt_utils
from collections import namedtuple

from pointnet2.utils.pointnet2_modules import PointnetSAModuleMSG, PointnetSAModule
import itertools
import torch.nn.functional as F
import numpy as np
try:
    from emd_ import emd_module
except:
    pass
try:
    from cd.chamferdist import ChamferDistance as CD
except:
    pass

def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])

    def model_fn(model, data, epoch=0, eval=False, idx_minor=None, mixrates=None, strategy=None, manilayer_batch=0):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.to("cuda", non_blocking=True).contiguous()  # [16, 1024, 3]
            labels = labels.to("cuda", non_blocking=True)  # [16, 1]

            preds = model(inputs)
            labels = labels.view(-1)
            loss = criterion(preds, labels)

            _, classes = torch.max(preds, -1)
            acc = (classes == labels).float().sum() / labels.numel()

            return ModelReturn(preds, loss, {"acc": acc.item(), "loss": loss.item()})

    return model_fn

def model_fn_decorator_mix(criterion_train, criterion_eval, num_class=40):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])

    def model_fn(model, data, epoch=0, eval=False, idx_minor=None, mixrates=None, strategy=None, manilayer_batch=0):
        with torch.set_grad_enabled(not eval):
            if eval:
                inputs, labels = data
                inputs = inputs.to("cuda", non_blocking=True)  # [16, 1024, 3]
                labels = labels.to("cuda", non_blocking=True)  # [16, 1]

                preds = model(inputs)

                labels = labels.view(-1)
                loss = criterion_eval(preds, labels)

                assert (preds.shape[1] - num_class) % (num_class + num_class*(num_class-1)/2) == 0
                n_strategies = int((preds.shape[1] - num_class) / (num_class + num_class*(num_class-1)/2)) # calculate n_strategies
                if n_strategies > 0:# new class calculation 820
                    preds = F.softmax(preds, dim=1)
                    preds_original = preds[:,0:num_class]
                    preds_original = preds_original + preds[:,num_class:int(2*num_class)]
                    preds_ex =  preds[:, int(2*num_class):int(2*num_class+num_class*(num_class-1)/2)]
                    for i in range(1,n_strategies):
                        preds_original = preds_original + preds[:, int(num_class + i*(num_class + num_class*(num_class-1)/2)): int(num_class*2 + i*(num_class + num_class*(num_class-1)/2))]
                        preds_ex = preds_ex + preds[:, int(num_class*2 + i * (
                                    num_class + num_class * (num_class - 1) / 2)): int(num_class + (i+1) * (
                                    num_class + num_class * (num_class - 1) / 2) )]
                    pairs = list(itertools.combinations(range(num_class), 2))
                    for i in range(len(pairs)):
                        preds_original[:, pairs[i][0]] = preds_original[:, pairs[i][0]] + preds_ex[:, i]/2
                        preds_original[:, pairs[i][1]] = preds_original[:, pairs[i][1]] + preds_ex[:, i]/2

                _, classes = torch.max(preds, -1)
                acc = (classes == labels).float().sum() / labels.numel()
            else:
                inputs, labels = data
                inputs = inputs.to("cuda", non_blocking=True)  # [16, 1024, 3]
                labels = labels.to("cuda", non_blocking=True)  # [16, num_class]
                if idx_minor is not None: # for manimix
                    preds = model(inputs, idx_minor=idx_minor, mixrates=mixrates, strategy=strategy, manilayer_batch=manilayer_batch)
                else:
                    preds = model(inputs)
                #labels = labels.view(-1)
                loss = criterion_train(preds, labels)

                acc = loss

            return ModelReturn(preds, loss, {"acc": acc.item(), "loss": loss.item()})

    return model_fn

class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Classification network

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=3, use_xyz=True):
        super(Pointnet2MSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[
                    [input_channels, 32, 32, 64],
                    [input_channels, 64, 64, 128],
                    [input_channels, 64, 96, 128],
                ],
                use_xyz=use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(mlp=[128 + 256 + 256, 256, 512, 1024], use_xyz=use_xyz)
        )
        self.FC_layer = (
            pt_utils.Seq(1024)
            .fc(512, bn=True)
            .dropout(0.5)
            .fc(256, bn=True)
            .dropout(0.5)
            .fc(num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)  # torch.Size([16, 1024, 1])

        return self.FC_layer(features.squeeze(-1))



#########################################################################################################


class Pointnet2MSG_manimix(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Classification network

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=3, use_xyz=True, n_strxmani=None, align=False):
        super(Pointnet2MSG_manimix, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[
                    [input_channels, 32, 32, 64],
                    [input_channels, 64, 64, 128],
                    [input_channels, 64, 96, 128],
                ],
                use_xyz=use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(mlp=[128 + 256 + 256, 256, 512, 1024], use_xyz=use_xyz)
        )
        self.FC_layer = (
            pt_utils.Seq(1024)
                .fc(512, bn=True)
                .dropout(0.5)
                .fc(256, bn=True)
                .dropout(0.5)
                .fc(num_classes, activation=None)
        )

        if n_strxmani is not None:
            self.mixupbias_layer = (
                pt_utils.Seq(n_strxmani)
                    .fc(64, bn=True)
                    .fc(num_classes, activation=None)
            )
        self.EMD = emd_module.emdModule()
        self.cd = CD()
        self.align = align


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, idx_minor=None, mixrates=None, strategy=None, manilayer_batch=0):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules[0:manilayer_batch]:
            xyz, features = module(xyz, features)  # torch.Size([16, 1024, 1])

        if strategy is not None: # in test there is no mixup
            B, N, C = xyz.shape
            # rotate the align
            xyz_minor = xyz[idx_minor]

            if strategy == 'pointmixup': # OT
                mix_rate = torch.tensor(mixrates).cuda().float()
                mix_rate = mix_rate.unsqueeze_(1).unsqueeze_(2)

                mix_rate_expand_xyz = mix_rate.expand(xyz.shape)
                if features is not None:
                    features_minor = features[idx_minor]
                    features_minor = features_minor.transpose(1, 2)
                    features_minor_new = torch.zeros_like(features_minor).cuda()
                    mix_rate_expand_features = mix_rate.expand(features.shape)

                if self.align:
                    with torch.no_grad():
                        cd_all = torch.zeros([60, B]).cuda()
                        for i in range(60):
                            theta_temp = (torch.ones([B]) * (i / 60 * 3.1415927))
                            refl = torch.zeros([3, 3, B]).cuda()
                            cos = torch.cos(theta_temp)
                            sin = torch.sin(theta_temp)
                            refl[0][0] = 1 - sin ** 2 * 2
                            refl[0][2] = 2 * sin * cos
                            refl[1][1] = 1
                            refl[2][0] = 2 * sin * cos
                            refl[2][2] = 1 - cos ** 2 * 2

                            refl = refl.permute([2, 0, 1])
                            xyz_refl = torch.matmul(xyz, refl)
                            cd0, cd1, _, _ = self.cd(xyz, xyz_refl)
                            cd_all[i] = (cd0 + cd1).sum(dim=1)

                        _, ind_all = torch.min(cd_all, dim=0)
                        thetas = (ind_all.float() / 60 * 3.1415927)

                        thetas_minor = thetas[idx_minor]
                        thetas_diff = thetas_minor - thetas
                        coss = torch.cos(-thetas_diff)
                        sins = torch.sin(-thetas_diff)
                        rota = torch.zeros([3, 3, B]).cuda()
                        rota[0][0] = coss
                        rota[0][2] = sins
                        rota[1][1] = 1
                        rota[2][0] = -sins
                        rota[2][2] = coss
                        rota = rota.permute([2, 0, 1])
                    xyz_minor_rot = torch.matmul(xyz_minor, rota)
                    _, ass = self.EMD(xyz, xyz_minor_rot, 0.005, 300) # mapping
                else:
                    _, ass = self.EMD(xyz, xyz_minor, 0.005, 300)  # mapping
                ass = ass.long()
                for i in range(B):
                    xyz_minor[i] = xyz_minor[i][ass[i]]
                    if features is not None:
                        features_minor_new[i] = features_minor[i][ass[i]]
                xyz = xyz * (1 - mix_rate_expand_xyz) + xyz_minor * mix_rate_expand_xyz
                if features is not None:
                    features_minor_new = features_minor_new.transpose(1, 2)
                    features = features * (1 - mix_rate_expand_features) + features_minor_new * mix_rate_expand_features
            else:
                raise NotImplementedError

        for module in self.SA_modules[manilayer_batch:3]:
            xyz, features = module(xyz, features)  # torch.Size([16, 1024, 1])

        x = self.FC_layer(features.squeeze(-1))

        return x


