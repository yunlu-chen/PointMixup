from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pointnet2.train.etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import os
import argparse
import tqdm
# from etw_pytorch_utils import checkpoint_state, save_checkpoint
import numpy as np
from utils import CrossEntropyLoss_with_prob, cross_entropy_with_probs

from pointnet2.models import Pointnet2ClsMSG as Pointnet
from pointnet2.models.pointnet2_msg_cls import Pointnet2MSG_manimix as Pointnet_manimix

from pointnet2.models.pointnet2_msg_cls import model_fn_decorator, model_fn_decorator_mix
from pointnet2.data import ModelNet40Cls
import pointnet2.data.data_utils as d_utils

from pytorchgo.utils import logger
from pytorchgo.utils.pytorch_utils import model_summary, optimizer_summary

from pytorchgo.utils.pytorch_utils import set_gpu

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-model", type=str, default='pointnet2', help="pointnet2"
    )
    parser.add_argument(
        "-data", type=str, default='modelnet40', help="modelnet40"
    )
    parser.add_argument("-batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "-num_points", type=int, default=1024, help="Number of points to train with"
    )
    parser.add_argument(
        "-weight_decay", type=float, default=-1, help="L2 regularization coeff, -1 use defined value"
    )
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=float, default=2.5e5, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"
    )
    parser.add_argument(
        "-savename", type=str, default='testexp', help="savename"
    )
    parser.add_argument('-pointmixup', type=boolean_string, default=True, help="if use pointmixup (including point manifold mixup)")
    parser.add_argument('-manimixup', type=boolean_string, default=True, help="if use manifold mixup instead of only input mixup in PointMixup")
    parser.add_argument(
        "-epochs", type=int, default=500, help="Number of epochs to train for"
    )
    # parser.add_argument(
    #     "-manilayer", type=int, default=0, help="[0,1,2]"
    # )
    parser.add_argument('-rot', type=boolean_string, default=True, help="random up-rotation")
    parser.add_argument(
        "-mixup_alpha", type=float, default=-1, help="mixup parameter that controls beta distribution where the mixrate is drawn from. -1 use defined parameter"
    )
    parser.add_argument('-evaluate', action='store_true',
                        help='evaluate')
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument('-align', type=boolean_string, default=False, help="if align the up-rotation of shapes by symmetry axis, recommended for pointmixup (input mixup)")

    return parser.parse_args()


lr_clip = 1e-5
bnm_clip = 1e-2

if __name__ == "__main__":
    args = parse_args()

    if args.weight_decay < 0:  # default
        if not args.pointmixup:
            args.weight_decay = 1e-5
        else:
            if not args.manimixup:
                args.weight_decay = 5e-6
            else:
                args.weight_decay = 1e-6

    if args.pointmixup and args.mixup_alpha < 0:
        if args.rot:
            if not args.manimixup:
                args.mixup_alpha = 0.4
            else:
                args.mixup_alpha = 1.5
        else:
            if not args.manimixup:
                args.mixup_alpha = 1.0
            else:
                args.mixup_alpha = 2.0

    if (args.pointmixup and args.rot) and (not args.manimixup):
        args.align = True
    else:
        args.align = False

    # logger.auto_set_dir('d', "{}_{}".format(args.data, args.model))
    logger.auto_set_dir('d', "{}".format(args.savename))


    if args.data == 'modelnet40':
        num_class = 40
        dataset_cls = ModelNet40Cls
    else:
        raise NotImplementedError

    args.epochs = int(args.epochs)
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(comment=args.savename)
    writer.add_text('args', str(args), 0)
    transforms_test = d_utils.PointcloudToTensor()
    if args.rot:
        transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudScale(),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(),
                d_utils.PointcloudJitter()
            ]
        )
    else:
        transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudScale(),
                d_utils.PointcloudTranslate(),
                d_utils.PointcloudJitter()
            ]
        )

    if args.data == 'modelnet40':
        num_class = 40
        dataset_cls = ModelNet40Cls
    test_set = dataset_cls(args.num_points, transforms=transforms_test, train=False)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    train_set = dataset_cls(args.num_points, transforms=transforms, keeprate=1.0)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
    )

    n_strategies = 0
    if args.pointmixup: n_strategies += 1
    if args.manimixup: n_strategies = n_strategies * 3

    if args.model == 'pointnet2':
        if args.pointmixup:  # input mixup is incorporated with manimix
            model = Pointnet_manimix(input_channels=0, num_classes=num_class, use_xyz=True, align=args.align)
        else:
            model = Pointnet(input_channels=0, num_classes=num_class, use_xyz=True)
    model.cuda()

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_lbmd = lambda it: max(
        args.lr_decay ** (int(it * args.batch_size / args.decay_step)),
        lr_clip / args.lr,
    )
    bn_lbmd = lambda it: max(
        args.bn_momentum
        * args.bnm_decay ** (int(it * args.batch_size / args.decay_step)),
        bnm_clip,
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1

    # load status from checkpoint
    if args.checkpoint is not None:
        logger.warning("loading checkpoint weight file")
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=it)
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bn_lbmd, last_epoch=it
    )
    it = max(it, 0)  # for the initialize value of `trainer.train`
    if args.pointmixup:
        model_fn = model_fn_decorator_mix(cross_entropy_with_probs, nn.CrossEntropyLoss(), num_class=num_class)
    else:
        model_fn = model_fn_decorator(nn.CrossEntropyLoss())

    if not osp.isdir("checkpoints"):
        os.makedirs("checkpoints")

    model_summary(model)
    optimizer_summary(optimizer)

    trainer = pt_utils.Trainer_mix(
        model,
        model_fn,
        optimizer,
        checkpoint_name="checkpoints/" + args.savename,
        best_name="checkpoints/best" + args.savename,
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        savename=args.savename,
        eval_frequency=int(len(train_loader)),
        pointmixup=args.pointmixup,
        manimixup=args.manimixup,
        alpha=args.mixup_alpha
    )

    if args.evaluate:
        logger.warning("evaluating mode")
        _ = trainer.eval_epoch(test_loader)
        exit(0)

    trainer.train(
        it, start_epoch, args.epochs, train_loader, test_loader, best_loss=best_loss, writer=writer
    )

    if start_epoch == args.epochs:
        _ = trainer.eval_epoch(test_loader)

# writer.export_scalars_to_json("all_scalars.json")
writer.close()
