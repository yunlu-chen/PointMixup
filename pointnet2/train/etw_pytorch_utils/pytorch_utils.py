from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction
from itertools import repeat
import numpy as np
import shutil
import tqdm
from scipy.stats import t as student_t
import statistics as stats
try:
    import emd_module as emd
except:
    pass
if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *
import os
from pytorchgo.utils import logger

class SharedMLP(nn.Sequential):

    def __init__(self,
                 args,
                 bn = False,
                 activation=nn.ReLU(inplace=True),
                 preact = False,
                 first = False,
                 name = ""):
        # type: (SharedMLP, List[int], bool, Any, bool, bool, AnyStr) -> None
        super(SharedMLP, self).__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation if (not first or not preact or
                                              (i != 0)) else None,
                    preact=preact))


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super(_BNBase, self).__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size,  name = ""):
        # type: (BatchNorm1d, int, AnyStr) -> None
        super(BatchNorm1d, self).__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size, name = ""):
        # type: (BatchNorm2d, int, AnyStr) -> None
        super(BatchNorm2d, self).__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):

    def __init__(self, in_size, name = ""):
        # type: (BatchNorm3d, int, AnyStr) -> None
        super(BatchNorm3d, self).__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class _ConvBase(nn.Sequential):

    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 activation,
                 bn,
                 init,
                 conv=None,
                 norm_layer=None,
                 bias=True,
                 preact=False,
                 name=""):
        super(_ConvBase, self).__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = norm_layer(out_size)
            else:
                bn_unit = norm_layer(in_size)

        if preact:
            if bn:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv1d(_ConvBase):

    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size= 1,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 activation=nn.ReLU(inplace=True),
                 bn = False,
                 init=nn.init.kaiming_normal_,
                 bias = True,
                 preact = False,
                 name = "",
                 norm_layer=BatchNorm1d):
        # type: (Conv1d, int, int, int, int, int, int, Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(Conv1d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            name=name)


class Conv2d(_ConvBase):

    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size = (1, 1),
                 stride = (1, 1),
                 padding = (0, 0),
                 dilation = (1, 1),
                 activation=nn.ReLU(inplace=True),
                 bn = False,
                 init=nn.init.kaiming_normal_,
                 bias = True,
                 preact = False,
                 name = "",
                 norm_layer=BatchNorm2d):
        # type: (Conv2d, int, int, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(Conv2d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            name=name)


class Conv3d(_ConvBase):

    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size = (1, 1, 1),
                 stride = (1, 1, 1),
                 padding = (0, 0, 0),
                 dilation = (1, 1, 1),
                 activation=nn.ReLU(inplace=True),
                 bn = False,
                 init=nn.init.kaiming_normal_,
                 bias = True,
                 preact = False,
                 name = "",
                 norm_layer=BatchNorm3d):
        # type: (Conv3d, int, int, Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(Conv3d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.Conv3d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            name=name)


class FC(nn.Sequential):

    def __init__(self,
                 in_size,
                 out_size,
                 activation=nn.ReLU(inplace=True),
                 bn = False,
                 init=None,
                 preact = False,
                 name = ""):
        # type: (FC, int, int, Any, bool, Any, bool, AnyStr) -> None
        super(FC, self).__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant_(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)


class _DropoutNoScaling(InplaceFunction):

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        if inplace:
            return None
        n = g.appendNode(
            g.create("Dropout", [input]).f_("ratio", p).i_(
                "is_test", not train))
        real = g.appendNode(g.createSelect(n, 0))
        g.appendNode(g.createSelect(n, 1))
        return real

    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = cls._make_noise(input)
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise.expand_as(input)
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(ctx.noise), None, None, None
        else:
            return grad_output, None, None, None


dropout_no_scaling = _DropoutNoScaling.apply


class _FeatureDropoutNoScaling(_DropoutNoScaling):

    @staticmethod
    def symbolic(input, p=0.5, train=False, inplace=False):
        return None

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(
            input.size(0), input.size(1), *repeat(1,
                                                  input.dim() - 2))


feature_dropout_no_scaling = _FeatureDropoutNoScaling.apply


def group_model_params(model, **kwargs):
    # type: (nn.Module, ...) -> List[Dict]
    decay_group = []
    no_decay_group = []

    for name, param in model.named_parameters():
        if name.find("normlayer") != -1 or name.find("bias") != -1:
            no_decay_group.append(param)
        else:
            decay_group.append(param)

    assert len(list(
        model.parameters())) == len(decay_group) + len(no_decay_group)

    return [
        dict(params=decay_group, **kwargs),
        dict(params=no_decay_group, weight_decay=0.0, **kwargs)
    ]


def checkpoint_state(model=None,
                     optimizer=None,
                     best_prec=None,
                     epoch=None,
                     it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        'epoch': epoch,
        'it': it,
        'best_prec': best_prec,
        'model_state': model_state,
        'optimizer_state': optim_state
    }


def save_checkpoint(state,
                    is_best,
                    filename='checkpoint',
                    bestname='model_best'):
    filename = '{}.pth.tar'.format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.pth.tar'.format(bestname))


def load_checkpoint(model=None, optimizer=None, filename='checkpoint'):
    filename = "{}.pth.tar".format(filename)
    
    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        it = checkpoint.get('it', 0.0)
        best_prec = checkpoint['best_prec']
        logger.warning("checkpoint it:{}, best_prec:{}".format(it,best_prec))
        if model is not None and checkpoint['model_state'] is not None:
            logger.warning("load model_state")
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            logger.warning("load optimizer_state")
            optimizer.load_state_dict(checkpoint['optimizer_state'])


        print("==> Done")
        return it, epoch, best_prec
    else:
        logger.warning("==> Checkpoint '{}' not found".format(filename))
        raise
        return None
    
def variable_size_collate(pad_val=0, use_shared_memory=True):
    import collections
    _numpy_type_map = {
        'float64': torch.DoubleTensor,
        'float32': torch.FloatTensor,
        'float16': torch.HalfTensor,
        'int64': torch.LongTensor,
        'int32': torch.IntTensor,
        'int16': torch.ShortTensor,
        'int8': torch.CharTensor,
        'uint8': torch.ByteTensor,
    }

    def wrapped(batch):
        "Puts each data field into a tensor with outer dimension batch size"

        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        elem_type = type(batch[0])
        if torch.is_tensor(batch[0]):
            max_len = 0
            for b in batch:
                max_len = max(max_len, b.size(0))

            numel = sum([int(b.numel() / b.size(0) * max_len) for b in batch])
            if use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            else:
                out = batch[0].new(numel)

            out = out.view(
                len(batch), max_len,
                *[batch[0].size(i) for i in range(1, batch[0].dim())])
            out.fill_(pad_val)
            for i in range(len(batch)):
                out[i, 0:batch[i].size(0)] = batch[i]

            return out
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                return wrapped([torch.from_numpy(b) for b in batch])
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return _numpy_type_map[elem.dtype.name](list(
                    map(py_type, batch)))
        elif isinstance(batch[0], int):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0], collections.Mapping):
            return {key: wrapped([d[key] for d in batch]) for key in batch[0]}
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [wrapped(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    return wrapped


class TrainValSplitter():
    r"""
        Creates a training and validation split to be used as the sampler in a pytorch DataLoader
    Parameters
    ---------
        numel : int
            Number of elements in the entire training dataset
        percent_train : float
            Percentage of data in the training split
        shuffled : bool
            Whether or not shuffle which data goes to which split
    """

    def __init__(self,
                 
                 numel,
                 percent_train,
                 shuffled = False):
        # type: (TrainValSplitter, int, float, bool) -> None
        indicies = np.array([i for i in range(numel)])
        if shuffled:
            np.random.shuffle(indicies)

        self.train = torch.utils.data.sampler.SubsetRandomSampler(
            indicies[0:int(percent_train * numel)])
        self.val = torch.utils.data.sampler.SubsetRandomSampler(
            indicies[int(percent_train * numel):-1])


class CrossValSplitter():
    r"""
        Class that creates cross validation splits.  The train and val splits can be used in pytorch DataLoaders.  The splits can be updated
        by calling next(self) or using a loop:
            for _ in self:
                ....
    Parameters
    ---------
        numel : int
            Number of elements in the training set
        k_folds : int
            Number of folds
        shuffled : bool
            Whether or not to shuffle which data goes in which fold
    """

    def __init__(self,  numel, k_folds, shuffled= False):
        # type: (CrossValSplitter, int, int, bool) -> None
        inidicies = np.array([i for i in range(numel)])
        if shuffled:
            np.random.shuffle(inidicies)

        self.folds = np.array(np.array_split(inidicies, k_folds), dtype=object)
        self.current_v_ind = -1

        self.val = torch.utils.data.sampler.SubsetRandomSampler(self.folds[0])
        self.train = torch.utils.data.sampler.SubsetRandomSampler(
            np.concatenate(self.folds[1:], axis=0))

        self.metrics = {}

    def __iter__(self):
        self.current_v_ind = -1
        return self

    def __len__(self):
        return len(self.folds)

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self)
        self.val.inidicies = self.folds[idx]
        self.train.inidicies = np.concatenate(
            self.folds[np.arange(len(self)) != idx], axis=0)

    def __next__(self):
        self.current_v_ind += 1
        if self.current_v_ind >= len(self):
            raise StopIteration

        self[self.current_v_ind]

    def update_metrics(self, to_post):
        # type: (CrossValSplitter, dict) -> None
        for k, v in to_post.items():
            if k in self.metrics:
                self.metrics[k].append(v)
            else:
                self.metrics[k] = [v]

    def print_metrics(self):
        for name, samples in self.metrics.items():
            xbar = stats.mean(samples)
            sx = stats.stdev(samples, xbar)
            tstar = student_t.ppf(1.0 - 0.025, len(samples) - 1)
            margin_of_error = tstar * sx / sqrt(len(samples))
            print("{}: {} +/- {}".format(name, xbar, margin_of_error))


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(self,
                 model,
                 bn_lambda,
                 last_epoch=-1,
                 setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(
                type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class Trainer(object):
    r"""
        Reasonably generic trainer for pytorch models

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    eval_frequency : int
        How often to run an eval
    log_name : str
        Name of file to output tensorboard_logger to
    """

    def __init__(self,
                 model,
                 model_fn,
                 optimizer,
                 checkpoint_name="ckpt",
                 best_name="best",
                 lr_scheduler=None,
                 bnm_scheduler=None,
                 eval_frequency=-1,
                 model_fn_val=None,
                 savename='savename'):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model, model_fn, optimizer, lr_scheduler, bnm_scheduler)

        self.checkpoint_name, self.best_name = checkpoint_name, best_name
        self.eval_frequency = eval_frequency

        self.training_best, self.eval_best = {}, {}
        self.savename = savename

    @staticmethod
    def _decode_value(v):
        if isinstance(v[0], float):
            return np.mean(v)
        elif isinstance(v[0], tuple):
            if len(v[0]) == 3:
                num = [l[0] for l in v]
                denom = [l[1] for l in v]
                w = v[0][2]
            else:
                num = [l[0] for l in v]
                denom = [l[1] for l in v]
                w = None

            return np.average(
                np.sum(num, axis=0) / (np.sum(denom, axis=0) + 1e-6), weights=w)
        else:
            raise AssertionError("Unknown type: {}".format(type(v)))

    def _train_it(self, it, batch):
        self.model.train()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(it)

        if self.bnm_scheduler is not None:
            self.bnm_scheduler.step(it)

        self.optimizer.zero_grad()
        _, loss, eval_res = self.model_fn(self.model, batch)


        loss.backward()
        self.optimizer.step()

        return eval_res

    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = 0.0
        count = 1.0
        for i, data in tqdm.tqdm(
                enumerate(d_loader, 0),
                total=len(d_loader),
                leave=False,
                desc='val'):
            self.optimizer.zero_grad()

            _, loss, eval_res = self.model_fn(self.model, data, eval=True)

            total_loss += loss.item()
            count += 1
            for k, v in eval_res.items():
                if v is not None:
                    eval_dict[k] = eval_dict.get(k, []) + [v]

        logger.warning("evaluation loss={}, result={}".format(total_loss / count, eval_dict))
        return total_loss / count, eval_dict

    def train(self,
              start_it,
              start_epoch,
              n_epochs,
              train_loader,
              test_loader=None,
              best_loss=0.0,
              writer=None):
        r"""
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : torch.utils.data.DataLoader
            DataLoader of the test_data
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_loss : float
            Testing loss of the best model
        """

        eval_frequency = (self.eval_frequency
                          if self.eval_frequency > 0 else len(train_loader))

        it = start_it
        with tqdm.trange(start_epoch, n_epochs + 1, desc='epochs') as tbar, \
                tqdm.tqdm(total=eval_frequency, leave=False, desc='train') as pbar:

            for epoch in tbar:
                for batch in train_loader:
                    res = self._train_it(it, batch)
                    it += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.refresh()


                    if (it % eval_frequency) == 0:
                        pbar.close()

                        if test_loader is not None:
                            val_loss, res = self.eval_epoch(test_loader)

                            if writer is not None:
                                writer.add_scalar('{}/valacc'.format(self.savename), np.mean(res['acc']), epoch)
                                writer.add_scalar('{}/valloss'.format(self.savename), np.mean(res['loss']), epoch)

                            is_best = val_loss < best_loss
                            best_loss = min(best_loss, val_loss)
                            save_checkpoint(
                                checkpoint_state(self.model, self.optimizer,
                                                 val_loss, epoch, it),
                                is_best,
                                filename=self.checkpoint_name,
                                bestname=self.best_name)

                        pbar = tqdm.tqdm(
                            total=eval_frequency, leave=False, desc='train')
                        pbar.set_postfix(dict(total_it=it))

        return best_loss


class Trainer_mix(object):
    r"""
        Adapted trainer for PointMixup / point manifold mixup

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    eval_frequency : int
        How often to run an eval
    log_name : str
        Name of file to output tensorboard_logger to
    """

    def __init__(self,
                 model,
                 model_fn,
                 optimizer,
                 checkpoint_name="ckpt",
                 best_name="best",
                 lr_scheduler=None,
                 bnm_scheduler=None,
                 eval_frequency=-1,
                 model_fn_val=None,
                 savename='savename',
                 pointmixup=False,
                 manimixup=False,
                 alpha=0.4,
                 n_class=40,
                 manilayer_all=3,# determined by model
                 ):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model, model_fn, optimizer, lr_scheduler, bnm_scheduler)

        self.checkpoint_name, self.best_name = checkpoint_name, best_name
        self.eval_frequency = eval_frequency

        self.training_best, self.eval_best = {}, {}
        self.savename = savename
        self.pointmixup = pointmixup
        self.manimixup = manimixup
        self.alpha = alpha
        self.n_class = n_class
        #self.manilayer = 0
        self.manilayer_all = manilayer_all
        self.strategies = []


        if pointmixup:
            self.strategies.append('pointmixup')
        self.n_strategies = len(self.strategies)

    @staticmethod
    def _decode_value(v):
        if isinstance(v[0], float):
            return np.mean(v)
        elif isinstance(v[0], tuple):
            if len(v[0]) == 3:
                num = [l[0] for l in v]
                denom = [l[1] for l in v]
                w = v[0][2]
            else:
                num = [l[0] for l in v]
                denom = [l[1] for l in v]
                w = None

            return np.average(
                np.sum(num, axis=0) / (np.sum(denom, axis=0) + 1e-6), weights=w)
        else:
            raise AssertionError("Unknown type: {}".format(type(v)))

    def _train_it(self, it, batch, idx_minor=None, mixrates=None, strategy=None, manilayer_batch=0):
        self.model.train()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(it)

        if self.bnm_scheduler is not None:
            self.bnm_scheduler.step(it)

        self.optimizer.zero_grad()
        _, loss, eval_res = self.model_fn(self.model, batch,  idx_minor=idx_minor, mixrates=mixrates, strategy=strategy, manilayer_batch=manilayer_batch)

        if it%50 == 0:
            logger.info("loss={}".format(loss.item()))

        loss.backward()
        self.optimizer.step()

        return eval_res

    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = 0.0
        count = 1.0
        for i, data in tqdm.tqdm(
                enumerate(d_loader, 0),
                total=len(d_loader),
                leave=False,
                desc='val'):
            self.optimizer.zero_grad()

            _, loss, eval_res = self.model_fn(self.model, data, eval=True)

            total_loss += loss.item()
            count += 1
            for k, v in eval_res.items():
                if v is not None:
                    eval_dict[k] = eval_dict.get(k, []) + [v]

        acc = sum(eval_dict['acc']) / len(eval_dict['acc'])
        logger.warning("evaluation loss={}, acc={}".format(total_loss / count, acc))
        return total_loss / count, eval_dict

    def train(self,
              start_it,
              start_epoch,
              n_epochs,
              train_loader,
              test_loader=None,
              best_loss=0.0,
              writer=None):
        r"""
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : torch.utils.data.DataLoader
            DataLoader of the test_data
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_loss : float
            Testing loss of the best model
        """

        eval_frequency = (self.eval_frequency
                          if self.eval_frequency > 0 else len(train_loader))

        it = start_it
        best_acc = -1
        with tqdm.trange(start_epoch, n_epochs + 1, desc='epochs') as tbar, \
                tqdm.tqdm(total=eval_frequency, leave=False, desc='train') as pbar:

            for epoch in tbar:
                for batch in train_loader:
                    idx_minor = None
                    mixrates = None
                    strategy = None
                    manilayer_batch = 0 # all use in no mixup case
                    if self.n_strategies > 0:
                        strategy_idx = np.random.randint(self.n_strategies)
                        strategy = self.strategies[strategy_idx]
                        if self.manimixup:
                            manilayer_batch = np.random.randint(self.manilayer_all)
                        else:
                            manilayer_batch = 0

                        B, N, C = batch[0].shape
                        idx_minor = torch.randperm(B)

                        mixrates = (0.5 - np.abs(np.random.beta(self.alpha, self.alpha, B) - 0.5))
                        label_main = batch[1]
                        label_minor = batch[1][idx_minor]

                        label = torch.zeros(B, self.n_class)
                        for i in range(B):
                            if label_main[i] == label_minor[i]: # same label
                                label[i][label_main[i]] = 1.0
                            else:
                                label[i][label_main[i]] = 1 - mixrates[i]
                                label[i][label_minor[i]] = mixrates[i]
                        batch[1] = label

                    res = self._train_it(it, batch, idx_minor, mixrates, strategy, manilayer_batch)
                    it += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.refresh()

                    if (it % eval_frequency) == 0:
                        pbar.close()

                        if test_loader is not None:
                            val_loss, res = self.eval_epoch(test_loader)

                            if writer is not None:
                                writer.add_scalar('{}/valacc'.format(self.savename), np.mean(res['acc']), epoch)
                                writer.add_scalar('{}/valloss'.format(self.savename), np.mean(res['loss']), epoch)

                            #is_best = val_loss < best_loss
                            best_loss = min(best_loss, val_loss)
                            is_best = np.mean(res['acc']) > best_acc
                            if is_best:
                                best_acc = np.mean(res['acc'])

                            logger.warning("eval_loss={}, eval_acc={}, eval_best_acc={}".format(np.mean(res['loss']), np.mean(res['acc']), best_acc))
                            save_checkpoint(
                                checkpoint_state(self.model, self.optimizer,
                                                 val_loss, epoch, it),
                                is_best,
                                filename=self.checkpoint_name,
                                bestname=self.best_name)

                        pbar = tqdm.tqdm(
                            total=eval_frequency, leave=False, desc='train')
                        pbar.set_postfix(dict(total_it=it))
        return best_loss
