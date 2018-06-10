
'''utils.py
'''

from __future__ import print_function, absolute_import

import errno
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.utils as vutils
from torch.autograd import Variable

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'compute_precision_top_k']

def compute_precision_top_k(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def make_image_grid(x, nrow):
    if pow(nrow,2) < x.size(0):
        grid = vutils.make_grid(
            x[:nrow*nrow],
            nrow=nrow,
            padding=0,
            normalize=False,
            scale_each=True)
    else:
        grid = torch.FloatTensor(nrow*nrow, x.size(1), x.size(2), x.size(3)).uniform_()
        grid[:x.size(0)] = x
        grid = vutils.make_grid(
            grid,
            nrow=nrow,
            padding=0,
            normalize=False,
            scale_each=True)
    return grid


def adjust_pixel_range(
    x,
    range_from=[0,1],
    range_to=[-1,1]):
    '''
    adjust pixel range from <range_from> to <range_to>.
    '''
    if (range_from[0] == range_to[0]) and (range_from[1] == range_to==[1]):
        return x
    else:
        scale = float(range_to[1]-range_to[0])/float(range_from[1]-range_from[0])
        bias = range_to[0]-range_from[0]*scale
        x = x.mul(scale).add(bias)
        return x.clamp(range_to[0], range_to[1])


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




