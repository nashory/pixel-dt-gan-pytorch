'''main.py
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import config


def main():
    # print config.
    state = {k: v for k, v in config._get_kwargs()}
    print(state)

    # if use cuda.
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    use_cuda = torch.cuda.is_available()

    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(config.manualSeed)
        torch.backends.cudnn.benchmark = True           # speed up training.

    # data loader
    from dataloader import get_loader
    dataloader = get_loader(config)

    # load model
    from pixel_dt_gan import PixelDtGan
    gen = PixelDtGan(mode='gen')        # gen (encoder / decoder)
    dis = PixelDtGan(mode='dis')        # real / fake discriminator
    dom = PixelDtGan(mode='dom')        # domain discriminator

    print('generator:')
    print(gen)
    print('real/fake discriminator:')
    print(dis)
    print('domain discriminator:')
    print(dom)

    # solver
    from solver import Solver
    solver = Solver(
        config = config,
        dataloader = dataloader,
        gen = gen,
        dis = dis,
        dom = dom)

    # train for N-epochs
    for epoch in range(config.epoch):
        solver.solve(epoch)
    
    print('Congrats! You just finished training Pixel-Dt-Gan.')
    





if __name__=='__main__':
    main()  

