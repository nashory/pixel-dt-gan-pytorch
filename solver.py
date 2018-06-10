#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''solver.py
'''

import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam
import torchvision.utils as vutils
import torch.nn.functional as F
from progress.bar import Bar as Bar

from tf import TensorBoard
from utils import (
    AverageMeter,
    adjust_pixel_range,
    make_image_grid,
    mkdir_p)


'''helper functions.
'''
def __cuda__(x):
    return x.cuda() if torch.cuda.is_available() else x

def __to_var__(x, volatile=False):
    return Variable(x, volatile=volatile)

def __to_tensor__(x):
    return x.data

class Solver(object):
    def __init__(
        self, 
        config, 
        dataloader,
        gen,
        dis,
        dom):
        
        self.config = config
        self.dataloader = dataloader
        self.epoch = 0
        self.globalIter = 0
        self.prefix = os.path.join('repo', config.expr)
        
        # model
        self.gen = __cuda__(gen)
        self.dis = __cuda__(dis)
        self.dom = __cuda__(dom)
        
        # criterion
        self.l1loss = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()

        # optimizer (support adam optimizer only at the moment.)
        if config.optimizer.lower() in ['adam']:
            betas = (0.5, 0.97)        # GAN is sensitive to the beta value. May be this could be the reason of the training failure.
            self.opt_gen = Adam(
                filter(lambda p: p.requires_grad, self.gen.parameters()), 
                lr=config.lr,
                betas=betas,
                weight_decay=0.0)
            self.opt_dis = Adam(
                filter(lambda p: p.requires_grad, self.dis.parameters()), 
                lr=config.lr,
                betas=betas,
                weight_decay=0.0)
            self.opt_dom = Adam(
                filter(lambda p: p.requires_grad, self.dom.parameters()), 
                lr=config.lr,
                betas=betas,
                weight_decay=0.0)
        elif config.optimizer.lower() in ['sgd']:
            self.opt_gen = optim.RMSprop(
                filter(lambda p: p.requires_grad, self.gen.parameters()),
                lr=config.lr)
            self.opt_dis = optim.RMSprop(
                filter(lambda p: p.requires_grad, self.dis.parameters()),
                lr=config.lr)
            self.opt_dom = optim.RMSprop(
                filter(lambda p: p.requires_grad, self.dom.parameters()),
                lr=config.lr)

        # tensorboard for visualization
        if config.use_tensorboard:
            self.tb = TensorBoard(os.path.join(self.prefix, 'tb'))
            self.tb.initialize()
    
    def solve(self, epoch):
        '''
        solve for 1 epoch.
        Args:
            xr: raw, target model image.
            xc: clean, relavant product image.
            xi: clean, irrelavant product image.
        '''
        batch_timer = AverageMeter()
        data_timer = AverageMeter()
        since = time.time()
        bar = Bar('[PixelDtGan] Training ...', max=len(self.dataloader))
      
        for batch_index, x in enumerate(self.dataloader):
            self.globalIter = self.globalIter + 1
            # measure data loading time
            data_timer.update(time.time() - since)

            # convert to cuda, variable
            xr = x['raw']
            xc = x['clean']
            xi = x['irre']
            xr = __to_var__(__cuda__(xr))
            xc = __to_var__(__cuda__(xc))
            xi = __to_var__(__cuda__(xi))
            
            # xr_test for test with fixed input.
            if self.globalIter == 1:
                xr_test = xr.clone()
                xc_test = xc.clone()
                
            # zero gradients.
            self.gen.zero_grad()
            self.dis.zero_grad()
            self.dom.zero_grad()

            '''update discriminator. (dis, dom)
            '''
            since = time.time()
            # train dis (real/fake)
            dl_xc = self.dis(xc)                             # real, relavant
            dl_xi = self.dis(xi)                             # real, irrelavant
            xc_tilde = self.gen(xr)
            dl_xc_tilde = self.dis(xc_tilde.detach())        # fake (detach)
            real_label = dl_xc.clone().fill_(1).detach()
            fake_label = dl_xc.clone().fill_(0).detach()
            loss_dis = self.mse(dl_xc, real_label) + self.mse(dl_xi, real_label) + self.mse(dl_xc_tilde, fake_label)
            
            # train dom (associated-pair/non-associated-pair)
            xp_ass = torch.cat((xr, xc), dim=1)
            xp_noass = torch.cat((xr, xi), dim=1)
            xp_tilde = torch.cat((xr, xc_tilde.detach()), dim=1)
            dl_xp_ass = self.dom(xp_ass)
            dl_xp_noass = self.dom(xp_noass)
            dl_xp_tilde = self.dom(xp_tilde)
            loss_dom = self.mse(dl_xp_ass, real_label) + self.mse(dl_xp_noass, fake_label) + self.mse(dl_xp_tilde, fake_label)
            loss_D_total = 0.5 * (loss_dis + loss_dom)
            loss_D_total.backward()
            self.opt_dis.step()
            self.opt_dom.step()
            
            '''update generator. (gen)
            '''
            # train gen (real/fake)
            gl_xc_tilde = self.dis(xc_tilde)
            gl_xp_tilde = self.dom(xp_tilde)
            loss_gen = self.mse(gl_xc_tilde, real_label)  + self.mse(gl_xp_tilde, real_label)
            loss_gen.backward()
            self.opt_gen.step()
            
            # measure batch process time
            batch_timer.update(time.time() - since)

            # print log
            log_msg = '\n[Epoch:{EPOCH:}][Iter:{ITER:}][lr:{LR:}] Loss_dis:{LOSS_DIS:.3f} | Loss_dom:{LOSS_DOM:.3f} | Loss_gen:{LOSS_GEN:.3f} | eta:(data:{DATA_TIME:.3f}),(batch:{BATCH_TIME:.3f}),(total:{TOTAL_TIME:})' \
            .format(
                EPOCH=epoch+1,
                ITER=batch_index+1,
                LR=self.config.lr,
                LOSS_DIS=loss_dis.data.sum(),
                LOSS_DOM=loss_dom.data.sum(),
                LOSS_GEN=loss_gen.data.sum(),
                DATA_TIME=data_timer.val,
                BATCH_TIME=batch_timer.val,
                TOTAL_TIME=bar.elapsed_td)
            print(log_msg)
            bar.next()

            # visualization
            if self.config.use_tensorboard:
                self.tb.add_scalar('data/loss_dis', float(loss_dis.data.cpu()), self.globalIter)
                self.tb.add_scalar('data/loss_dom', float(loss_dom.data.cpu()), self.globalIter)
                self.tb.add_scalar('data/loss_gen', float(loss_gen.data.cpu()), self.globalIter)

                if self.globalIter % self.config.save_image_every == 0:
                    xall = torch.cat((xc_tilde, xc, xr), dim=0)
                    xall = adjust_pixel_range(xall, range_from=[-1,1], range_to=[0,1])
                    self.tb.add_image_grid('grid/output', 8, xall.cpu().data, self.globalIter)

                    xc_tilde_test = self.gen(xr_test)
                    xall_test = torch.cat((xc_tilde_test, xc_test, xr_test), dim=0)
                    xall_test = adjust_pixel_range(xall_test, range_from=[-1,1], range_to=[0,1])
                    self.tb.add_image_grid('grid/output_fixed', 8, xall_test.cpu().data, self.globalIter)
                    
                    # save image as png.
                    mkdir_p(os.path.join(self.prefix, 'image'))
                    image = make_image_grid(xc_tilde_test.cpu().data, 5)
                    image = F.upsample(image.unsqueeze(0), size=(800, 800), mode='bilinear').squeeze()
                    filename = 'Epoch_{}_Iter{}.png'.format(self.epoch, self.globalIter)
                    vutils.save_image(image, os.path.join(self.prefix, 'image', filename), nrow=1)

        bar.finish()
            
        


    def save_checkpoint(self):
        print('save checkpoint')

