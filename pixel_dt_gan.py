
# -*- coding: utf-8 -*-

'''pixel-dt-gan.py
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from layers import (
    Identity,
    Reshape,
    Flatten,)

'''helper func.
'''
def __cuda__(x):
    return x.cuda() if torch.cuda.is_available() else x

def __load_weights_from__(module_dict, load_dict, modulenames):
    for modulename in modulenames:
        module = module_dict[modulename]
        print('loaded weights from module "{}" ...'.format(modulename))
        module.load_state_dict(load_dict[modulename])


'''PixelDtGan
'''
class PixelDtGan(nn.Module):
    def __init__(
        self,
        mode):
        
        super(PixelDtGan, self).__init__()

        self.mode = mode
        self.module_list = nn.ModuleList()
        self.module_dict = {}
        self.end_points = {}

        if mode.lower() == 'gen':
            # endpoint: encoder
            layers = []
            layers = self.__add_conv_layer__(layers, 3, 128, k_size=4, stride=2, pad=1, act='leakyrelu', bn=False)
            layers = self.__add_conv_layer__(layers, 128, 256, k_size=4, stride=2, pad=1, act='leakyrelu', bn=True)
            layers = self.__add_conv_layer__(layers, 256, 512, k_size=4, stride=2, pad=1, act='leakyrelu', bn=True)
            layers = self.__add_conv_layer__(layers, 512, 1024, k_size=4, stride=2, pad=1, act='leakyrelu', bn=True)
            layers = self.__add_conv_layer__(layers, 1024, 64, k_size=4, stride=2, pad=1, act='leakyrelu', bn=True)
            self.__register_module__('enc', layers)

            # endpoint: decoder
            layers = []
            layers = self.__add_deconv_layer__(layers, 64, 1024, k_size=4, stride=2, pad=1, act='relu', bn=True)
            layers = self.__add_deconv_layer__(layers, 1024, 512, k_size=4, stride=2, pad=1, act='relu', bn=True)
            layers = self.__add_deconv_layer__(layers, 512, 256, k_size=4, stride=2, pad=1, act='relu', bn=True)
            layers = self.__add_deconv_layer__(layers, 256, 128, k_size=4, stride=2, pad=1, act='relu', bn=True)
            layers = self.__add_deconv_layer__(layers, 128, 3, k_size=4, stride=2, pad=1, act='tanh', bn=False)
            self.__register_module__('dec', layers)

        elif mode.lower() == 'dis':
            # endpoint: real/fake discriminator
            layers = []
            layers = self.__add_conv_layer__(layers, 3, 128, k_size=4, stride=2, pad=1, act='leakyrelu', bn=False)
            layers = self.__add_conv_layer__(layers, 128, 256, k_size=4, stride=2, pad=1, act='leakyrelu', bn=True)
            layers = self.__add_conv_layer__(layers, 256, 512, k_size=4, stride=2, pad=1, act='leakyrelu', bn=True)
            layers = self.__add_conv_layer__(layers, 512, 1024, k_size=4, stride=2, pad=1, act='leakyrelu', bn=True)
            layers = self.__add_conv_layer__(layers, 1024, 1, k_size=4, stride=1, pad=0, act='sigmoid', bn=False)
            self.__register_module__('dis', layers)

        elif mode.lower() == 'dom':
            # endpoint: domain discriminator
            layers = []
            layers = self.__add_conv_layer__(layers, 6, 128, k_size=4, stride=2, pad=1, act='leakyrelu', bn=False)
            layers = self.__add_conv_layer__(layers, 128, 256, k_size=4, stride=2, pad=1, act='leakyrelu', bn=True)
            layers = self.__add_conv_layer__(layers, 256, 512, k_size=4, stride=2, pad=1, act='leakyrelu', bn=True)
            layers = self.__add_conv_layer__(layers, 512, 1024, k_size=4, stride=2, pad=1, act='leakyrelu', bn=True)
            layers = self.__add_conv_layer__(layers, 1024, 1, k_size=4, stride=1, pad=0, act='sigmoid', bn=False)
            self.__register_module__('dom', layers)
            
    def __add_deconv_layer__(self, layers, in_c, out_c, k_size, stride, pad, act, bn):
        layers.append(nn.ConvTranspose2d(in_c, out_c, k_size, stride, pad))
        if bn:
            layers.append(nn.BatchNorm2d(out_c))
        if act == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif act == 'relu':
            layers.append(nn.ReLU())
        elif act == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif act == 'tanh':
            layers.append(nn.Tanh())
        return layers

    def __add_conv_layer__(self, layers, in_c, out_c, k_size, stride, pad, act, bn):
        layers.append(nn.Conv2d(in_c, out_c, k_size, stride, pad))
        if bn:
            layers.append(nn.BatchNorm2d(out_c))
        if act == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif act == 'relu':
            layers.append(nn.ReLU())
        elif act == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif act == 'tanh':
            layers.append(nn.Tanh())
        return layers
        
    def __register_module__(self, modulename, module):
            if isinstance(module, list) or isinstance(module, tuple):
                module = nn.Sequential(*module)
            self.module_list.append(module)
            self.module_dict[modulename] = module

    def __forward_and_save__(self, x, modulename):
        module = self.module_dict[modulename]
        x = module(x)
        self.end_points[modulename] = x
        return x

    def forward(self, x):
        if self.mode.lower() == 'gen':
            x = self.__forward_and_save__(x, 'enc')
            x = self.__forward_and_save__(x, 'dec')
        elif self.mode.lower() == 'dis':
            x = self.__forward_and_save__(x, 'dis')
        else:
            assert self.mode.lower() == 'dom' 
            x = self.__forward_and_save__(x, 'dom')
        return x
            



        
 
