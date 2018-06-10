'''layers.py
'''

import torch
import torch.nn as nn

class Flatten(nn.Module):
    '''nn.Flatten layer in Torch7.
    '''
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
    def __repr__(self):
        return self.__class__.__name__

class Reshape(nn.Module):
    '''nn.Reshape in Torch7.
    '''
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(self.shape)
    def __repr__(self):
        return self.__class__.__name__ + ' (reshape to size: {})'.format(" ".join(str(x) for x in self.shape))

class Identity(nn.Module):
    '''nn.Identity in Torch7.
    '''
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    def __repr__(self):
        return self.__class__.__name__ + ' (skip connection)'


