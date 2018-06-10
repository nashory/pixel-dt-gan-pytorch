'''dataloader.py
'''

import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import adjust_pixel_range

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class FashionDomainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            csv_file (string): <raw_id>.<ext>, <clean_id>.<ext>
            root_dir (string): Directory including all raw/clean images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.label_csv = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.length = len(self.label_csv)

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        raw_id = os.path.join(self.root_dir, 'raw', self.label_csv.ix[idx, 0])
        raw_im = Image.open(raw_id)
        clean_id = os.path.join(self.root_dir, 'clean', self.label_csv.ix[idx, 1])
        clean_im = Image.open(clean_id)
        irre_idx = random.randrange(0, self.length)
        while(self.label_csv.ix[idx,0] == self.label_csv.ix[irre_idx,0]):
            irre_idx = random.randrange(0, self.length)
        irre_id = os.path.join(self.root_dir, 'clean', self.label_csv.ix[irre_idx, 1])
        irre_im = Image.open(irre_id)

        if self.transform:
            raw_im = self.transform(raw_im)
            clean_im = self.transform(clean_im)
            irre_im = self.transform(irre_im)
        
        # adjust pixel range [0,255] --> [-1, 1]
        raw_im = adjust_pixel_range(raw_im, [0,1], [-1,1])
        clean_im = adjust_pixel_range(clean_im, [0,1], [-1,1])
        irre_im = adjust_pixel_range(irre_im, [0,1], [-1,1])
        return {'raw':raw_im, 'clean':clean_im, 'irre':irre_im}

class ResizeWithPadding(object):
    def __init__(self, imsize, fill=255):
        self.fill = fill
        self.imsize = imsize

    def __call__(self, x):
        return self.__add_padding__(x, self.imsize)

    def __add_padding__(self, x, imsize):
        w, h = x.size
        new_w = int(w / max(w,h) * imsize)
        new_h = int(h / max(w,h) * imsize)
    
        x = x.resize((new_w, new_h), resample=Image.BILINEAR)
        
        delta_w = imsize - new_w
        delta_h = imsize - new_h
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        x = ImageOps.expand(x, padding, fill=(self.fill, self.fill, self.fill))
        return x

def get_loader(config):
    prepro = []
    #prepro.append(transforms.Resize(config.load_size))
    #prepro.append(transforms.CenterCrop(config.load_size))
    prepro.append(ResizeWithPadding(config.load_size))
    prepro.append(transforms.ToTensor())
    
    transform = transforms.Compose(prepro)

    # dataset.    
    dataset = FashionDomainDataset(
        csv_file = config.csv_file,
        root_dir = config.root_dir,
        transform = transform)
    
    # dataloader.
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = config.batch_size,
        shuffle = True,
        num_workers = config.workers)

    return dataloader




