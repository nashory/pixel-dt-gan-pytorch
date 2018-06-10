import os, sys
from os.path import join
import glob
from PIL import Image
from skimage import io
import numpy as np
import json
import argparse
import random
import argparse
import csv

parser = argparse.ArgumentParser('prepare dataset')
parser.add_argument('--data_root', type=str, default='data/raw')
parser.add_argument('--out_dir', type=str, default='data/prepro')


## parse and save config.
config, _ = parser.parse_known_args()


cnt = 0
valid_ext = ['.jpg', '.png']
os.system('mkdir -p {}/raw'.format(config.out_dir))
os.system('mkdir -p {}/clean'.format(config.out_dir))

csvfile = open('{}/label.csv'.format(config.out_dir), 'wb')
writer = csv.writer(csvfile, delimiter=',')
for filename in glob.glob(os.path.join(config.data_root, '*')):
    flist = os.path.splitext(filename)
    fname = os.path.basename(flist[0])
    fext = flist[1]
    if fext.lower() not in valid_ext:
        continue

    image = Image.open(filename)
    fid = fname.split('_')
    if fid[1] == 'CLEAN0':
        image.save('{}/raw/{}{}'.format(config.out_dir, fid[2], fext))
        writer.writerow([fid[2]+fext, fid[0]+fext])
        
    elif fid[1] == 'CLEAN1':
        image.save('{}/clean/{}{}'.format(config.out_dir, fid[0], fext))
        
    # logging.
    cnt = cnt +1
    print '[' + str(cnt) + '] ' + 'processed @ ' + os.path.join(config.out_dir, fname+'.jpg') 
