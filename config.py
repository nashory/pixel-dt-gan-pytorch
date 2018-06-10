'''config.py
'''

import time
import argparse

# helper func.
def str2bool(v):
    return v.lower() in ('true', '1')

# Parser
parser = argparse.ArgumentParser('pixel-dt-gan')

# Common options.
parser.add_argument('--gpu_id', 
                    default='4', 
                    type=str, 
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--root_dir',
                    default='../data/prepro', 
                    type=str) 
parser.add_argument('--csv_file',
                    default='label.csv', 
                    type=str) 
parser.add_argument('--manualSeed', 
                    type=int, 
                    default=int(time.time()), 
                    help='manual seed')
parser.add_argument('--expr', 
                    default='devel', 
                    type=str, 
                    help='experiment name')
parser.add_argument('--workers',
                    type=int,
                    default=8)
# hyperparameters
parser.add_argument('--batch_size',
                    type=int,
                    default=24)
parser.add_argument('--load_size',
                    type=int,
                    default=64)
parser.add_argument('--epoch',
                    type=int,
                    default=40)
parser.add_argument('--lr',
                    type=float,
                    default=0.0002)
parser.add_argument('--optimizer', 
                    default='adam', 
                    type=str)

# visualization
parser.add_argument('--use_tensorboard', 
                    default=True, 
                    type=bool)
parser.add_argument('--save_image_every',
                    type=int,
                    default=50)


## parse and save config.
config, _ = parser.parse_known_args()



