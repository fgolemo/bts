from __future__ import absolute_import, division, print_function

import os
import sys
import time
import argparse
import numpy as np

# Computer Vision
import cv2
from matplotlib import colors
from scipy import ndimage
from skimage.transform import resize, rescale

# Visualization
import matplotlib.pyplot as plt

# from bts.bts_model import BtsModel
from tqdm import tqdm

from bts.bts_dataloader_in import BtsDataLoaderIN
from bts.models.bts_nyu_v2_pytorch_densenet161.bts_nyu_v2_pytorch_densenet161 import BtsModel
from bts.test_imgs import get_all_imgs, get_img, PATH


plasma = plt.get_cmap('plasma')
greys = plt.get_cmap('Greys')


class args:
    smol = True
    # dataset = "IN"
    data_path = "/Users/florian/intnet-bedrooms-png-sample/"
    gt_path = "/Users/florian/intnet-bedrooms-png-sample/"
    bts_size = 512
    model_name = "bts_inet_pytorch_train"
    encoder = "densenet121_bts"
    max_depth = 50
    checkpoint_path = "../bts/models/bts_inet_pytorch_train/model2.pth"
    input_height = 128
    input_width = 128
    dataset = "inet"
    mode = "test"
    fix_first_conv_blocks = False
    fix_first_conv_block = False
    filenames_file = "../train_test_inputs/interiornet_train_files_with_gt.txt"
    multiprocessing_distributed = False
    distributed = False
    batch_size = 4
    num_threads = 1

# class args:
#     bts_size = 512
#     model_name = "bts_nyu_v2_pytorch_densenet161"
#     encoder = "densenet161_bts"
#     max_depth = 50
#     checkpoint_path = "../bts/models/bts_nyu_v2_pytorch_densenet161/model"
#     input_height = 128
#     input_width = 128
#     dataset = "inet"
#     mode = "test"
#     fix_first_conv_blocks = False
#     fix_first_conv_block = False


import torch
from torch.autograd import Variable

model = BtsModel(params=args)
model = torch.nn.DataParallel(model)

checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()

ds = BtsDataLoaderIN(args, "train")

all_depths = []

for idx,data in enumerate(tqdm(ds.data)):
    with torch.no_grad():
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_cropped = model(data["image"], data["focal"])

    all_depths.append(depth_cropped.numpy())

print (np.mean(all_depths), np.std(all_depths), np.min(all_depths), np.max(all_depths))
