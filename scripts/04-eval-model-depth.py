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
from tqdm import tqdm

from bts.bts_dataloader_in import DataLoadPreprocessIN, BtsDataLoaderIN
from bts.bts_model import BtsModel
# from bts.models.bts_nyu_v2_pytorch_densenet161.bts_nyu_v2_pytorch_densenet161 import BtsModel
from bts.test_imgs import get_all_imgs, get_img, PATH


plasma = plt.get_cmap('plasma')
greys = plt.get_cmap('Greys')


class args:
    bts_size = 512
    model_name = "bts_inet_pytorch_train"
    encoder = "densenet121_bts"
    max_depth = 50
    checkpoint_path = "../bts/models/bts_inet_pytorch_train/model.pth"
    input_height = 128
    input_width = 128
    dataset = "inet"
    mode = "test"
    fix_first_conv_blocks = False
    fix_first_conv_block = False
    smol = True
    data_path = "/data/fgolemo/intnet-bedrooms-png/"
    gt_path = "/data/fgolemo/intnet-bedrooms-png/"
    filenames_file = "../train_test_inputs/interiornet_test_files_with_gt.txt"
    distributed = False
    batch_size = 10
    num_threads = 8

dl = BtsDataLoaderIN(args, "train")

import torch
from torch.autograd import Variable

model = BtsModel(params=args)
model = torch.nn.DataParallel(model).to(torch.device("cuda:0"))

checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()

d_min = []
d_max = []
d_mean = []
d_std = []

for step, sample_batched in enumerate(tqdm(dl.data)):
    input_image = sample_batched["image"].to(torch.device("cuda:0"))

    # # Normalize image
    #
    with torch.no_grad():
        focal = torch.tensor([600]).to(torch.device("cuda:0"))
        # Predict
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_cropped = model(input_image, focal)

    depth = depth_cropped.cpu().numpy()
    d_min.append(np.min(depth))
    d_max.append(np.max(depth))
    d_mean.append(np.mean(depth))
    d_std.append(np.std(depth))

    if step == 100:
        break

print (f"Global depth min/max: {np.min(d_min)}/{np.max(d_max)}\n"
       f"Global depth mean mean/mean std: {np.mean(d_mean)}/{np.mean(d_std)}")