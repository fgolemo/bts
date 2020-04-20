import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


### OUTPUTS
# NYU:
# (480, 640, 3) uint8 0 255
# (480, 640) float32 0.0 0.035614558
# 100%|██████████| 197/197 [00:08<00:00, 23.95it/s]
# global min/max RGB: 0/255
# global min/max D: 0.0/0.15259021520614624
# mean/std RGB: [82.81281129 69.98840933 67.55189782]/[68.93296759 68.55954934 69.47609191]
# mean/std D: 0.022469453513622284/0.01488421019166708
#
# ===
#
# IN:
# (9600, 640, 3) float32 0.0 0.99607843
# (9600, 640) float32 0.007721065 0.06131075
# 100%|██████████| 50/50 [00:44<00:00,  1.14it/s]
# global min/max RGB: 0.0/255.0
# global min/max D: 0.0/0.12114137411117554
# mean/std RGB: [130.67242 128.60162 125.68408]/[49.410275 50.143192 51.87701 ]
# mean/std D: 0.039533697068691254/0.014748318120837212
#
# ====
#
# Rooms-1, Chairs
# (64, 64, 3) float32 0.05490196 1.0
# (64, 64, 3) float32 0.24313726 0.8745098
# global min/max RGB: 0.0/1.0
# global min/max D: 0.24313725531101227/0.9803921580314636
# mean/std RGB: [0.6774684  0.6279848  0.53581077]/[0.21116914 0.19589706 0.18118107]
# mean/std D: 0.6418006420135498/0.21939150989055634



IN_PATH = "/Users/florian/Downloads/ds1-roomy-chairs-prepped/train/"

rgb_imgs = [x for x in os.listdir(IN_PATH) if "rgb" in x]
depth_imgs = [x for x in os.listdir(IN_PATH) if "depth" in x]

rgb_imgs.sort()
depth_imgs.sort()

print("Rooms:")

# load sample image from NYU
sample_rgb = plt.imread(os.path.join(IN_PATH, rgb_imgs[0]))
print(sample_rgb.shape, sample_rgb.dtype, sample_rgb.min(), sample_rgb.max())

# sample depth
sample_depth = plt.imread(os.path.join(IN_PATH, depth_imgs[0]))
print(sample_depth.shape, sample_depth.dtype, sample_depth.min(),
      sample_depth.max())

## find global min/max/mean/std on color and depth

nyu_mean_rgb = []
nyu_std_rgb = []
nyu_mean_d = []
nyu_std_d = []

nyu_min_rgb = np.array([np.inf])
nyu_max_rgb = np.array([-np.inf])
nyu_min_d = np.array([np.inf])
nyu_max_d = np.array([-np.inf])

for i in trange(len(rgb_imgs)):
    sample_rgb = plt.imread(os.path.join(IN_PATH, rgb_imgs[i]))
    sample_depth = plt.imread(os.path.join(IN_PATH, depth_imgs[i]))

    nyu_min_rgb = np.min([nyu_min_rgb, sample_rgb.min()])
    nyu_max_rgb = np.max([nyu_max_rgb, sample_rgb.max()])
    nyu_min_d = np.min([nyu_min_d, sample_depth.min()])
    nyu_max_d = np.max([nyu_max_d, sample_depth.max()])

    nyu_mean_rgb.append(sample_rgb.mean(axis=(0,1)))
    nyu_std_rgb.append(sample_rgb.std(axis=(0,1)))
    nyu_mean_d.append(sample_depth.mean())
    nyu_std_d.append(sample_depth.std())


print(f"global min/max RGB: {nyu_min_rgb}/{nyu_max_rgb}")
print(f"global min/max D: {nyu_min_d}/{nyu_max_d}")
print(f"mean/std RGB: {np.mean(nyu_mean_rgb, axis=0)}/{np.mean(nyu_std_rgb, axis=0)}")
print(f"mean/std D: {np.mean(nyu_mean_d)}/{np.mean(nyu_std_d)}")

