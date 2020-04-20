import os
import shutil

import numpy as np
import glob
from pathlib import Path

from tqdm import trange

DIR_IN = os.path.expanduser("~/Downloads/ds2-roomy-sofas")
DIR_OUT = os.path.expanduser("~/Downloads/ds2-roomy-sofas-prepped")

NO_OF_CHAIRS = 4500

# possible focal lengths (FOV = 60):
# 55 - from FL = 1/(tan(OV/2) and from Unity and setting the camera to be a physical camera
# 600 - just because we have this in interiornet too

if not os.path.isdir(DIR_OUT):
    os.mkdir(DIR_OUT)

if not os.path.isdir(f"{DIR_OUT}/canonical"):
    os.mkdir(f"{DIR_OUT}/canonical")

if not os.path.isdir(f"{DIR_OUT}/train"):
    os.mkdir(f"{DIR_OUT}/train")

if not os.path.isdir(f"{DIR_OUT}/alternative"):
    os.mkdir(f"{DIR_OUT}/alternative")

file_list_normal = []
file_list_alternative = []
file_list_canonical = []

output_lines = []
counter = 0

for root_dir in [x for x in os.listdir(DIR_IN) if "renderings" in x]:
    parts = root_dir.split("-")
    num = int(parts[-1])
    for fidx in trange(1, NO_OF_CHAIRS):
        full_path = f"{DIR_IN}/{root_dir}/object-{fidx}-azimuth-*-rgb.png"
        files = glob.glob(full_path)

        non_canonical_files = []
        canon = None

        for f in files:
            f_parts = Path(f).stem.split("-")
            azimuth = f_parts[3]
            if not azimuth == "canonical":
                azimuth = np.around(float(azimuth) - 270, 3)
            out_path = f"scene_{counter:05d}_azimuth_{azimuth}_rgb.png"

            f_d = f.replace("rgb", "depth")
            out_path_d = out_path.replace("rgb", "depth")

            if azimuth == "canonical":
                canon = (f, out_path, f_d, out_path_d)
            else:
                non_canonical_files.append((f, out_path, f_d, out_path_d))

        if len(non_canonical_files) != 2:
            continue

        coin = np.random.rand()
        if coin < .5:
            train = non_canonical_files[0]
            alt = non_canonical_files[1]
        else:
            train = non_canonical_files[1]
            alt = non_canonical_files[0]

        # move train
        shutil.copy2(train[0], f"{DIR_OUT}/train/{train[1]}")
        shutil.copy2(train[2], f"{DIR_OUT}/train/{train[3]}")
        file_list_normal.append(f"{train[1]} {train[3]} 55\n")

        # move alt
        shutil.copy2(alt[0], f"{DIR_OUT}/alternative/{alt[1]}")
        shutil.copy2(alt[2], f"{DIR_OUT}/alternative/{alt[3]}")
        file_list_alternative.append(f"{alt[1]} {alt[3]} 55\n")

        # move canonical
        shutil.copy2(canon[0], f"{DIR_OUT}/canonical/{canon[1]}")
        shutil.copy2(canon[2], f"{DIR_OUT}/canonical/{canon[3]}")
        file_list_canonical.append(f"{canon[1]} {canon[3]} 55\n")

        counter += 1

        # if counter == 5:
        #     break
    # break

with open(f"{DIR_OUT}/roomysofas2_train_files_with_gt.txt", "w") as f:
    f.writelines(file_list_normal)

with open(f"{DIR_OUT}/roomysofas2_alt_files_with_gt.txt", "w") as f:
    f.writelines(file_list_alternative)

with open(f"{DIR_OUT}/roomysofas2_canon_files_with_gt.txt", "w") as f:
    f.writelines(file_list_canonical)


