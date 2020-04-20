import time

import numpy as np
import torch
from skimage.transform import rescale
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random

from tqdm import trange, tqdm


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms_rooms(mode):
    return transforms.Compose([ToTensorIN(mode=mode)])


class BtsDataLoaderRooms(object):

    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocessRooms(
                args, mode, transform=preprocessing_transforms_rooms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(
                self.training_samples,
                args.batch_size,
                shuffle=(self.train_sampler is None),
                num_workers=args.num_threads,
                pin_memory=True,
                sampler=self.train_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocessRooms(
                args, mode, transform=preprocessing_transforms_rooms(mode))
            self.data = DataLoader(
                self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'
                  .format(mode))


class DataLoadPreprocessRooms(Dataset):

    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        with open(args.filenames_file, 'r') as f:
            self.filenames = f.readlines()

        print (f"read {len(self.filenames)} lines from {args.filenames_file}")

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensorIN
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])

        if self.mode == 'train':
            image_path = os.path.join(self.args.data_path,
                                      "./" + sample_path.split()[0])
            depth_path = os.path.join(self.args.gt_path,
                                      "./" + sample_path.split()[1])

            image = np.array(Image.open(image_path),dtype=np.float32)
            image /= 255
            depth_gt = np.array(Image.open(depth_path),dtype=np.float32)
            depth_gt = depth_gt[:,:,0:1]/255

            # depth_gt = np.expand_dims(depth_gt, axis=2)
            #TODO: last implementation had depth divided by 100 here

            # image, depth_gt = self.random_crop(image, depth_gt,
            #                                    self.args.input_height,
            #                                    self.args.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)



            # print(image.shape, image.dtype, image.min(), image.max(),
            #       image.mean())
            # print(depth_gt.shape, depth_gt.dtype, depth_gt.min(),
            #       depth_gt.max(), depth_gt.mean())
            # print(focal)

            sample = {'image': image, 'depth': depth_gt, 'focal': focal}

        else:
            data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            sample = {'image': image, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)

        return sample

    # def random_crop(self, img, depth, height, width):
    #     assert img.shape[0] >= height
    #     assert img.shape[1] >= width
    #     assert img.shape[0] == depth.shape[0]
    #     assert img.shape[1] == depth.shape[1]
    #     x = random.randint(0, img.shape[1] - width)
    #     y = random.randint(0, img.shape[0] - height)
    #     img = img[y:y + height, x:x + width, :]
    #     depth = depth[y:y + height, x:x + width, :]
    #     return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image**gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)  # fixme
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensorIN(object):

    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.6774684,  0.6279848,  0.53581077], std=[0.21116914, 0.19589706, 0.18118107])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        depth = self.to_tensor(depth)
        return {'image': image, 'depth': depth, 'focal': focal}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(
                type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


if __name__ == '__main__':

    # class args:
    #     smol = True
    #     dataset = "IN"
    #     input_width = 128
    #     input_height = 128
    #     # data_path = "/Users/florian/intnet-bedrooms-png-sample/"
    #     # gt_path = "/Users/florian/intnet-bedrooms-png-sample/"
    #     data_path = "/data/fgolemo/intnet-bedrooms-png/"
    #     gt_path = "/data/fgolemo/intnet-bedrooms-png/"
    #     filenames_file = "../train_test_inputs/interiornet_train_files_with_gt.txt"
    #
    # ds = DataLoadPreprocessIN(args, "train")
    # print(len(ds))
    # start = time.time()
    # for i in trange(1000):
    #     a = ds[i]
    # diff = time.time() - start

    class args:
        dataset="Rooms"
        # input_width = 128
        # input_height = 128
        data_path = os.path.expanduser("~/Downloads/ds1-roomy-chairs-prepped/train")
        gt_path = os.path.expanduser("~/Downloads/ds1-roomy-chairs-prepped/train")
        filenames_file = "../train_test_inputs/roomychairs1_train_files_with_gt.txt"
        multiprocessing_distributed = False
        distributed = False
        batch_size = 4
        num_threads = 1

    ds = BtsDataLoaderRooms(args, "train")
    start = time.time()
    for sample_batched in tqdm(ds.data):
        pass
    diff = time.time() - start
