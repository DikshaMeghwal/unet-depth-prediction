from __future__ import print_function
import zipfile
import os
import pdb
import torch
import h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, transforms, utils

import numpy as np
import torch.nn as nn
import torch

output_height=64
output_width=64

class TransposeDepthInput(object):
    def __call__(self, depth):
        depth = depth.transpose((2, 0, 1))
        depth = torch.from_numpy(depth)
        depth = depth.view(1, depth.shape[0], depth.shape[1], depth.shape[2])
        depth = nn.functional.interpolate(depth, size=(output_height, output_width), mode='bilinear', align_corners=False)
        depth = torch.log(depth)
        return depth[0]

rgb_data_transforms = transforms.Compose([
    transforms.Resize((output_height, output_width)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])

depth_data_transforms = transforms.Compose([
    TransposeDepthInput(),
])

input_for_plot_transforms = transforms.Compose([
    transforms.Resize((output_height, output_width)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])

class NYUDataset(Dataset):
    def __init__(self, filename, type, rgb_transform = None, depth_transform = None):
        f = h5py.File(filename, 'r')
        if type == "training":
            self.images = f['images'][0:1024]
            self.depths = f['depths'][0:1024]
        elif type == "validation":
            self.images = f['images'][1024:1248]
            self.depths = f['depths'][1024:1248]
        elif type == "test":
            self.images = f['images'][1248:]
            self.depths = f['depths'][1248:]
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.transpose((2, 1, 0))
        image = Image.fromarray(image)
        if self.rgb_transform:
            image = self.rgb_transform(image)
        depth = self.depths[idx]
        depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1]))
        depth = depth.transpose((2, 1, 0))
        if self.depth_transform:
            depth = self.depth_transform(depth)
        sample = {'image': image, 'depth': depth}
        return sample
