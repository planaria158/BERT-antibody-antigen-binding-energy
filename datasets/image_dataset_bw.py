import math
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datasets.scFv_dataset import scFv_Dataset

class Image_Dataset_BW(Dataset):
    """
        Dataset class for scFv sequence, Kd data

        Wraps the scFv_Dataset and converts its output 
        into a single-channel 2D image for use in image-based models
        like Vision Transformer, for example

        Args:
            config: dict with configuration parameters
            csv_file_path: path to the csv file
            skiprows: number of rows to skip at the beginning of the file
            inference: if True, the dataset is used for inference
            augment: if True, the dataset is used for training and data augmentation is applied

    """
    def __init__(self, config, img_shape, block_size, csv_file_path, transform=None, skiprows=0, inference=False, regularize=False):  
        super().__init__()
        self.scFv_dataset = scFv_Dataset(config, block_size, csv_file_path, skiprows, inference, regularize)
        self.config = config
        self.block_size = block_size
        self.img_shape = img_shape
        self.transform = transform
        
    def get_vocab_size(self):
        return self.scFv_dataset.vocab_size

    def get_block_size(self):
        return self.scFv_dataset.get_block_size()

    def __len__(self):
        return self.scFv_dataset.__len__()

    def _bin(self, x):
        return format(x, '08b')

    """
        Encodes a channel of the image
        linear tensor is binary encoded (over 8-bits) and reshaped into a 2D tensor

        Args:
            x: tensor of integers
            shape: shape of the image

        Returns:
            tensor: tensor of shape (1, shape[0], shape[1])
                    the tensor will contain 1's and 0's only

    """
    def _encode_channel(self, x, shape):
        d = ''.join([self._bin(i) for i in x.numpy()])
        d = [int(x) for x in d] # turn d into a list of integers, one for each bit
        t = torch.tensor(d[:(shape[0]*shape[1])], dtype=torch.float32) 
        t = t.reshape(shape)
        t = t.unsqueeze(0) # add channel dimension
        return t

    """ Returns image, kd pairs """
    def __getitem__(self, idx):
        dix, kd, name = self.scFv_dataset.__getitem__(idx)
        img = self._encode_channel(dix, self.img_shape) # all values are 0 or 1

        # Normalize image [-1, 1]
        img = (img - 0.5)/0.5

        return img, kd, name