import math
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datasets.scFv_dataset import scFv_Dataset

#--------------------------------------------------------
# Simple wrapper Dataset to turn output from the scFv dataset
# into a B&W image for use in a CNN model
#--------------------------------------------------------
class CNN_Dataset(Dataset):
    """
    Emits 2D B&W images and binding energies
    """
    def __init__(self, config, csv_file_path, img_shape=(46,46), skiprows=0, inference=False):  
        super().__init__()
        self.scFv_dataset = scFv_Dataset(config, csv_file_path, skiprows, inference)
        self.config = config
        self.img_shape = img_shape
        
    def get_vocab_size(self):
        return self.scFv_dataset.vocab_size

    def get_block_size(self):
        return self.config['block_size']

    def __len__(self):
        return self.scFv_dataset.__len__()

    def _bin(self, x):
        return format(x, '08b')

    def _make_img(self, x, shape=(46,46)):
        assert(x.shape[0] == shape[0]*shape[1])
        d = ''.join([self._bin(x[i]) for i in x.numpy()])
        # turn d into a list of integers, one for each bit
        d = [int(x) for x in d]    
        t = torch.tensor(d[:(shape[0]*shape[1])], dtype=torch.float32) # this is for 46,46 matrix
        t = t.reshape(shape)
        return t

    """ Returns image, kd pairs used for CNN training """
    def __getitem__(self, idx):
        dix, kd = self.scFv_dataset.__getitem__(idx)
        img = self._make_img(dix, self.img_shape)
        return img, kd
