import math
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datasets.scFv_dataset import scFv_Dataset

#--------------------------------------------------------------
# Simple wrapper Dataset to turn output from the scFv dataset
# into a 3-channel image for use in image-based models
#--------------------------------------------------------------
class Image_Dataset_BGR(Dataset):
    """
    Emits 2D B&W images and binding energies
    """
    def __init__(self, config, img_shape, block_size, csv_file_path, transform=None, skiprows=0, inference=False, regularize=False):  
        super().__init__()
        self.scFv_dataset = scFv_Dataset(config, block_size, csv_file_path, skiprows, inference, regularize)
        self.config = config
        self.block_size = block_size
        self.img_shape = img_shape
        self.transform = transform
         
        chars = self.scFv_dataset.chars
        groups= ['none', 'nonpolar', 'nonpolar', 'neg', 'neg', 'nonpolar', 'nonpolar', 'pos', 'nonpolar', 'pos', 'nonpolar', 'nonpolar', 'neg', 
                'nonpolar', 'neg', 'pos', 'polar', 'polar', 'nonpolar', 'nonpolar', 'polar', 'none', 'none', 'none']
        
        # I manually created these group encodings.  
        group_encodings = { 'none'    : int('11001100', base=2), 
                            'polar'   : int('00110011', base=2),
                            'nonpolar': int('01100110', base=2), 
                            'pos'     : int('01010101', base=2),
                            'neg'     : int('10101010', base=2)} 
        
        print('group_encodings:', group_encodings)

        # map encoded sequence to groups
        self.i_to_grp = {self.scFv_dataset.stoi[ch]:group_encodings[i] for ch,i in zip(chars, groups)} 

        # The relative mutation frequence for each amino acid position in the scFv sequences over the entire clean_3 dataset
        # This fixed-array is 241 elements long. (I clipped off the last 5 residues from the 246 residue sequences for the VIT model)
        self.rel_mutation_freq = torch.tensor([ 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.7778,
                                                0.9444, 0.9444, 1.0000, 1.0000, 1.0000, 0.9444, 1.0000, 1.0000, 0.8889,
                                                0.5556, 0.5000, 0.2222, 0.3333, 0.2778, 0.6111, 0.4444, 0.5556, 1.0000,
                                                0.8333, 0.8333, 0.8333, 0.9444, 0.7778, 0.8333, 0.6111, 1.0000, 0.8889,
                                                0.3333, 0.9444, 0.8889, 0.1111, 0.3889, 0.9444, 0.2778, 0.9444, 0.8333,
                                                0.5000, 1.0000, 1.0000, 1.0000, 0.9444, 0.6111, 0.6111, 0.7778, 0.2778,
                                                0.8889, 0.3889, 0.9444, 1.0000, 0.3889, 0.9444, 1.0000, 0.9444, 0.2222,
                                                0.7778, 0.5556, 0.8889, 0.2222, 0.7778, 0.6111, 0.6667, 0.8333, 0.8333,
                                                1.0000, 1.0000, 0.8889, 0.8333, 0.8333, 0.9444, 0.7222, 0.9444, 0.9444,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]) 
        
        self.mutation_freq_encoded = torch.round(self.rel_mutation_freq * 255).to(torch.long)
        print('min mutation freq:', torch.min(self.mutation_freq_encoded), 'max mutation freq:', torch.max(self.mutation_freq_encoded))
        print('a sample of some mutation_freq_encoded values:', self.mutation_freq_encoded[60:70])
        
    def get_vocab_size(self):
        return self.scFv_dataset.vocab_size

    def get_block_size(self):
        return self.config['block_size']

    def __len__(self):
        return self.scFv_dataset.__len__()

    def _bin(self, x):
        return format(x, '08b')

    # Input is a tensor of integers.
    # Output is a tensor of binary encoded input tensor (over 8 bits)
    # and reshaped into a [1, shape[0], shape[1]]) tensor
    # everything that comes out of this method is 1's and 0's only
    def _encode_channel(self, x, shape):
        d = ''.join([self._bin(val) for val in x])
        d = [int(x) for x in d] # turn d into a list of integers, one for each bit
        t = torch.tensor(d[:(shape[0]*shape[1])], dtype=torch.float32) # this is for shape matrix
        t = t.reshape(shape)
        return t

    """ Returns image, Kd pairs """
    def __getitem__(self, idx):

        dix, kd, name = self.scFv_dataset.__getitem__(idx)

        # channel 1: the residue encoding channel
        ch_1 = self._encode_channel(dix, self.img_shape)

        # channel 2: the residue group encoding channel
        dix_grp = torch.tensor([self.i_to_grp[i] for i in dix.numpy().tolist()], dtype=torch.long)
        ch_2 = self._encode_channel(dix_grp, self.img_shape)

        # channel 3: the mutation frequency channel
        ch3_in = torch.zeros_like(dix)
        # First aa is always position 1 (0 is a CLS token)
        ch3_in[1:len(self.mutation_freq_encoded)+1] = self.mutation_freq_encoded
        ch_3 = self._encode_channel(ch3_in, self.img_shape)

        # stack the 3 channels into a bgr image
        bgr_img = torch.stack((ch_1, ch_2, ch_3), dim=0)

        # Normalize image [-1, 1]
        bgr_img = (bgr_img - 0.5)/0.5

        return bgr_img, kd, name