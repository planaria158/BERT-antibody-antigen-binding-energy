import math
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datasets.scFv_dataset import scFv_Dataset

#--------------------------------------------------------
# Simple wrapper Dataset to turn output from the scFv dataset
# into a 3-channel BGR image
#--------------------------------------------------------
class CNN_Dataset_BGR(Dataset):
    """
    Emits 2D B&W images and binding energies
    """
    def __init__(self, config, csv_file_path, skiprows=0, inference=False):  
        super().__init__()
        self.scFv_dataset = scFv_Dataset(config, csv_file_path, skiprows, inference)
        self.config = config
        self.img_shape = config['image_shape']
         
        chars = self.scFv_dataset.chars
        groups= ['none', 'nonpolar', 'nonpolar', 'neg', 'neg', 'nonpolar', 'nonpolar', 'pos', 'nonpolar', 'pos', 'nonpolar', 'nonpolar', 'neg', 
                'nonpolar', 'neg', 'pos', 'polar', 'polar', 'nonpolar', 'nonpolar', 'polar', 'none', 'none', 'none']
        
        # for VIT, since the residue encodings are spread over 8-bits, assign encodings to groups that spread across the 8-bits
        group_encodings = { 'none'    : int('00101000', base=2), 
                            'polar'   : int('00110011', base=2),
                            'nonpolar': int('11001100', base=2), 
                            'pos'     : int('01010101', base=2),
                            'neg'     : int('10101010', base=2)} 

        # map encoded sequence to groups
        self.i_to_grp = {self.scFv_dataset.stoi[ch]:group_encodings[i] for ch,i in zip(chars, groups)} 
        print('i_to_grp:', self.i_to_grp)

        # The normalized frequence for each amino acid position in the scFv sequences over the entire clean_3 dataset
        # This fixed-array is 246 elements long.
        self.norm_mutation_freq = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 
                            0.95, 0.95, 0.95, 0.95, 0.95, 0.75, 0.9, 0.9, 0.95, 0.95, 0.95, 0.9, 0.95, 0.95, 0.85, 0.55,
                            0.5, 0.25, 0.35, 0.3, 0.6, 0.45, 0.55, 0.95, 0.8, 0.8, 0.8, 0.9, 0.75, 0.8, 0.6, 0.95, 0.85, 
                            0.35, 0.9, 0.85, 0.15, 0.4, 0.9, 0.3, 0.9, 0.8, 0.5, 0.95, 0.95, 0.95, 0.9, 0.6, 0.6, 0.75, 
                            0.3, 0.85, 0.4, 0.9, 0.95, 0.4, 0.9, 0.95, 0.9, 0.25, 0.75, 0.55, 0.85, 0.25, 0.75, 0.6, 0.65, 
                            0.8, 0.8, 0.95, 0.95, 0.85, 0.8, 0.8, 0.9, 0.7, 0.9, 0.9, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                            0.05, 0.05, 0.05, 0.05])
        
        # promote values to 8-bit value range (255), then round up the norm_mutation_freq_encoded tensor and convert to long
        self.norm_mutation_freq_encoded = self.norm_mutation_freq * 255
        self.norm_mutation_freq_encoded = torch.round(self.norm_mutation_freq_encoded).to(torch.long)
        
    def get_vocab_size(self):
        return self.scFv_dataset.vocab_size

    def get_block_size(self):
        return self.config['block_size']

    def __len__(self):
        return self.scFv_dataset.__len__()

    def _bin(self, x):
        return format(x, '08b')

    def _encode_channel(self, x, shape=(48,48)):
        d = ''.join([self._bin(x[i]) for i in x.numpy()])
        # turn d into a list of integers, one for each bit
        d = [int(x) for x in d]    
        t = torch.tensor(d[:(shape[0]*shape[1])], dtype=torch.float32) # this is for 46,46 matrix
        t = t.reshape(shape)
        return t

    """ Returns image, kd pairs used for CNN training """
    def __getitem__(self, idx):
        dix, kd = self.scFv_dataset.__getitem__(idx)

        # The residue encoding channel
        ch_1 = self._encode_channel(dix, self.img_shape)

        # The residue group encoding channel
        dix_grp = torch.tensor([self.i_to_grp[i] for i in dix.numpy().tolist()], dtype=torch.long)
        ch_2 = self._encode_channel(dix_grp, self.img_shape)

        # The mutation frequency channel
        # anything not an amino acid gets a zero.
        ch3_in = torch.zeros_like(dix)
        # First aa is always position 1 (0 is a CLS token)
        ch3_in[1:len(self.norm_mutation_freq_encoded)+1] = self.norm_mutation_freq_encoded
        ch_3 = self._encode_channel(ch3_in)

        # stack the 3 channels into an bgr image
        bgr_img = torch.stack((ch_1, ch_2, ch_3), dim=0)

        return bgr_img, kd