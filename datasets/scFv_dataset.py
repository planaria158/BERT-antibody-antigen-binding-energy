import math
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class scFv_Dataset(Dataset):
    """
        Dataset class for scFv sequence, Kd data

        Args:
            config: dict with configuration parameters
            csv_file_path: path to the csv file
            skiprows: number of rows to skip at the beginning of the file
            inference: if True, the dataset is used for inference
            augment: if True, the dataset is used for training and data augmentation is applied
    """
    def __init__(self, config, csv_file_path, skiprows=0, inference=False, regularize=False):  
        super().__init__()
        self.config = config
        self.inference = inference
        self.regularize = regularize # sequence flipping etc...
        print('reading the data from:', csv_file_path)
        self.df = pd.read_csv(csv_file_path, skiprows=skiprows)
        
        # 20 naturally occuring amino acids in human proteins plus MASK token, 
        # 'X' is a special token for unknown amino acids, CLS token is for classification, and PAD for padding
        self.chars = ['CLS', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                      'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X', 'MASK', 'PAD']
        self.first_aa = self.chars[1]
        self.last_aa = self.chars[20]
        print('vocabulary:', self.chars)
        data_size, vocab_size = self.df.shape[0], len(self.chars)
        print('data has %d rows, %d vocab size (unique).' % (data_size, vocab_size))

        # encoding and decoding residues
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.vocab_size = vocab_size

    def get_random_aa(self):
        return self.chars[np.random.randint(1, 21)]
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config['block_size']

    def __len__(self):
        return self.df.shape[0] 

    def __getitem__(self, idx):
        """ 
            Returns sequence, affinity pairs
        """
        seq = self.df.loc[idx, 'sequence_a']
        affinity = self.df.loc[idx, 'Kd'] if self.inference == False else 0.0
        assert not math.isnan(affinity), 'affinity is nan'
        assert affinity >= 0.0, 'affinity cannot be negative'

        # get a randomly located block_size substring from the sequence
        if len(seq) <= self.config['block_size']:
            chunk = seq
        else:
            start_idx = np.random.randint(0, len(seq) - (self.config['block_size']))
            chunk = seq[start_idx:start_idx + self.config['block_size']]

        # encode the string
        dix = torch.tensor([self.stoi[s] for s in chunk], dtype=torch.long)

        # some sequence-level regularization & augmentation can be done here
        if self.regularize:
            # occasionally flip the aa sequences back-to-front as a regularization technique 
            dix = torch.flip(dix, [0]) if (random.random() < self.config['seq_flip_prob']) else dix

            # mask a small perentage of the amino acids with the MASK token
            # acts like a dropout
            if self.config['seq_mask_prob'] > 0.0:
                num_2_mask = max(0, int(round((dix.shape[0])*self.config['seq_mask_prob'])))
                masked_idx = torch.randperm((dix.shape[0]), dtype=torch.long)[:num_2_mask]
                dix[masked_idx] = self.stoi['MASK']

        # prepend the CLS token to the sequence
        # dix = torch.cat((torch.tensor([self.stoi['CLS']], dtype=torch.long), dix))

        # pad the end with PAD tokens if necessary
        if dix.shape[0] < self.config['block_size']:
            dix = torch.cat((dix, torch.tensor([self.stoi['PAD']] * (self.config['block_size'] - len(dix)), dtype=torch.long)))

        return dix, torch.tensor([affinity], dtype=torch.float32) 
