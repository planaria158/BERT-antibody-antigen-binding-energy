import math
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class scFv_Dataset_v2(Dataset):
    """
        Dataset class for scFv sequence, Kd data

        Args:
            config: dict with configuration parameters
            csv_file_path: path to the csv file
            skiprows: number of rows to skip at the beginning of the file
            inference: if True, the dataset is used for inference
            regularize: if True, the dataset is used for training and data augmentation/regularization is applied
    """
    def __init__(self, config, block_size, csv_file_path, skiprows=0, inference=False, regularize=False):  
        super().__init__()
        self.config = config
        self.block_size = block_size
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
        self.vocab_size = vocab_size

        # aa groups: group name for each position in the self.chars array above
        self.aa_groups = ['none', 'nonpolar', 'nonpolar', 'neg', 'neg', 'nonpolar', 'nonpolar', 
                          'pos', 'nonpolar', 'pos', 'nonpolar', 'nonpolar', 'neg', 
                          'nonpolar', 'neg', 'pos', 'polar', 'polar', 'nonpolar', 
                          'nonpolar', 'polar', 'none', 'none', 'none']
        
        self.groups = ['none', 'nonpolar', 'neg', 'pos', 'polar']
        print('aa groups:', self.aa_groups)

        # The relative variability frequence for each amino acid position in the scFv sequences over the entire clean_3 dataset
        # This fixed-array is 247 (246 residues + an extra 0.0000 added for holdout set)
        #
        # A better way to do this would have been to pre-computed for each dataset separately.
        raw_pos_variability = torch.tensor([ 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
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
                                             0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 
                                             0.0000, 0.0000, 0.0000, 0.0000]) 
        
        # Map the raw_pos_variability into 10 buckets based on value
        var_buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        def get_bucket(n):
            assert(n >= 0 and n <= 1), "make sure n is in the range [0,1.0]"
            for i, k in enumerate(var_buckets):
                if n <= k:
                    return i
            return 0        # assign each variability value to a bucket
        
        self.enc_pos_variability = [get_bucket(v) for v in raw_pos_variability]

        print('a sample of some raw variability values:', raw_pos_variability[60:65])
        print('and their corresponding encoded values :', self.enc_pos_variability[60:65])


        # encoding and decoding residues
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        # encoding decoding groups
        self.groupstoi = { group:i for i,group in enumerate(self.aa_groups) }
        self.groupitos = { i:group for i,group in enumerate(self.aa_groups) }
        self.gtoi = { group:i for i,group in enumerate(self.groups) }
        self.itog = { i:group for i,group in enumerate(self.groups) }

    def encode_aa(self, aa_name):
        return self.stoi[aa_name]
    
    def decode_aa(self, aa_idx):
        assert aa_idx < self.vocab_size, 'aa_idx is out of bounds'
        return self.itos[aa_idx]
    
    # Get the group encoding for a given amino acid
    def encode_aa_to_group(self, aa_name):
        aa_idx = self.stoi[aa_name]
        group_name = self.groupitos[aa_idx]
        group_enc = self.gtoi[group_name]
        return group_enc
    
    def decode_group(self, group_idx):
        return self.itog[group_idx]
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return self.df.shape[0] 

    def __getitem__(self, idx):
        """ 
            Returns sequence encoding, group encoding, affinity
        """
        seq = self.df.loc[idx, 'sequence_a']

        # apologies: next couple lines are overly dataset-specific
        if self.inference == False: # training or test mode
            Kd = self.df.loc[idx, 'Kd']
            # Kd_min = self.df.loc[idx, 'Kd_min']
            # Kd_max = self.df.loc[idx, 'Kd_max']
            assert not math.isnan(Kd), 'Kd is nan'
            # assert not math.isnan(Kd_min), 'Kd_min is nan'
            # assert not math.isnan(Kd_max), 'Kd_max is nan'
            # assert Kd_min <= Kd <= Kd_max, 'Kd_min, Kd, Kd_max are inconsistent'
            name = 'none'
        else:
            # Kd = Kd_min = Kd_max = 0 # inference mode - Kd is not available
            Kd = 0 # inference mode - Kd is not available
            name = self.df.loc[idx, 'description_a']

        assert Kd >= 0.0, 'affinity cannot be negative'

        # get a randomly located block_size substring from the sequence
        if len(seq) <= self.block_size:
            chunk = seq
        else:
            start_idx = np.random.randint(0, len(seq) - (self.block_size))
            chunk = seq[start_idx:start_idx + self.block_size]

        # encode residues, residues' groups, and position variability
        dix = torch.tensor([self.stoi[s] for s in chunk], dtype=torch.long)
        gix = torch.tensor([self.encode_aa_to_group(s) for s in chunk], dtype=torch.long)
        vix = torch.tensor(self.enc_pos_variability[:len(dix)], dtype=torch.long)

        # some sequence-level regularization & augmentation can be done here
        if self.regularize:
            # occasionally flip the aa sequences back-to-front as a regularization technique 
            dix = torch.flip(dix, [0]) if (random.random() < self.config['seq_flip_prob']) else dix
            gix = torch.flip(gix, [0]) if (random.random() < self.config['seq_flip_prob']) else gix
            vix = torch.flip(vix, [0]) if (random.random() < self.config['seq_flip_prob']) else vix

            # mask a small perentage of the amino acids with the MASK token
            # acts like a dropout
            if self.config['seq_mask_prob'] > 0.0:
                num_2_mask = max(0, int(round((dix.shape[0])*self.config['seq_mask_prob'])))
                masked_idx = torch.randperm((dix.shape[0]), dtype=torch.long)[:num_2_mask]
                dix[masked_idx] = self.stoi['MASK']
                gix[masked_idx] = self.gtoi['none']
                vix[masked_idx] = 0  # ?? not sure about this...

            # Choose a new value for Kd that is randomly chosen from [Kd_lower, Kd_upper]
            # if random.random() < self.config['kd_mod_prob']:
            #     Kd = random.uniform(Kd_min, Kd_max)
                

        # prepend the CLS token to the sequence
        # dix = torch.cat((torch.tensor([self.stoi['CLS']], dtype=torch.long), dix))

        # pad the end with PAD tokens if necessary
        if dix.shape[0] < self.block_size:
            dix = torch.cat((dix, torch.tensor([self.stoi['PAD']] * (self.block_size - len(dix)), dtype=torch.long)))
            gix = torch.cat((gix, torch.tensor([self.gtoi['none']] * (self.block_size - len(gix)), dtype=torch.long)))
            vix = torch.cat((vix, torch.tensor([0] * (self.block_size - len(vix)), dtype=torch.long)))

        return dix, gix, vix, torch.tensor([Kd], dtype=torch.float32), name
