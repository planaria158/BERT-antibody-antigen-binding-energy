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
            regularize: if True, the dataset is used for training and data augmentation/regularization is applied

        Returns:
            dix: tensor of encoded amino acid sequence
            mask: tensor of masked amino acid sequence
            affinity: tensor of affinity values
            name: name of the sequence

        If inference is True, affinity is set to 0 and name is the name of the sequence
        If regularize is True, the sequence is regularized with a small percentage of the amino acids masked

        This dataset operates in two modes:
        1. Regression training
           The original sequence with a small percentage of the amino acids masked
           sequence will look something like: [aa1, aa2, MASK, aa4, MASK, aa6, ...PAD,..PAD]
           mask will be None

        2. Masked language model training
           The masked language model training sequence with a small percentage of the amino acids masked
           sequence will look something like:           [CLS, aa, aa, MASK, aa, MASK, aa, ...PAD,..PAD]
           corresponding mask will look something like: [0,   0,   0,   aa,  0,   aa, 0, ...0, 0]

    """
    def __init__(self, config, block_size, csv_file_path, skiprows=0, inference=False, regularize=False):  
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.inference = inference
        self.regularize = regularize # sequence flipping etc...
        print('reading the data from:', csv_file_path)
        self.df = pd.read_csv(csv_file_path, skiprows=skiprows)
        
        # 20 amino acids + special tokens (CLS, X, PAD, MASK) 
        self.chars = ['CLS', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                      'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X', 'MASK', 'PAD']
        self.first_aa_idx = 1
        self.last_aa_idx = len(self.chars) - 4
        print('vocabulary:', self.chars)
        data_size, vocab_size = self.df.shape[0], len(self.chars)
        print('data has %d rows, %d vocab size (unique).' % (data_size, vocab_size))

        # encoding and decoding residues
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.vocab_size = vocab_size

    def get_random_aa_tokens(self, length):
        return torch.tensor([self.stoi[aa] for aa in random.choices(self.chars[self.first_aa_idx:self.last_aa_idx+1], k=length)], dtype=torch.long)

    #-------------------------------------------------------
    # Create the mask for masked language model training
    # and the corresponding modified dix sequence
    #-------------------------------------------------------
    def create_mask(self, dix):
        mask = torch.zeros_like(dix)
        
        # training will be masked language model
        if self.config['mask_prob'] > 0:
            # prepend the CLS token to the sequence
            dix = torch.cat((torch.tensor([self.stoi['CLS']], dtype=torch.long), dix))

            # get number of tokens to mask
            n_pred = max(1, int(round(self.block_size * self.config['mask_prob'])))

            # indices of the tokens that will be masked (a random permutation of n_pred of the tokens)
            masked_idx = torch.randperm(len(dix)-1, dtype=torch.long)[:n_pred]
            masked_idx += 1  # so we never mask the CLS token
            mask = torch.zeros_like(dix)

            # copy the actual tokens to be masked, to the mask
            mask[masked_idx] = dix[masked_idx]
            # ... and overwrite them in the data
            # with 80% probability, change to MASK token
            # with 10% probability, change to random aa token
            # with 10% probability, keep the same token
            p8 = int(math.ceil(n_pred*0.8))
            p9 = int(math.ceil(n_pred*0.9))
            mask_token_idxs = masked_idx[0:p8]
            rand_aa_idxs = masked_idx[p8:p9]
            keep_idxs = masked_idx[p9:]
            assert(masked_idx.shape[0] == mask_token_idxs.shape[0] + rand_aa_idxs.shape[0] + keep_idxs.shape[0])
            
            dix[mask_token_idxs] = self.stoi['MASK']    
            dix[rand_aa_idxs] = self.get_random_aa_tokens(rand_aa_idxs.shape[0])

        return dix, mask


    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return self.df.shape[0] 


    #-------------------------------------------------------
    # Get the sequence, mask, affinity and name data
    #-------------------------------------------------------
    def __getitem__(self, idx):
        seq = self.df.loc[idx, 'sequence_a']

        # apologies: next couple lines are overly dataset-specific
        if self.inference == False: # training or test mode
            Kd = self.df.loc[idx, 'Kd'] if self.inference == False else 0
            assert not math.isnan(Kd), 'Kd is nan'
            name = 'none'
        else:
            Kd = 0 # inference mode - Kd is not available
            name = self.df.loc[idx, 'description_a']

        assert Kd >= 0.0, 'not allowing for negative affinities'

        # get a randomly located block_size substring from the sequence
        if len(seq) <= self.block_size:
            chunk = seq
        else:
            start_idx = np.random.randint(0, len(seq) - (self.block_size))
            chunk = seq[start_idx:start_idx + self.block_size]

        # encode the string
        dix = torch.tensor([self.stoi[s] for s in chunk], dtype=torch.long)

        # Sequence-level regularization & augmentation can be done here
        # Only do this if we are training for regression 
        # Do not do this for masked language model training
        if self.config['mask_prob'] == 0 and self.regularize:
            # mask a small perentage of the amino acids with the MASK token
            # acts like a dropout.  This is NOT the same as the masked language model training
            # and the model should not be trained with this regularization during masked 
            # language model training
            if self.config['seq_mask_prob'] > 0.0:
                num_2_mask = max(0, int(round((dix.shape[0])*self.config['seq_mask_prob'])))
                masked_idx = torch.randperm((dix.shape[0]), dtype=torch.long)[:num_2_mask]
                dix[masked_idx] = self.stoi['MASK']

        # create the mask for the masked language model training
        # and the corresponding modified dix sequence
        dix, mask = self.create_mask(dix)

        # Append with PAD tokens if necessary
        if dix.shape[0] < self.block_size:
            dix = torch.cat((dix, torch.tensor([self.stoi['PAD']] * (self.block_size - len(dix)), dtype=torch.long)))
            if mask != None:
                mask = torch.cat((mask, torch.tensor([0] * (self.block_size - len(mask)), dtype=torch.long)))


        return dix, mask, torch.tensor([Kd], dtype=torch.float32), name
