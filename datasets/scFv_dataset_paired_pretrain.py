import math
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class scFv_paired_pretrain_Dataset(Dataset):
    """
    Dataset class for the scFv paired sequences for pretraining the model
    The dataset is a csv file with the following columns:
    - sequence_alignment_aa_heavy: amino acid sequence of the heavy chain
    - sequence_alignment_aa_light: amino acid sequence of the light chain

    The dataset is used for pretraining the model with the following task:
    Masked language model training
        The masked language model training sequence with a small percentage of the amino acids masked
        sequence will look something like:           [CLS, aa, aa, MASK, aa, MASK, aa, ...PAD,..PAD]
        corresponding mask will look something like: [0,   0,   0,   aa,  0,   aa, 0, ...0, 0]

    The two sequences are concatenated with a SEP token in between

    """
    def __init__(self, config, block_size, csv_file_path, skiprows=0, inference=False, regularize=False):  
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.inference = inference
        self.regularize = regularize # sequence flipping etc...
        assert config['train_type'] in ['mask_lang_model', 'regression'], 'train_type must be either "mask_lang_model" or "regression"'
        self.train_type = config['train_type']

        print('reading the data from:', csv_file_path)
        self.df = pd.read_csv(csv_file_path, skiprows=skiprows)
        
        # 20 amino acids + special tokens (CLS, X, PAD, MASK, SEP) 
        self.chars = ['CLS', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                      'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X', 'MASK', 'PAD', 'SEP']
        self.first_aa_idx = 1
        self.last_aa_idx = len(self.chars) - 4
        print('vocabulary:', self.chars)
        data_size, vocab_size = self.df.shape[0], len(self.chars)
        print('data has %d rows, %d vocab size' % (data_size, vocab_size))

        # encoding and decoding residues
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.vocab_size = vocab_size

    #-------------------------------------------------------
    # Create the mask for masked language model training
    # and the corresponding modified dix sequence
    #
    # if cls = True, prepend a CLS token, else don't
    #-------------------------------------------------------
    def create_mask(self, dix, cls=True):

        # prepend the CLS token to the sequence
        if cls:
          dix = torch.cat((torch.tensor([self.stoi['CLS']], dtype=torch.long), dix))
          
        # NOTE: this next bit is incorrect.  I've fixed it in the dvm_transformer project
        # I'll flag this with an assert statement and fix it later
        assert (False), 'The mask creation is incorrect in this module.  The correct version is in the dvm_transformer project.  Please use that version.'
        mask = torch.zeros_like(dix)

        # training will be masked language model
        if self.train_type == 'mask_lang_model':

            # get number of tokens to mask
            n_pred = max(1, int(round(self.block_size * self.config['mask_prob'])))

            # indices of the tokens that will be masked (a random permutation of n_pred of the tokens)
            masked_idx = torch.randperm(len(dix)-1, dtype=torch.long)[:n_pred]
            masked_idx += 1  # so we never mask the CLS token
            mask = torch.zeros_like(dix)

            # copy the actual tokens to be masked, to the mask
            mask[masked_idx] = dix[masked_idx]
            # And replace all masked tokens with the MASK token
            dix[masked_idx] = self.stoi['MASK']

        assert(dix.shape[0] == mask.shape[0]), 'dix and mask shape is not equal'
        return dix, mask


    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return self.df.shape[0] 

    def get_chunk(self, seq):
        # get a randomly located block_size substring from the sequence
        if len(seq) <= self.block_size:
            chunk = seq
        else:
            start_idx = np.random.randint(0, len(seq) - (self.block_size))
            chunk = seq[start_idx:start_idx + self.block_size]

        return chunk


    #-------------------------------------------------------
    # Get the sequence, and mask data for the given index
    #-------------------------------------------------------
    def __getitem__(self, idx):
        seq1 = self.df.loc[idx, 'sequence_alignment_aa_heavy']
        seq2 = self.df.loc[idx, 'sequence_alignment_aa_light']

        # hardwired for now
        Kd = 0
        name = 'none'

        # make sure both seq1 and seq2 are strings
        if isinstance(seq1, str) == False :
            print('WARNING, seq1 is not a string:', seq1, ', idx:', idx, ', picking another row at random')
            idx = np.random.randint(0, self.df.shape[0])
            seq1 = self.df.loc[idx, 'sequence_alignment_aa_heavy']
            seq2 = self.df.loc[idx, 'sequence_alignment_aa_light']

        if isinstance(seq2, str) == False :
            print('WARNING, seq2 is not a string:', seq1, ', idx:', idx, ', picking another row at random')
            idx = np.random.randint(0, self.df.shape[0])
            seq1 = self.df.loc[idx, 'sequence_alignment_aa_heavy']
            seq2 = self.df.loc[idx, 'sequence_alignment_aa_light']


        chunk1 = self.get_chunk(seq1)
        chunk2 = self.get_chunk(seq2)
        # if the sum of the two chunks is greater than the block_size, then we need to truncate
        # trim evenly from ends of both chunks
        if len(chunk1) + len(chunk2) >= (self.block_size-2):  # '-2' is to accomodate the CLS & SEP tokens below
            trim = (len(chunk1) + len(chunk2) - (self.block_size-2)) 
            trim1 = trim // 2
            trim2 = trim - trim1
            chunk1 = chunk1[0:len(chunk1)-trim1]
            chunk2 = chunk2[0:len(chunk2)-trim2]
            assert (len(chunk1) + len(chunk2)) == (self.block_size-2), 'chunk1 + chunk2 is not equal to block_size-1'

        
        # encode the string chunks
        dix1 = torch.tensor([self.stoi[s] for s in chunk1], dtype=torch.long)
        dix2 = torch.tensor([self.stoi[s] for s in chunk2], dtype=torch.long)

        # create the mask for the masked language model training
        # and the corresponding modified dix sequence
        dix1, mask1 = self.create_mask(dix1, cls=True)
        dix2, mask2 = self.create_mask(dix2, cls=False)

        # concatenate the two sequences with a SEP token in between
        dix = torch.cat((dix1, torch.tensor([self.stoi['SEP']], dtype=torch.long), dix2)) 
        mask = torch.cat((mask1, torch.tensor([0], dtype=torch.long), mask2)) 

        # Append with PAD tokens if necessary
        if dix.shape[0] < self.block_size:
            dix = torch.cat((dix, torch.tensor([self.stoi['PAD']] * (self.block_size - len(dix)), dtype=torch.long)))
            mask = torch.cat((mask, torch.tensor([0] * (self.block_size - len(mask)), dtype=torch.long)))

        assert(dix.shape[0] == self.block_size), 'dix shape is not equal to block_size'
        assert(mask.shape[0] == self.block_size), 'mask shape is not equal to block_size'

        return dix, mask, torch.tensor([Kd], dtype=torch.float32), name
