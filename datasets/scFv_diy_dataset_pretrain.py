import math
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class scFv_diy_pretrain_Dataset(Dataset):
    """
    Dataset class for the paired sequences for pretraining the model
    The data consists of paired heavy and light chain sequences
    They are concatenated with a "standard" scFv linker sequence
    Hence the "DIY" in the name
    50% of the time, the heavy and light chains are swapped

    The dataset is a csv file with the following columns:
    - sequence_alignment_aa_heavy: amino acid sequence of the heavy chain
    - sequence_alignment_aa_light: amino acid sequence of the light chain

    The dataset is used for pretraining the model with the following task:
    Masked language model training
        The masked language model training sequence with a small percentage of the amino acids masked
        sequence will look something like:           [CLS, aa, aa, MASK, aa, MASK, aa, ...PAD,..PAD]
        corresponding mask will look something like: [0,   0,   0,   aa,  0,   aa, 0, ...0, 0]

    """
    def __init__(self, config, block_size, csv_file_path, skiprows=0, inference=False, regularize=False):  
        super().__init__()
        print('DIY scFv sequences created by this dataset class')
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

        self.linker = 'SSGGGGSGGGGSGGGGSE' # fixed linker sequnce
        print('Using fixed linker sequence:', self.linker)

    #-------------------------------------------------------
    # Create the mask for masked language model training
    # and the corresponding modified dix sequence
    #
    # if cls = True, prepend a CLS token, else don't
    #-------------------------------------------------------
    def create_mask(self, dix, chain_id, cls=True):

        # prepend the CLS token to the sequence
        if cls:
            dix = torch.cat((torch.tensor([self.stoi['CLS']], dtype=torch.long), dix))
            chain_id = torch.cat((torch.tensor([self.stoi['CLS']], dtype=torch.long), chain_id))
          
        mask = torch.zeros_like(dix)
        # NOTE: this next bit is incorrect.  I've fixed it in the dvm_transformer project
        # I'll flag this with an assert statement and fix it later
        assert (False), 'The mask creation is incorrect in this module.  The correct version is in the dvm_transformer project.  Please use that version.'


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
        return dix, chain_id, mask


    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return self.df.shape[0] 

    #-------------------------------------------------------
    # Get the sequence, and mask data for the given index
    #-------------------------------------------------------
    def __getitem__(self, idx):
        seq1 = self.df.loc[idx, 'sequence_alignment_aa_heavy']
        seq2 = self.df.loc[idx, 'sequence_alignment_aa_light']

        assert isinstance(seq1, str), 'seq1 is not a string'
        assert isinstance(seq2, str), 'seq2 is not a string'

        # chain id to distinguish heavy, light, and linker
        # 1: heavy chain, 2: light chain, 3: linker.  CLS: 0, PAD: 4
        ch1 = torch.ones(len(seq1), dtype=torch.long)
        ch2 = torch.ones(len(seq2), dtype=torch.long) * 2
        chlinker = torch.ones(len(self.linker), dtype=torch.long) * 3
        ch_PAD = 4 # fixed here for now...

        # flip the order of heavy, light chains 50% of the time
        if random.random() > 0.5:
            seq1, seq2, ch1, ch2 = seq2, seq1, ch2, ch1

        chunk = seq1 + self.linker + seq2
        chain_id = torch.cat((ch1, chlinker, ch2))

        # hardwired for now
        Kd = 0
        name = 'none'

        # trim if necessary
        if len(chunk) >= (self.block_size-1):  # '-1' is to accomodate the CLS tokens below
            chunk = chunk[0:self.block_size-1]
            chain_id = chain_id[0:self.block_size-1]

        # encode the string chunk
        dix = torch.tensor([self.stoi[s] for s in chunk], dtype=torch.long)

        # create the mask for the masked language model training
        # and the corresponding modified dix sequence
        dix, chain_id, mask = self.create_mask(dix, chain_id, cls=True)

        # Append with PAD tokens if necessary
        if dix.shape[0] < self.block_size:
            dix = torch.cat((dix, torch.tensor([self.stoi['PAD']] * (self.block_size - len(dix)), dtype=torch.long)))
            chain_id = torch.cat((chain_id, torch.tensor([ch_PAD] * (self.block_size - len(chain_id)), dtype=torch.long)))  
            mask = torch.cat((mask, torch.tensor([0] * (self.block_size - len(mask)), dtype=torch.long)))

        assert(dix.shape[0] == self.block_size), 'dix shape is not equal to block_size'
        assert(chain_id.shape[0] == self.block_size), 'chain_id shape is not equal to block_size'
        assert(mask.shape[0] == self.block_size), 'mask shape is not equal to block_size'

        return dix, chain_id, mask, torch.tensor([Kd], dtype=torch.float32), name
