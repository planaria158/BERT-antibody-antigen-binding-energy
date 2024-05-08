import torch
from torch import nn
from einops import repeat
from models.encodings import sinusoidal_encoding


class FixedPosEmbedding(nn.Module):
    def __init__(self, total_dim, emb_dim, name):
        super(FixedPosEmbedding, self).__init__()
        self.total_dim = total_dim
        print(name, ', FixedPosEmbedding, total dimensions:', self.total_dim)
        positions = torch.arange(0, self.total_dim).long()
        # self.pos_embedding = 'pos_embedding_' + name
        
        # calculate embeddings once and store them as a buffer
        self.register_buffer('pos_embedding', #self.pos_embedding, 
                             sinusoidal_encoding(torch.as_tensor(positions), emb_dim))
        # print('pos_embedding.shape:', self.pos_embedding.shape)

    def forward(self, x):
        print('SHIT x shape:', x.shape)
        embed = repeat(self.pos_embedding, 'n d -> b n d', b = x.shape[0])
        return x + embed
