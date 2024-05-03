import torch
from torch import nn
from einops import repeat, rearrange

#----------------------------------------------------------
# These are the "parts" needed to make a transformer
# block that can be inserted into other models
#----------------------------------------------------------
class FeedForwardBlock(nn.Module):
    def __init__(self, dim, dropout=0., multiplier=4):
        super().__init__()
        inner_dim = dim * multiplier
        self.block = nn.Sequential(
                     nn.Linear(dim, inner_dim),
                     nn.GELU(),
                     nn.Linear(inner_dim, dim), 
                     nn.Dropout(dropout)
            )

    def forward(self, x):
        out = self.block(x)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):  
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_head, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_block = AttentionBlock(dim, num_heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_block = FeedForwardBlock(dim, dropout)

    def forward(self, x):
        x = x + self.attn_block(self.norm1(x))
        x = x + self.ff_block(self.norm2(x)) 
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, dim, num_heads, dim_head, dropout):
        super().__init__()
        layers = [EncoderBlock(dim, num_heads, dim_head, dropout) for _ in range(num_layers)]
        self.encoders = nn.Sequential(*layers)

    def forward(self, x):
        out = self.encoders(x)
        return out

class MLP_Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(in_dim), 
                                  nn.Linear(in_dim, out_dim))
        
    def forward(self, x):
        return self.head(x)
