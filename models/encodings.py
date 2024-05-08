
import torch
from torch import nn
from einops import repeat


"""
    Convert position indices tensor into an embedding using the
    sinusoidal embedding formula

    Args:
    position: 1D tensor of length sequence length
    temb_dim: Dimension of the embedding

    Returns:
    BxD embedding representation of B time steps
"""
def sinusoidal_encoding(position, temb_dim):
    assert temb_dim % 2 == 0, "embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=position.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    emb = position[:, None].repeat(1, temb_dim // 2) / factor
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb
