import torch
from torch import nn
from einops import repeat, rearrange

"""
    These are the building blocks needed to make a transformer
    block that can be inserted into other models
"""


class FeedForwardBlock(nn.Module):
    """
        Feed Forward Block
        Consists of two linear layers with GELU activation
        and dropout

        Args:
            dim: int, dimension of the input tensor
            dropout: float, dropout rate
            multiplier: int, multiplier for the inner dimension
    """
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
    """
        Attention Block: multi-headed self-attention
        uses pre-normalization

        Args:
            dim: int, dimension of the input tensor
            heads: int, number of attention heads
            dim_head: int, dimension of the attention head
            dropout: float, dropout rate
    """
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
    """
        Encoder Block
        Consists of an attention block and a feedforward block
        Uses residual connections and layer normalization

        Args:
            dim: int, dimension of the input tensor
            num_heads: int, number of attention heads
            dim_head: int, dimension of the attention head
            dropout: float, dropout rate
    """
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
    """
        Transformer Encoder
        The transformer encoder block

        Args:
            num_layers: int, number of encoder blocks
            dim: int, dimension of the input tensor
            num_heads: int, number of attention heads
            dim_head: int, dimension of the attention head
            dropout: float, dropout rate
    """
    def __init__(self, num_layers, dim, num_heads, dim_head, dropout):
        super().__init__()
        layers = [EncoderBlock(dim, num_heads, dim_head, dropout) for _ in range(num_layers)]
        self.encoders = nn.Sequential(*layers)

    def forward(self, x):
        out = self.encoders(x)
        return out


class Layer(nn.Module):
    """
        Layer : a single layer of the MLP used in transformer models
                uses prenormalization with layer normalization

        Args:
            in_dim: int, input dimension
            out_dim: int, output dimension
            dropout: float, dropout rate
            normalize: bool, whether to apply batch normalization
            activation: bool, whether to apply GELU activation
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, normalize=True, activation=True):
        super(Layer, self).__init__()
        self.normalize = normalize
        self.activation = activation
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(in_dim) if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x):
        # Use pre-normalization; i.e. first operation applied is normalization
        out = self.dropout(self.activation(self.linear(self.norm(x))))
        return out



class MLP(nn.Module):
    """
        MLP: a simple multi-layer perceptron
        
        A fixed-size 3-layer mlp: 
        input layer (input_dim, 4*input_dim)
        hidden layer(4*input_dim, 2*input_dim)
        output layer(2*input_dim, 1)

        Args:
            input_dim: int, input dimension 
            dropout: float, dropout rate
    """
    def __init__(self, input_dim, dropout):
        super(MLP, self).__init__()

        mlp_hidden_mults = (4, 4, 3, 2) # fixed architecture 

        hidden_dimensions = [input_dim * t for t in  mlp_hidden_mults]
        all_dimensions = [input_dim, *hidden_dimensions, 1]
        dims_pairs = list(zip(all_dimensions[:-1], all_dimensions[1:]))
        layers = []
        for ind, (in_dim, out_dim) in enumerate(dims_pairs):
            print('making mlp. in_dim, out_dim:', in_dim, out_dim)
            if ind >= (len(dims_pairs) - 1) :
                # For regression, the very last Layer has no dropout, normalization, and activation
                layer = Layer(in_dim, out_dim, normalize=False, activation=False)
            else:
                layer = Layer(in_dim, out_dim, dropout)
            
            layers.append(layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x_in):
        return self.net(x_in) 

