
import pickle as pk
from pathlib import Path
import os
import time
import torch
from torch import nn
from pytorch_lightning.core import LightningModule
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import repeat, rearrange

#----------------------------------------------------------
# Create a simple Vision Transformer to be used for
# image regression on 2D B&W images (single channel)
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
    def __init__(self, dim, heads, dim_head, dropout = 0.):  
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
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_block = AttentionBlock(dim, num_heads, dim_head=32, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_block = FeedForwardBlock(dim, dropout)

    def forward(self, x):
        x = x + self.attn_block(self.norm1(x))
        x = x + self.ff_block(self.norm2(x)) 
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, dim, num_heads, dropout):
        super().__init__()
        layers = [EncoderBlock(dim, num_heads, dropout)
                  for _ in range(num_layers)]
        self.encoders = nn.Sequential(*layers)

    def forward(self, x):
        out = self.encoders(x)
        return out

class MLP_Head(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(dim), 
                                  nn.Linear(dim, num_classes))
        
    def forward(self, x):
        return self.head(x)


#--------------------------------------------------------------------
# The model
# For a 46x46 image, patch_dim = 2, num_patches = 64, dim = 512
#--------------------------------------------------------------------
class VIT(nn.Module):
    def __init__(self, config):
        super(VIT, self).__init__()
        patch_dim = config['patch_dim']
        num_heads = config['num_heads'] 
        num_layers = config['num_layers']
        dim = config['dim']
        dropout = config['dropout']
        img_shape = config['image_shape']

        self.in_channels = 1 # B&W image
        self.token_dim = patch_dim * patch_dim * self.in_channels  # length of linearized patchs
        self.dim = dim # The embedding dimension for the Encoder
        self.num_patches = ((img_shape[0]//patch_dim) * (img_shape[0]//patch_dim))

        self.patch_embedding = nn.Sequential(
                                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_dim, p2 = patch_dim),
                                    nn.LayerNorm(self.token_dim),
                                    nn.Linear(self.token_dim, dim))
        self.class_embedding = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.emb_dropout = nn.Dropout(p=0.2)
        self.transformer_encoder = TransformerEncoder(num_layers, dim, num_heads, dropout)
        self.mlp_head = MLP_Head(dim, 1)
           
    def forward(self, imgs): 
        patch_emb = self.patch_embedding(imgs) # [N, 64, dim]
        b, n, _ = patch_emb.shape
        class_tokens = repeat(self.class_embedding, '() n d -> b n d', b = b)
        embeddings = torch.cat((class_tokens, patch_emb), dim=1) # [N, 65, dim]
        embeddings += self.pos_embedding[:, :(n + 1)]
        x = self.emb_dropout(embeddings)
        x = self.transformer_encoder(x)
        logits = self.mlp_head(x[:, 0, :])  # [N, 1] just apply to the CLS token
        return logits 


#----------------------------------------------------------
# Pytorch Lightning Module that hosts a simple vision
# transformer model
#----------------------------------------------------------
class VIT_Lightning(LightningModule):
    def __init__(self, config):
        super(VIT_Lightning, self).__init__()
        self.config = config
        self.model = VIT(config)
        self.criteriion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def common_forward(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y_hat = self.model(x)
        loss = self.criteriion(y_hat, y)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_norm_clip'])
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_forward(batch, batch_idx)
        self.log_dict({"loss": loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, _, _ = self.common_forward(batch, batch_idx)
        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return val_loss
    
    def on_predict_start(self):
        self.preds = []
        self.y = []

    def predict_step(self, batch, batch_idx):
        y_hat = self.forward(batch[0].float())
        self.y.extend(batch[1].cpu().numpy().tolist())
        self.preds.extend(y_hat.cpu().numpy().tolist())
        return

    def on_predict_end(self):
        # save the preds to file
        path = Path(self.config['inference_results_folder'])
        path.mkdir(parents=True, exist_ok=True)
        filename = os.path.join(path, 'preds_vit_' + str(time.time()) + '.pkl')      
        print('saving', len(self.preds), 'preds to:', filename)
        pk.dump(self.preds, open(filename, 'wb'))

        filename = os.path.join(path, 'y_vit_' + str(time.time()) + '.pkl')      
        print('saving', len(self.y), 'y values to:', filename)
        pk.dump(self.y, open(filename, 'wb'))
        return 

    def configure_optimizers(self):
        lr = self.config['learning_rate']
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]

        


