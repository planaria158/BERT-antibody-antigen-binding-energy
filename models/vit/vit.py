
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
from models.transformer_parts.transformer_parts import TransformerEncoder, MLP_Head


#--------------------------------------------------------------------
# The vision transformer model
# typical settings: 48x48 image, patch_dim=2, num_patches=144
#--------------------------------------------------------------------
class VIT(nn.Module):
    def __init__(self, config):
        super(VIT, self).__init__()
        patch_dim = config['patch_dim']
        emb_dim   = config['emb_dim']
        img_shape = config['image_shape']

        self.token_dim = patch_dim * patch_dim * config['image_channels']   # length of linearized patchs
        self.num_patches = ((img_shape[0]//patch_dim) * (img_shape[0]//patch_dim))

        self.patch_embedding = nn.Sequential(
                                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_dim, p2 = patch_dim),
                                    nn.LayerNorm(self.token_dim),
                                    nn.Linear(self.token_dim, emb_dim))
        self.class_embedding = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))
        self.emb_dropout = nn.Dropout(p=config['emb_dropout'])
        self.transformer_encoder = TransformerEncoder(config['num_layers'], emb_dim, config['num_heads'], 
                                                      config['dim_head'], config['dropout'])
        self.mlp_head = MLP_Head(emb_dim, 1)
           
    def forward(self, imgs): 
        patch_emb = self.patch_embedding(imgs) # i.e. [b, 64, dim]
        b, n, _ = patch_emb.shape
        class_tokens = repeat(self.class_embedding, '() n d -> b n d', b = b)
        embeddings = torch.cat((class_tokens, patch_emb), dim=1) # i.e. [b, 65, dim]
        embeddings += self.pos_embedding[:, :(n + 1)]
        x = self.emb_dropout(embeddings)
        x = self.transformer_encoder(x)
        logits = self.mlp_head(x[:, 0, :])  # [b, 1] just apply to the CLS token
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

        


