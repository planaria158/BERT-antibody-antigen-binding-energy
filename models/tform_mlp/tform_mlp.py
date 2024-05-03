
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
from models.residual_mlp.residual_mlp import ResidualMLP
from models.transformer_parts.transformer_parts import TransformerEncoder

#----------------------------------------------------------
# This is a simple transformer front-ending a residual-MLP
# It operates on sequences of residues of block_size length
#----------------------------------------------------------
class TFormMLP(nn.Module):
    def __init__(self, config):
        super(TFormMLP, self).__init__()
        emb_dim   = config['emb_dim']
        self.wte = nn.Embedding(config['vocab_size'], emb_dim) # token embedding
        self.wpe = nn.Embedding(config['block_size'], emb_dim) # position embedding 
        self.emb_dropout = nn.Dropout(p=config['emb_dropout'])
        self.transformer = TransformerEncoder(config['num_layers'], emb_dim, config['num_heads'], 
                                              config['dim_head'], config['dropout'])
        self.ln_f = nn.LayerNorm(emb_dim)

        # The residual MLP regression head
        self.regression_head = ResidualMLP(config, emb_dim)
           
    def forward(self, x): 
        device = x.device
        b, t = x.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t) 

        # Embeddings and dropout
        tok_emb = self.transformer.wte(x)   # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.emb_dropout(tok_emb + pos_emb)

        # Transformer block
        x = self.transformer(x)
        x = self.ln_f(x)
        
        logits = self.regression_head(x[:, 0, :])  # [b, 1, emb] just apply to the CLS token
        return logits 


#----------------------------------------------------------
# Pytorch Lightning Module that hosts the TFormMLP model
#----------------------------------------------------------
class TFormMLP_Lightning(LightningModule):
    def __init__(self, config):
        super(TFormMLP_Lightning, self).__init__()
        self.config = config
        self.model = TFormMLP(config)
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
        filename = os.path.join(path, 'preds_tform_mlp_' + str(time.time()) + '.pkl')      
        print('saving', len(self.preds), 'preds to:', filename)
        pk.dump(self.preds, open(filename, 'wb'))

        filename = os.path.join(path, 'y_tform_mlp_' + str(time.time()) + '.pkl')      
        print('saving', len(self.y), 'y values to:', filename)
        pk.dump(self.y, open(filename, 'wb'))
        return 

    def configure_optimizers(self):
        lr = self.config['learning_rate']
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]

        


