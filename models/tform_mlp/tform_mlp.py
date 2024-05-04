
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
from models.transformer_parts.transformer_parts import TransformerEncoder, MLP_Head

#----------------------------------------------------------
# This is a simple transformer front-ending a residual-MLP
# It operates on sequences of residues of block_size length
# Regression is done via the usual class token added to the
# start of the sequence
#----------------------------------------------------------
class TFormMLP(nn.Module):
    def __init__(self, config):
        super(TFormMLP, self).__init__()
        emb_dim   = config['emb_dim']
        self.block_size = config['block_size']

        self.token_embedding = nn.Embedding(config['vocab_size'], emb_dim) # token embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.block_size + 1, emb_dim))
        self.class_embedding = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.emb_dropout = nn.Dropout(p=config['emb_dropout'])
        self.transformer = TransformerEncoder(config['num_layers'], emb_dim, config['num_heads'], 
                                              config['dim_head'], config['tform_dropout'])
        self.ln_f = nn.LayerNorm(emb_dim)

        # The residualMLP regression head
        self.regression_head = ResidualMLP(config, emb_dim)
           
    def forward(self, x): 
        b, n = x.shape
        assert n <= self.block_size, f"Cannot forward sequence of length {n}, block size is only {self.block_size}"
        tok_emb = self.token_embedding(x)   # token embeddings of shape (b, n, n_embd)
        class_tokens = repeat(self.class_embedding, '() n d -> b n d', b = b)
        embeddings = torch.cat((class_tokens, tok_emb), dim=1) # i.e. [b, n+1, dim]
        embeddings += self.pos_embedding[:, :(n + 1)]
        x = self.emb_dropout(embeddings)
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.regression_head(x[:, 0, :])  # [b, 1, emb] just apply to the class token
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
        y_hat = self.model(x)
        loss = self.criteriion(y_hat, y)
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
        y_hat = self.forward(batch[0])
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

        


