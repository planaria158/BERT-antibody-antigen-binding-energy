
import pickle as pk
import json
import csv
from pathlib import Path
import os
import time
import torch
from torch import nn
from pytorch_lightning.core import LightningModule
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from models.residual_mlp import ResidualMLP
from models.model_parts import TransformerEncoder
from train_test_inference.test_metrics import test_metrics

class TFormMLP_v2(nn.Module):
    def __init__(self, model_config, config):
        super(TFormMLP_v2, self).__init__()
        emb_dim   = model_config['emb_dim']
        self.block_size = model_config['block_size']

        self.token_embedding = nn.Embedding(model_config['vocab_size'], emb_dim) # token embedding
        self.group_embedding = nn.Embedding(model_config['aa_groups_size'], emb_dim) # aa group embedding
        self.var_embedding   = nn.Embedding(model_config['pos_variab_size'], emb_dim) # variability of each position in sequence

        self.pos_embedding = nn.Parameter(torch.randn(1, self.block_size + 1, emb_dim))
        self.class_embedding = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.emb_dropout = nn.Dropout(p=config['emb_dropout'])
        self.transformer = TransformerEncoder(model_config['num_layers'], emb_dim, model_config['num_heads'], 
                                              model_config['dim_head'], config['tform_dropout'])
        
        # The residualMLP regression head
        self.regression_head = ResidualMLP(config, emb_dim, num_layers=4) # fixed here at 4 layers
           
    def forward(self, x, aa_grps, aa_var):  
        b, n = x.shape
        assert n <= self.block_size, f"Cannot forward sequence of length {n}, block size is only {self.block_size}"
        tok_emb = self.token_embedding(x)       # token embeddings of shape (b, n, n_embd)
        grp_emb = self.group_embedding(aa_grps) # token embeddings of shape (b, n, n_embd)
        var_emb = self.var_embedding(aa_var)    # token embeddings of shape (b, n, n_embd)
        tok_emb += (grp_emb + var_emb)
        class_tokens = repeat(self.class_embedding, '() n d -> b n d', b = b)
        embeddings = torch.cat((class_tokens, tok_emb), dim=1) # i.e. [b, n+1, dim]
        embeddings += self.pos_embedding[:, :(n + 1)]
        x = self.emb_dropout(embeddings)
        tform_out = self.transformer(x)
        logits = self.regression_head(tform_out[:, 0, :])  # [b, 1, emb] just apply to the class token
        return logits, tform_out


class TFormMLP_Lightning_v2(LightningModule):
    """
        Pytorch Lightning Module that hosts the TFormMLP model

        Args:
            model_config: dictionary of model parameters containing the following keys:
                'vocab_size': size of the vocabulary of residues
                'emb_dim': dimension of the token embeddings
                'block_size': length of the sequences to be processed
                'num_layers': number of transformer layers
                'num_heads': number of attention heads
                'dim_head': dimension of the attention head

            config: dictionary of config parameters containing the following keys:
                'vocab_size': size of the vocabulary of residues
                'emb_dim': dimension of the token embeddings
                'block_size': length of the sequences to be processed
                'num_layers': number of transformer layers
                'num_heads': number of attention heads
                'dim_head': dimension of the attention head
                'tform_dropout': dropout rate for the transformer layers
                'emb_dropout': dropout rate for the token embeddings
                'learning_rate': initial learning rate
                'betas': betas for the AdamW optimizer
                'lr_gamma': gamma for the learning rate scheduler
                'inference_results_folder': folder to save inference results
    """
    def __init__(self, model_config, config):
        super(TFormMLP_Lightning_v2, self).__init__()
        self.config = config
        self.model = TFormMLP_v2(model_config, config)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x, x2, x3):
        return self.model(x, x2, x3)

    def common_forward(self, batch, batch_idx):
        x1, x2, x3, y, names = batch
        y_hat, _ = self.model(x1, x2, x3) # x1 is the sequence, x2 is the group encoding, x3 is the mutation frequency
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_forward(batch, batch_idx)
        self.log_dict({"loss": loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, _, _ = self.common_forward(batch, batch_idx)
        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return val_loss
    
    #--------------------------------------------------------
    # Test methods
    #--------------------------------------------------------
    def on_test_start(self):
        self.preds = []
        self.y = []
        self.metrics = None

    def test_step(self, batch, batch_idx):
        test_loss, y_hat, y = self.common_forward(batch, batch_idx)
        self.log_dict({"test_loss": test_loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.y.extend(y.cpu().numpy().tolist())
        self.preds.extend(y_hat.cpu().numpy().tolist())
        return test_loss

    def on_test_end(self):
        assert(len(self.preds) == len(self.y))
        self.metrics = test_metrics(self.y, self.preds)
        print(self.metrics)

        # save the metrics, preds, and y values to file
        path = Path(self.config['test_results_folder'])
        path.mkdir(parents=True, exist_ok=True)
         
        timestamp = str(time.time())
        filename = os.path.join(path, 'metrics_tform_mlp_' + timestamp + '.txt')      
        print('saving metrics to:', filename)
        with open(filename, 'w') as out_file: 
            out_file.write(json.dumps(self.metrics))

        filename = os.path.join(path, 'preds_tform_mlp_' + timestamp + '.pkl')      
        print('saving', len(self.preds), 'preds to:', filename)
        pk.dump(self.preds, open(filename, 'wb'))

        filename = os.path.join(path, 'y_tform_mlp_' + timestamp + '.pkl')      
        print('saving', len(self.y), 'y values to:', filename)
        pk.dump(self.y, open(filename, 'wb'))
        return 
    
    #--------------------------------------------------------
    # Inference methods
    #--------------------------------------------------------
    def on_predict_start(self):
        self.preds = []
        self.seq_name = []

    def predict_step(self, batch, batch_idx):
        y_hat, _ = self.forward(batch[0], batch[1], batch[2])
        self.preds.extend(y_hat.cpu().numpy().tolist())
        self.seq_name.extend(batch[-1])
        return

    def on_predict_end(self):
        # save the preds to file
        path = Path(self.config['inference_results_folder'])
        path.mkdir(parents=True, exist_ok=True)
        timestamp = str(time.time())
        filename = os.path.join(path, 'preds_tform_mlp_' + timestamp + '.csv')      
        print('saving', len(self.preds), 'preds to:', filename)
        fields = ['name', 'pred_Kd']
        rows = [{'name': self.seq_name[i], 'pred_Kd': self.preds[i][0]} for i in range(len(self.preds))]
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return
        

    def configure_optimizers(self):
        lr = self.config['learning_rate']
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]

        


