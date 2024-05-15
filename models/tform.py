
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
from models.model_parts import MLP
from models.model_parts import TransformerEncoder
from train_test_inference.test_metrics import test_metrics

class TFormMLP(nn.Module):
    """
        Model class for a transformer model that operates on sequences of residues
        It operates on sequences of residues of block_size length
        Regression is done via the usual class token added to the start of the sequence

        This model consists of a front-end Transformer and a ResidualMLP regression head

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
    """
    def __init__(self, model_config, config):
        super(TFormMLP, self).__init__()
        emb_dim   = model_config['emb_dim']
        self.vocab_size = model_config['vocab_size']
        self.block_size = model_config['block_size']
        assert config['train_type'] in ['mask_lang_model', 'regression'], 'train_type must be either "mask_lang_model" or "regression"'
        self.train_type = config['train_type']
        self.config = config

        # old constant pos embedding..  self.pos_embedding = nn.Parameter(torch.randn(1, self.block_size, emb_dim))
        self.token_embedding = nn.Embedding(self.vocab_size, emb_dim) # token embedding
        self.pos_embedding   = nn.Embedding(self.block_size, emb_dim) # position embedding 
        self.emb_dropout = nn.Dropout(p=config['emb_dropout'])
        self.transformer = TransformerEncoder(model_config['num_layers'], emb_dim, model_config['num_heads'], 
                                              model_config['dim_head'], config['tform_dropout'])

        # The Masked Language Model (MLM) head                                      
        self.mlm_head = nn.Linear(emb_dim, self.vocab_size, bias=False) # predictions are tokens
        # The residualMLP regression head
        print('Consider using a much simpler regression head!!!!')
        self.regression_head = ResidualMLP(config, emb_dim, num_layers=4) # predictions are real values
        # self.regression_head = MLP(emb_dim, config['mlp_dropout'])

        # This generally means we've just loaded pretrained weights from
        # a fine tuned model and we want to freeze the encoder
        # In this case, only the regression head will be trained on fine-tune dataset
        if config['freeze_base_model'] == True:
            print('freezing base model')
            for param in self.transformer.parameters():
                param.requires_grad = False

            # # BUT unfreeze the final layer of the transformer
            # print('Final layer of transformer is unfrozen!')
            # for param in self.transformer.encoders[-1].parameters():
            #     param.requires_grad = True

            for param in self.token_embedding.parameters():
                param.requires_grad = False

            for param in self.pos_embedding.parameters():
                param.requires_grad = False
            
            for param in self.emb_dropout.parameters():
                params.requires_grad = False
            
            for param in self.mlm_head.parameters():
                param.requires_grad = False

           
    def forward(self, x, mask=None): 
        b, n = x.shape
        device = x.device
        assert n <= self.block_size, f"Cannot forward sequence of length {n}, block size is only {self.block_size}"
        pos = torch.arange(0, n, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t) 
        tok_emb = self.token_embedding(x) # token embeddings of shape (b, n, n_embd)        
        pos_emb = self.pos_embedding(pos) # position embeddings of shape (1, t, n_embd)
        embeddings = tok_emb + pos_emb
        x = self.emb_dropout(embeddings)
        tform_out = self.transformer(x)

        # Run in Masked Language Model (MLM) mode
        if self.train_type == 'mask_lang_model':
            logits = self.mlm_head(tform_out) # [b, n, vocab_size]  
            mask = mask.view(-1)
            mask_idx = torch.nonzero(mask)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size),  mask, reduction='none')
            loss = loss.sum() / mask_idx.shape[0]
        else:
            # else running in regression mode
            loss = 0
            logits = self.regression_head(tform_out[:, 0, :])  # [b, 1, emb] just apply to the class token

        return logits, loss, tform_out


class TFormMLP_Lightning(LightningModule):
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
        super(TFormMLP_Lightning, self).__init__()
        self.config = config
        self.model = TFormMLP(model_config, config)
        assert config['loss_type'] in ['mse', 'mae'], 'loss_type must be either "mse" or "mae"'
        self.criterion = nn.MSELoss() if config['loss_type'] == 'mse' else nn.L1Loss()
        assert config['train_type'] in ['mask_lang_model', 'regression'], 'train_type must be either "mask_lang_model" or "regression"'
        self.train_type = config['train_type']
        self.save_hyperparameters()

    def forward(self, x, mask=None):
        return self.model(x, mask)

    def common_forward(self, batch, batch_idx):
        x, mask, y, names = batch

        # Masked Language Model (MLM) mode
        if self.train_type == 'mask_lang_model':
            y_hat, loss, tform_out = self.model(x, mask)     
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_norm_clip'])
        else:   
            # Regression mode
            y_hat, _, _ = self.model(x)
            loss = self.criterion(y_hat, y)

        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_forward(batch, batch_idx)
        self.log_dict({"loss": loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True, batch_size=self.config['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, _, _ = self.common_forward(batch, batch_idx)
        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True, batch_size=self.config['batch_size'])
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
        y_hat, _ = self.forward(batch[0])
        self.preds.extend(y_hat.cpu().numpy().tolist())
        self.seq_name.extend(batch[2])
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

        


