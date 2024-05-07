
import pickle as pk
import json
from pathlib import Path
import os
import time
import torch
from torch import nn
from pytorch_lightning.core import LightningModule
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from models.transformer_parts.transformer_parts import TransformerEncoder
from models.residual_mlp.residual_mlp import ResidualMLP
from train_test_inference.test_metrics import test_metrics


class VIT(nn.Module):
    """
        Vision Transformer Model
        Intended to operate on 2D images constructed from scFv sequences

        Args:
            config: dict with configuration parameters
    """
    def __init__(self, model_config, config):
        super(VIT, self).__init__()

        patch_dim = model_config['patch_dim']
        emb_dim   = model_config['emb_dim']
        img_shape = model_config['image_shape']

        self.token_dim = patch_dim * patch_dim * model_config['image_channels']   # length of linearized patchs
        self.num_patches = ((img_shape[0]//patch_dim) * (img_shape[0]//patch_dim))
        print('num_patches:', self.num_patches, ', token_dim:', self.token_dim)

        self.patch_embedding = nn.Sequential(
                                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_dim, p2 = patch_dim),
                                    nn.LayerNorm(self.token_dim),
                                    nn.Linear(self.token_dim, emb_dim))
        self.class_embedding = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))
        self.emb_dropout = nn.Dropout(p=config['emb_dropout'])
        self.transformer_encoder = TransformerEncoder(model_config['num_layers'], emb_dim, model_config['num_heads'], 
                                                      model_config['dim_head'], config['tform_dropout'])
        # The residualMLP regression head
        self.regression_head = ResidualMLP(config, emb_dim, num_layers=4)

           
    def forward(self, imgs): 
        patch_emb = self.patch_embedding(imgs) # i.e. [b, 64, dim]
        b, n, _ = patch_emb.shape
        class_tokens = repeat(self.class_embedding, '() n d -> b n d', b = b)
        embeddings = torch.cat((class_tokens, patch_emb), dim=1) # i.e. [b, 65, dim]
        embeddings += self.pos_embedding[:, :(n + 1)]
        x = self.emb_dropout(embeddings)
        x = self.transformer_encoder(x)
        logits = self.regression_head(x[:, 0, :])  # [b, 1] just apply to the CLS token
        return logits 


class VIT_Lightning(LightningModule):
    """
        Pytorch Lightning Module for training Vision Transformer

        Args:
            config: dict with configuration parameters
    """
    def __init__(self, model_config, config):
        super(VIT_Lightning, self).__init__()
        self.config = config
        self.model = VIT(model_config, config)
        self.criteriion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def common_forward(self, batch, batch_idx):
        x, y, names = batch
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
        filename = os.path.join(path, 'metrics_vit_' + timestamp + '.txt')      
        print('saving metrics to:', filename)
        with open(filename, 'w') as out_file: 
            out_file.write(json.dumps(self.metrics))

        filename = os.path.join(path, 'preds_vit_' + timestamp + '.pkl')      
        print('saving', len(self.preds), 'preds to:', filename)
        pk.dump(self.preds, open(filename, 'wb'))

        filename = os.path.join(path, 'y_vit_' + timestamp + '.pkl')      
        print('saving', len(self.y), 'y values to:', filename)
        pk.dump(self.y, open(filename, 'wb'))
        return 
    
    #--------------------------------------------------------
    # Inference methods
    #--------------------------------------------------------
    def on_predict_start(self):
        self.preds = []

    def predict_step(self, batch, batch_idx):
        y_hat = self.forward(batch[0].float())
        self.preds.extend(y_hat.cpu().numpy().tolist())
        return

    def on_predict_end(self):
        # save the preds to file
        path = Path(self.config['inference_results_folder'])
        path.mkdir(parents=True, exist_ok=True)
        timestamp = str(time.time())
        filename = os.path.join(path, 'preds_vit_' + timestamp + '.pkl')      
        print('saving', len(self.preds), 'preds to:', filename)
        pk.dump(self.preds, open(filename, 'wb'))
        return

    def configure_optimizers(self):
        lr = self.config['learning_rate']
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]

        


