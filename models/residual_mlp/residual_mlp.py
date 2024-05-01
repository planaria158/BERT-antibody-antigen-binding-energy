
import pickle as pk
from pathlib import Path
import os
import time
import torch
from torch import nn
from pytorch_lightning.core import LightningModule

#----------------------------------------------------------
# Layer class for the MLP
#----------------------------------------------------------
class Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, normalize=True, activation=True):
        super(Layer, self).__init__()
        self.normalize = normalize
        self.activation = activation
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim) if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x):
        out = self.dropout(self.activation(self.norm(self.linear(x))))
        return out

"""
    An MLP with residual connections
"""
class ResidualMLP(nn.Module):
    def __init__(self, config, input_dim):
        super(ResidualMLP, self).__init__()
        print('Regression head is ResidualMLP')
        self.out_dim = input_dim  # output dim each layer. 
        self.num_layers = 8 # hardwired number of layers. 
        self.in_dim = input_dim

        #  may look something like this (depending on input_dim; ie input_dim = 248)
        #  network_params: [[248,248], [248,248], [248,248], [248,248], [248,248], [248,248], [248,248], [248,1]]
        self.net = nn.ModuleList()
        in_dim = self.in_dim
        for idx in range(self.num_layers-1):
            print('making residual mlp layer in_dim, out_dim:', in_dim, self.out_dim)
            layer = Layer(in_dim, self.out_dim, config['mlp_dropout'])
            self.net.append(layer)
            # in_dim = self.in_dim + self.out_dim

        # For regression, the very last Layer has no normalization or activation
        print('making dense mlp layer in_dim, out_dim:', in_dim, 1)
        layer = Layer(self.out_dim, 1, normalize=False, activation=False)
        self.net.append(layer)

    def forward(self, x_in):
        x = x_in
        for i, layer in enumerate(self.net[:-1]):
            x = layer(x) + x

        logits = self.net[-1](x)

        return logits


    
#----------------------------------------------------------
# Pytorch Lightning Module that hosts a DenseMLP model
# and runs the training, validation, and testing loops
#----------------------------------------------------------
class ResidualMLP_Lightning(LightningModule):
    def __init__(self, config):
        super(ResidualMLP_Lightning, self).__init__()
        self.config = config
        self.model = ResidualMLP(config, config['block_size'])
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
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_forward(batch, batch_idx)
        self.log_dict({"loss": loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.common_forward(batch, batch_idx)
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
        filename = os.path.join(path, 'preds_residual_mlp_' + str(time.time()) + '.pkl')      
        print('saving', len(self.preds), 'preds to:', filename)
        pk.dump(self.preds, open(filename, 'wb'))

        filename = os.path.join(path, 'y_residual_mlp_' + str(time.time()) + '.pkl')      
        print('saving', len(self.y), 'y values to:', filename)
        pk.dump(self.y, open(filename, 'wb'))

        return 

    def configure_optimizers(self):
        lr = self.config['learning_rate']
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]

        


