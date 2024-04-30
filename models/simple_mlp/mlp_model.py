
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
        self.norm = nn.LayerNorm(out_dim) if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
        self.activation = nn.ReLU() if activation else nn.Identity()

    def forward(self, x):
        out = self.dropout(self.activation(self.norm(self.linear(x))))
        return out

"""
    Vanilla MLP
    The regular mlp is a 3-layer mlp: (TabTransformer paper)
        input layer (input_dim, 4*input_dim)
        hidden layer(4*input_dim, 2*input_dim)
        output layer(2*input_dim, 1)
"""
class MLP(nn.Module):
    def __init__(self, config, input_dim):
        super(MLP, self).__init__()
        print('Regression head is MLP')
        mlp_hidden_mults = (4, 4, 3, 2) # hardwired 

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
                layer = Layer(in_dim, out_dim, config['mlp_dropout'])
            
            layers.append(layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x_in):
        return self.net(x_in) 

    
#----------------------------------------------------------
# Pytorch Lightning Module that hosts a simple MLP model
# and runs the training, validation, and testing loops
#----------------------------------------------------------
class MLP_Lightning(LightningModule):
    def __init__(self, config):
        super(MLP_Lightning, self).__init__()
        self.config = config
        self.model = MLP(config, config['block_size'])
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

    def configure_optimizers(self):
        lr = self.config['learning_rate']
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]

        


