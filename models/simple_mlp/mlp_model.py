
import pickle as pk
from pathlib import Path
import os
import time
import torch
from torch import nn
from pytorch_lightning.core import LightningModule
from models.transformer_parts.transformer_parts import MLP

"""
    Pytorch Lightning Module that hosts a simple MLP model
    and runs the training, validation, and testing loops

    Args:
        config: dictionary containing the configuration parameters
            block_size: int, the size of the input block
            mlp_dropout: float, the dropout rate for the MLP
            learning_rate: float, the learning rate for the optimizer
            betas: tuple, the betas for the optimizer
            lr_gamma: float, the gamma for the scheduler
            inference_results_folder: string, the folder where the inference results will be saved
"""
class MLP_Lightning(LightningModule):
    def __init__(self, config):
        super(MLP_Lightning, self).__init__()
        self.config = config
        self.model = MLP(config['block_size'], config['mlp_dropout'])
        self.criteriion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def common_forward(self, batch, batch_idx):
        x, y = batch
        x = x.float()
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
        self.inference_criterion = nn.MSELoss(reduction='none')
        self.preds = []
        self.y = []
        self.loss = []

    def predict_step(self, batch, batch_idx):
        y_hat = self.forward(batch[0].float())
        self.y.extend(batch[1].cpu().numpy().tolist())
        self.preds.extend(y_hat.cpu().numpy().tolist())

        loss = self.inference_criterion(y_hat, batch[1].float())
        self.loss.extend(loss.cpu().numpy().tolist())
        return

    def on_predict_end(self):
        # save the preds to file
        path = Path(self.config['inference_results_folder'])
        path.mkdir(parents=True, exist_ok=True)
        filename = os.path.join(path, 'preds_mlp_' + str(time.time()) + '.pkl')      
        print('saving', len(self.preds), 'preds to:', filename)
        pk.dump(self.preds, open(filename, 'wb'))

        filename = os.path.join(path, 'y_mlp_' + str(time.time()) + '.pkl')      
        print('saving', len(self.y), 'y values to:', filename)
        pk.dump(self.y, open(filename, 'wb'))

        filename = os.path.join(path, 'loss_mlp_' + str(time.time()) + '.pkl')  
        print('saving', len(self.loss), 'loss values to:', filename)
        pk.dump(self.loss, open(filename, 'wb'))
        return 

    def configure_optimizers(self):
        lr = self.config['learning_rate']
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]

        


