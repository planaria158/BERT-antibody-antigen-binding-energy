import pickle as pk
from pathlib import Path
import time
import os
import torch
from torch import nn
from pytorch_lightning.core import LightningModule
from models.bert.fine_bert_model import fineBERT

#----------------------------------------------------------
# Pytorch Lightning Module that hosts the fineBERT model
# and runs the training, validation, and testing loops
#----------------------------------------------------------
class fineBERT_Lightning(LightningModule):
    def __init__(self, config):
        super(fineBERT_Lightning, self).__init__()
        self.config = config
        self.model = fineBERT(config)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def common_forward(self, batch):
        x, y = batch
        y_hat = self.model(x)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_norm_clip'])
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_forward(batch)
        self.log_dict({"loss": loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, _, _ = self.common_forward(batch)
        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return val_loss

    def on_predict_start(self):
        self.preds = []
        self.y = []

    def predict_step(self, batch, batch_idx):
        y_hat = self.forward(batch[0]) #.float())
        self.y.extend(batch[1].cpu().numpy().tolist())
        self.preds.extend(y_hat.cpu().numpy().tolist())
        return

    def on_predict_end(self):
        # save the preds to file
        path = Path(self.config['inference_results_folder'])
        path.mkdir(parents=True, exist_ok=True)
        filename = os.path.join(path, 'preds_bert_' + str(time.time()) + '.pkl')      
        print('saving', len(self.preds), 'preds to:', filename)
        pk.dump(self.preds, open(filename, 'wb'))

        filename = os.path.join(path, 'y_bert_' + str(time.time()) + '.pkl')      
        print('saving', len(self.y), 'y values to:', filename)
        pk.dump(self.y, open(filename, 'wb'))
        return 


    def configure_optimizers(self):
        lr = self.config['learning_rate']
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]
