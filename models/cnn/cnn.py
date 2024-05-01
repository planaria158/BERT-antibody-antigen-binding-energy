
import pickle as pk
from pathlib import Path
import os
import time
import torch
from torch import nn
from pytorch_lightning.core import LightningModule

#----------------------------------------------------------
# Create a simple 4-layer CNN regression model for  
# 2D B&W images (single channel)
#----------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, config, block_size):
        super(CNN, self).__init__()
        self.config = config
        self.block_size = block_size  ???
        self.conv1 = nn.Conv2d(1,    32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32,   64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * self.block_size * self.block_size, 512)
        self.fc2 = nn.Linear(512, 1)
        self.gelu = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # assuming input image is 46x46
        x = self.pool(self.gelu(self.conv1(x)))  # 23x23
        x = self.pool(self.gelu(self.conv2(x)))  # 11x11
        x = self.pool(self.gelu(self.conv3(x)))  # 5x5
        x = self.pool(self.gelu(self.conv4(x)))  # 2x2
        x = x.view(-1, 256 * self.block_size * self.block_size)
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


#----------------------------------------------------------
# Pytorch Lightning Module that hosts a simple CNN model
# and runs the training, validation, and testing loops
#----------------------------------------------------------
class CNN_Lightning(LightningModule):
    def __init__(self, config):
        super(CNN_Lightning, self).__init__()
        self.config = config
        self.model = CNN(config, config['block_size'])
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
        filename = os.path.join(path, 'preds_cnn_' + str(time.time()) + '.pkl')      
        print('saving', len(self.preds), 'preds to:', filename)
        pk.dump(self.preds, open(filename, 'wb'))

        filename = os.path.join(path, 'y_cnn_' + str(time.time()) + '.pkl')      
        print('saving', len(self.y), 'y values to:', filename)
        pk.dump(self.y, open(filename, 'wb'))
        return 

    def configure_optimizers(self):
        lr = self.config['learning_rate']
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]

        


