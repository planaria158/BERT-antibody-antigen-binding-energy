import os
import yaml
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.vit.vit import VIT_Lightning

#----------------------------------------------------------------------
# This file is for training the simple VIT model
#----------------------------------------------------------------------
def train(args):
    # Read the config
    config_path = './config/vit_params.yaml'  
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config = config['model_params']
    print(config)
    pl.seed_everything(config['seed'])


    #----------------------------------------------------------
    # Load the dataset
    #----------------------------------------------------------
    if config['image_channels'] == 1:
        from datasets.cnn_dataset_bw import CNN_Dataset_BW as dataset
    elif config['image_channels'] == 3:
        from datasets.cnn_dataset_bgr import CNN_Dataset_BGR as dataset

    train_data_path = config['train_data_path']  
    train_dataset = dataset(config, train_data_path)
    print(train_dataset.__len__())
    config['vocab_size'] = train_dataset.get_vocab_size()
    print('config[vocab_size]:', config['vocab_size'], ', config[block_size]:', config['block_size'])

    test_data_path = config['test_data_path'] 
    test_dataset = dataset(config, test_data_path)
    print(test_dataset.__len__())
    
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=config['batch_size'], 
                              num_workers=config['num_workers'], persistent_workers=True)
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=config['batch_size'], 
                             num_workers=5, persistent_workers=True)

    #----------------------------------------------------------
    # Model
    #----------------------------------------------------------
    model = VIT_Lightning(config) 
    total_params = sum(param.numel() for param in model.parameters())
    print('Model has:', int(total_params), 'parameters')

    #--------------------------------------------------------------------
    # Training
    #--------------------------------------------------------------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=config['save_top_k'],
        every_n_train_steps=config['checkpoint_every_n_train_steps'],
        save_on_train_epoch_end=True,
        monitor = config['monitor'],
        mode = config['mode']
    )

    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=config['log_dir'], default_hp_metric=False)

    print('Using', config['accelerator'])
    trainer = pl.Trainer(#strategy='ddp', 
                         accelerator=config['accelerator'], 
                         devices=config['devices'],
                         max_epochs=config['num_epochs'],   
                         logger=logger, 
                         log_every_n_steps=config['log_every_nsteps'], 
                         callbacks=[checkpoint_callback])   


    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vit_model')
    parser.add_argument('--config', dest='config_path',
                        default='config/vit_params.yaml', type=str)
    args = parser.parse_args()
    train(args)


