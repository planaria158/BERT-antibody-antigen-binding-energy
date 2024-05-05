import os
import yaml
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.tform_mlp.tform_mlp import TFormMLP_Lightning
from datasets.scFv_dataset import scFv_Dataset as dataset

#----------------------------------------------------------------------
# This file is for training the Transformer-residualMLP model
#----------------------------------------------------------------------
def train(args):
    # Read the config
    config_path = '../config/tform_mlp_params.yaml'  
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config = config['model_params']
    print(config)
    pl.seed_everything(config['seed'])

    #----------------------------------------------------------
    # Load the dataset and dataloaders
    #----------------------------------------------------------
    train_dataset = dataset(config, config['train_data_path'], regularize=config['sequence_regularize'])
    print(train_dataset.__len__())
    config['vocab_size'] = train_dataset.get_vocab_size()
    print('config[vocab_size]:', config['vocab_size'], ', config[block_size]:', config['block_size'])

    test_dataset = dataset(config, config['test_data_path'], regularize=False)
    print(test_dataset.__len__())
    
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=config['batch_size'], 
                              num_workers=config['num_workers'], persistent_workers=True)
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=config['batch_size'], 
                             num_workers=5, persistent_workers=True)

    #----------------------------------------------------------
    # Model
    #----------------------------------------------------------
    if config['checkpoint_name'] != 'None':
        print('Restarting from checkpoint: ', config['checkpoint_name'])
        path = config['checkpoint_name']
        model = TFormMLP_Lightning.load_from_checkpoint(checkpoint_path=path, config=config)
    else:
        print('Starting from new model instance')
        model = TFormMLP_Lightning(config) 

    total_params = sum(param.numel() for param in model.parameters())
    print('Model has:', int(total_params), 'parameters')

    #--------------------------------------------------------------------
    # Training
    #--------------------------------------------------------------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch}-{step}-{val_loss:.2f}-{loss:.2f}',
        save_top_k=config['save_top_k'],
        every_n_train_steps=config['checkpoint_every_n_train_steps'],
        save_on_train_epoch_end=True,
        monitor = config['monitor'],
        mode = config['mode']
    )

    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=config['log_dir'], default_hp_metric=False)

    print('Using', config['accelerator'])
    trainer = pl.Trainer(strategy='ddp', 
                         accelerator=config['accelerator'], 
                         devices=config['devices'],
                         max_epochs=config['num_epochs'],   
                         logger=logger, 
                         log_every_n_steps=config['log_every_nsteps'], 
                         callbacks=[checkpoint_callback],)   


    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for tform_mlp')
    parser.add_argument('--config', dest='config_path',
                        default='config/tform_mlp_params.yaml', type=str)
    args = parser.parse_args()
    train(args)


