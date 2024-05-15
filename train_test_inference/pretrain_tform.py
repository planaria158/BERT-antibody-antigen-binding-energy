import os
import yaml
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.tform import TFormMLP_Lightning
from datasets.scFv_diy_dataset_pretrain import scFv_diy_pretrain_Dataset as dataset

#----------------------------------------------------------------------
# This file is for pretraining the Transformer-residualMLP model
# in masked language model training mode
#----------------------------------------------------------------------
def main():
    # Read the config
    config_path = '../config/tform_params.yaml'  
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model_config = config['model_params']
    train_config = config['train_params']    

    print(model_config)
    print(train_config)
    pl.seed_everything(config['seed'])

    #----------------------------------------------------------
    # Load the dataset and dataloaders
    #----------------------------------------------------------
    train_dataset = dataset(train_config, 
                            model_config['block_size'],
                            train_config['train_data_path'], 
                            regularize=train_config['sequence_regularize'])

    val_dataset = dataset(train_config, 
                          model_config['block_size'],
                          train_config['val_data_path'] , 
                          regularize=False)

    print('length of training set:', train_dataset.__len__())
    print('length of validation set:', val_dataset.__len__())
    
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, 
                              batch_size=train_config['batch_size'], 
                              num_workers=train_config['num_workers'],
                              persistent_workers=True,
                              drop_last=True) # sometimes the very last batch is size 1
                                              # and batchnorm layers don't like that)
    
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, 
                            batch_size=train_config['batch_size'], 
                            num_workers=train_config['num_workers'],
                            persistent_workers=True)

    #----------------------------------------------------------
    # Model
    #----------------------------------------------------------
    if train_config['checkpoint_name'] != 'None':
        print('Restarting from checkpoint: ', train_config['checkpoint_name'])
        model = TFormMLP_Lightning.load_from_checkpoint(checkpoint_path=train_config['checkpoint_name'], 
                                                        model_config=model_config,
                                                        config=train_config,
                                                        strict=False)
    else:
        print('Starting from new model instance')
        model = TFormMLP_Lightning(model_config, train_config) 

    total_params = sum(param.numel() for param in model.parameters())
    print('Model has:', int(total_params), 'parameters')

    #--------------------------------------------------------------------
    # Training
    #--------------------------------------------------------------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch}-{step}-{val_loss:.2f}-{loss:.2f}',
        save_top_k=train_config['save_top_k'],
        every_n_train_steps=train_config['checkpoint_every_n_train_steps'],
        save_on_train_epoch_end=True,
        monitor = train_config['monitor'],
        mode = train_config['mode']
    )

    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=train_config['log_dir'], default_hp_metric=False)

    print('Using', train_config['accelerator'])
    trainer = pl.Trainer(#strategy='ddp', 
                         accelerator=train_config['accelerator'], 
                         devices=train_config['devices'],
                         max_epochs=train_config['num_epochs'],   
                         logger=logger, 
                         log_every_n_steps=train_config['log_every_nsteps'], 
                         callbacks=[checkpoint_callback],)   


    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print('Done!!')
    

if __name__ == '__main__':
    main()


