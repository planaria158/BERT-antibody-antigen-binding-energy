import os
import yaml
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.scFv_dataset import scFv_Dataset as dataset
from models.simple_mlp.mlp_model import MLP_Lightning

#----------------------------------------------------------------------
# This file is for running on the test set for MLP model
# in the Lightning framework
#----------------------------------------------------------------------
def main():
    # Read the config
    config_path = '../config/mlp_params.yaml'  
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model_config = config['model_params']
    train_config = config['train_params']    
    test_config = config['test_params']    

    print(model_config)
    print(train_config)
    print(test_config)
    pl.seed_everything(config['seed'])

    #----------------------------------------------------------
    # Load the test dataset
    #----------------------------------------------------------
    test_dataset = dataset(train_config, model_config['block_size'],
                           test_config['test_data_path'], 
                           inference=False, regularize=False)
    
    print('length of test set:', test_dataset.__len__())

    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, 
                             batch_size=train_config['batch_size'], 
                             num_workers=train_config['num_workers'])

    #----------------------------------------------------------
    # Model
    #----------------------------------------------------------
    assert test_config['checkpoint_name'] != None, 'checkpoint_name is None'
    print('Loading pre-trained model from:', test_config['checkpoint_name'])
    model = MLP_Lightning.load_from_checkpoint(checkpoint_path=test_config['checkpoint_name'],
                                               in_dim = model_config['block_size'],
                                               config=train_config)
    total_params = sum(param.numel() for param in model.parameters())
    print('Model has:', int(total_params), 'parameters')

    #--------------------------------------------------------------------
    # Test calculation
    #--------------------------------------------------------------------
    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=train_config['log_dir'], default_hp_metric=False)

    print('Using', test_config['accelerator'])
    trainer = pl.Trainer(strategy='ddp', 
                         accelerator=test_config['accelerator'], 
                         devices=test_config['devices'],
                         max_epochs=1,   
                         logger=logger, 
                         log_every_n_steps=train_config['log_every_nsteps'])   

    trainer.test(model=model, dataloaders=test_loader)
    

if __name__ == '__main__':
    main()


