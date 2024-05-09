import os
import yaml
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.scFv_dataset_v2 import scFv_Dataset_v2 as dataset
from models.tform_mlp_v2 import TFormMLP_Lightning_v2 

#----------------------------------------------------------------------
# This file is for running inference with the Transformer model
#
# The output from this will be a csv file with the predictions in 
# the same order as the input dataset file.
# 
# The output file will be saved in the inference_results_folder
# found in the config file
#----------------------------------------------------------------------
def main():

    # Read the config
    config_path = '../config/tform_mlp_params.yaml'  
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model_config = config['model_params']
    train_config = config['train_params']    
    inference_config = config['inference_params']    

    print('model_config:', model_config)
    print('\ntrain_config:', train_config)
    print('\ninference_config:', inference_config)
    pl.seed_everything(config['seed'])

    #----------------------------------------------------------
    # Load the dataset and dataloaders
    #----------------------------------------------------------
    print('Loading inference dataset from:', inference_config['inference_data_path'])
    inference_dataset = dataset(train_config, model_config['block_size'],
                                inference_config['inference_data_path'], 
                                inference=True,
                                regularize=train_config['sequence_regularize'])
    
    print('length of inference dataset:', inference_dataset.__len__())

    inference_loader = DataLoader(inference_dataset, shuffle=False, 
                                  batch_size=train_config['batch_size'])


    #----------------------------------------------------------
    # Model
    #----------------------------------------------------------
    assert inference_config['checkpoint_name'] != None, 'checkpoint_name is None'
    print('Loading checkpoint: ', inference_config['checkpoint_name'])
    model = TFormMLP_Lightning_v2.load_from_checkpoint(checkpoint_path=inference_config['checkpoint_name'], 
                                                    model_config=model_config,
                                                    config=train_config)
    model.eval() # just to be darn sure...
    total_params = sum(param.numel() for param in model.parameters())
    print('Model has:', int(total_params), 'parameters')

    #--------------------------------------------------------------------
    # Inference
    #--------------------------------------------------------------------
    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=train_config['log_dir'], default_hp_metric=False)

    print('Using', inference_config['accelerator'])
    trainer = pl.Trainer(strategy='ddp', 
                         accelerator=inference_config['accelerator'], 
                         devices=inference_config['devices'],
                         logger=logger)   

    trainer.predict(model=model, dataloaders=inference_loader)
    

if __name__ == '__main__':
    main()


