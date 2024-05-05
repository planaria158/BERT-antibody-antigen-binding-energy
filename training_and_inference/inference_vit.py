import os
import yaml
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.vit.vit import VIT_Lightning

#----------------------------------------------------------------------
# This file is for running inference on the Vision Transformer
# model with the scFv dataset
#----------------------------------------------------------------------
def train(args):

    # Read the config
    config_path = '../config/vit_params.yaml'  
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config = config['model_params']
    print(config)
    pl.seed_everything(config['seed'])

    #----------------------------------------------------------
    # Import the correct dataset
    #----------------------------------------------------------
    if config['image_channels'] == 1:
        from datasets.image_dataset_bw import Image_Dataset_BW as dataset
    elif config['image_channels'] == 3:
        from datasets.image_dataset_bgr import Image_Dataset_BGR as dataset

    #----------------------------------------------------------
    # Load the dataset
    #----------------------------------------------------------
    inference_dataset = dataset(config, config['inference_data_path'], inference=False, augment=False) #inference=True)
    print(inference_dataset.__len__())
    config['vocab_size'] = inference_dataset.get_vocab_size()
    print('config[vocab_size]:', config['vocab_size'], ', config[block_size]:', config['block_size'])
    inference_loader = DataLoader(inference_dataset, shuffle=False, pin_memory=True, batch_size=config['batch_size'], num_workers=config['num_workers'])

    #----------------------------------------------------------
    # Model
    #----------------------------------------------------------
    assert config['checkpoint_name'] != None, 'checkpoint_name is None'
    print('Loading pre-trained model from:', config['checkpoint_name'])
    model = VIT_Lightning.load_from_checkpoint(checkpoint_path=config['checkpoint_name'], config=config)
    total_params = sum(param.numel() for param in model.parameters())
    print('Model has:', int(total_params), 'parameters')

    #--------------------------------------------------------------------
    # Inference
    #--------------------------------------------------------------------
    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=config['log_dir'], default_hp_metric=False)

    print('Using', config['accelerator'])
    trainer = pl.Trainer(strategy='ddp', 
                         accelerator=config['accelerator'], 
                         devices=config['devices'],
                         max_epochs=config['num_epochs'],   
                         logger=logger, 
                         log_every_n_steps=config['log_every_nsteps'])   

    trainer.predict(model=model, dataloaders=inference_loader)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vit_model')
    parser.add_argument('--config', dest='config_path',
                        default='config/vit_params.yaml', type=str)
    args = parser.parse_args()
    train(args)


