


# A-Alpha Bio homework submitted by Mark Thompson. May 7, 2024


Purpose:
This study outlines creating regression models to predict binding affinites from sequence data for single-chain variable regions of human immunoglobins 

Repository for this project: https://github.com/planaria158/aAlphaBio-Homework  (I'll need to make sure it's public)


<p align="left">
<img src="./images/predictions.png" alt="drawing" width="600"/>
</p>

----
## Model architectures

<p align="left">
  <img src="./images/model_architectures.png" alt="drawing" width="600"/>
</p>


- 4-layer MLP
- Vision Transformer: trained on 1-channel and 3-channel, with a  
- Transformer with 4-layer residual-MLP regression head


----
## Project Structure


| folder / file  | Contents  |
|:----------|:----------|
| config               | configuration files for the models    |
| data                 | raw and processed data    |
| datasets             | dataset classes    |
| inference_results    | results of running inference on holdout set    |
| lightning_logs       | all training logs and checkpoints    |
| misc_analysis        | misc files generated in t-SNE and related..    |
| models               | model classes    |
| test_results         | results of all test runs    |
| train_test_inference | scripts for train, test, and inference of the models    |
| Analysis.ipynb       | notebook for misc. analysis and t-SNE plots   |
| DataAnalysis.ipynb   | notebook used to create training sets from raw daya    |
| Homework-aAlphaBio.pptx | PowerPoint slide deck for this study

----
## How to run training, test, inference jobs

You should be able to run these commands with no changes
- Test....
- Inference....

- Edit the relevant config file
- cd to the train-test-inference folder
- run the desired script from the command line (no command line arguments needed)

An inconveniente source of human error when editing the config files.....

Files produced
- training: xxxx
- testing: xxxx
- inference: xxxx

----
## Config files

The config files are divided into 4 main sections.  The contents should be largely self-explanatory
- model_params
- train_params
- test_params
- inference_params

Some caveats however: 
- All paths are relative to the train-test-inference folder
- You may need to change `accelerator` and `devices` to suite your environment
- `checkpoint_name`  Typically this is given as `None`
    - 	training_params: checkpoint_name is given a specific value, the training job will restart from this checkpoint
    -   test_params: indicates which checkpoint to use for test run
    -   inference_params: indicates which checkpoint to use for inference run
- All outputs from test and inference calcs are deposited in the stated folder(s) in their respective sections
- The files are never overwritten, but are tagged with a trailing timestamp given as milliseconds since epoch


----
## Pre-trained weights files locations

| Model  | Dataset  | Weights file |
|:----------|:----------|:----------|
| MLP    | clean-3b            | lightning_logs/mlp_model/cleaned-3b-data/trained/checkpoints    |
| VIT 1-channel   | clean-3b   | lightning_logs/vit_model/cleaned-3b-data/1-channel/trained/checkpoints    |
| VIT 3-channel  | clean-3b    | lightning_logs/vit_model/cleaned-3b-data/3-channel/trained/checkpoints    |
| Transformer    | clean-1     | lightning_logs/tform_mlp_model/cleaned-1-data/trained/checkpoints    |
| Transformer    | clean-3b    | lightning_logs/tform_mlp_model/cleaned-3b-data/trained/checkpoints    |
| Transformer    | clean-4     | lightning_logs/tform_mlp_model/cleaned-4-data/trained/checkpoints   |

