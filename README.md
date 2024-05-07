


### A-Alpha Bio homework submitted by Mark Thompson. May XX, 2024


Purpose:
This study outlines creating regression models to predict binding affinites from sequence data for single-chain variable regions of human immunoglobins 

Repository for this project: https://github.com/planaria158/aAlphaBio-Homework  (I'll need to make sure it's public)

----
## Model architectures in this repository
- 4-layer MLP
- Vision Transformer: trained on 1-channel and 3-channel
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
| train-test-inference | scripts for train, test, and inference of the models    |
| Analysis.ipynb       | notebook for misc. analysis    |
| DataAnalysis.ipynb   | notebook used to create training sets from raw daya    |

----
## How to run training, test, inference jobs

- Edit the relevant config file
- cd to the train-test-inference folder
- run the desired script from the command line (no command line arguments needed)


---
## Pre-trained weights files locations

| Model  | Dataset  | Weights file |
|:----------|:----------|:----------|
| MLP    | clean-3b            | lightning_logs/mlp_model/cleaned-3b-data/trained/checkpoints    |
| VIT 1-channel   | clean-3b   | lightning_logs/vit_model/cleaned-3b-data/1-channel/trained/checkpoints    |
| VIT 3-channel  | clean-3b    | lightning_logs/vit_model/cleaned-3b-data/3-channel/trained/checkpoints    |
| Transformer    | clean-1     | lightning_logs/tform_mlp_model/cleaned-1-data/trained/checkpoints    |
| Transformer    | clean-3b    | lightning_logs/tform_mlp_model/cleaned-3b-data/trained/checkpoints    |
| Transformer    | clean-4     | lightning_logs/tform_mlp_model/cleaned-4-data/trained/checkpoints   |

