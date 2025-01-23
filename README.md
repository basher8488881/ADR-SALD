# ADR-SALD
ADR-SALD: Attention-Based Deep Residual Sign Agnostic Learning with Derivatives for Implicit Surface Reconstruction 
## Reconstruction Preview
![plot](https://github.com/basher8488881/ADR-SALD/blob/main/shapenet_imgAll_1.png)
# Installation 
The code is implemented and tested on Ubuntu 20.4 linux environment.\
cd ./code \
conda env create -f environment.yaml \
conda activate adr-sald 
# Data 
sample data is provided in the data folder. To change the data path, please go to this directory <br/>
cd ./code/confs <br/>

Open this file <br/>

recon.conf <br/>

replace the "dataset_path" to a new file path

# Training 
cd ./code \
python training/exp_runner.py --batch_size 1 --nepoch 16000
# Generation and Evaluation
cd ./code \
python evaluate/eval.py --expname recon_vae --exps_folder_name exps --checkpoint 16000 --timestamp 2025_01_21_17_33_57 \

Generated mesh and evaluation result can be found in exps folder 

# Aknowledgement 
This code is based on SALD (https://github.com/matanatz/SALD), thanks for this wonderful work.
