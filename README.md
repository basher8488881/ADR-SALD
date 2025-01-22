# ADR-SALD
ADR-SALD: Attention-Based Deep Residual Sign Agnostic Learning with Derivatives for Implicit Surface Reconstruction 

# Installation 
The code is implemented and tested on Ubuntu 20.4 linux environment.

conda env create -f environment.yaml 
conda activate adr-sald
# Data 
sample data is provided in the data folder. 
# Training 
cd ./code
python training/exp_runner.py --batch_size 1 --nepoch 16000
# Testing and Evaluation
cd ./code 
python evaluate/eval.py --expname recon_vae --exps_folder_name exps --checkpoint 16000 --timestamp 2025_01_21_17_33_57 

# Aknowledgement 
This code is based on SALD (https://github.com/matanatz/SALD), thanks for this wonderful work.
