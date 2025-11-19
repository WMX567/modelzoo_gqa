#!/bin/bash
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=160GB
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=train_sophiam.out

eval "$(conda shell.bash hook)"

python train_mup_modelzoo.py --mode suite --backend GPU
