#!/bin/bash
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=numerical_validation.out

eval "$(conda shell.bash hook)"
conda activate cerebras

python verify_mup.py --test_all --num_kv_groups 1

python verify_mup.py --test_all --num_kv_groups 4