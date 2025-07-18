#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=50G
#SBATCH -p kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --account kempner_krajan_lab
#SBATCH -e ./sbatch/preprocess.out
#SBATCH -o ./sbatch/preprocess.out
conda activate metafish
python run_preprocess.py