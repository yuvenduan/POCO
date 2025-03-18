#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=512G
#SBATCH -p test
#SBATCH -e ./sbatch/preprocess.out
#SBATCH -o ./sbatch/preprocess.out
conda activate metafish
python run_preprocess.py