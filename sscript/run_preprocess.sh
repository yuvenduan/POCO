#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=256G
#SBATCH -p test
#SBATCH -e ./sbatch/preprocess.out
#SBATCH -o ./sbatch/preprocess.out

source ~/.bashrc
conda activate metafish
cd /n/holylabs/LABS/krajan_lab/Users/yuduan/projects/MetaFish
python run_preprocess.py