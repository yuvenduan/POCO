import os
import itertools

# Parameters to iterate over
seeds = [0, 1, ]
contexts = [256, ]
models = ["poco", ]
save_dir = "./experiments"
batch_size = 8
num_epochs = 50
lr_list = [1e-3, ]
loss_fn_list = ["L1Loss", ]
compression_factor_list = [4, 16, 64, 128, 256]

# SLURM job header (template)
slurm_header_template = """#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account kempner_krajan_lab
#SBATCH -p {partition}
#SBATCH -e ./sbatch/slurm-%j.out
#SBATCH -o ./sbatch/slurm-%j.out

source ~/.bashrc
conda activate zapbench
"""

# Create sbatch directory if it doesn't exist
os.makedirs("sbatch", exist_ok=True)

# Generate and submit job scripts
for seed, context, model, lr, loss_fn, cf in itertools.product(seeds, contexts, models, lr_list, loss_fn_list, compression_factor_list):
    # Choose partition based on model
    partition = "kempner_h100" if model == "poco" else "kempner"
    slurm_header = slurm_header_template.format(partition=partition)
    
    job_name = f"{model}_ctx{context}_seed{seed}"
    job_script_path = f"sbatch/{job_name}.sh"

    train_cmd = (
        f"python zapbench_train.py"
        f" --seed {seed}"
        f" --context {context}"
        f" --model {model}"
        f" --save_dir {save_dir}"
        f" --batch_size {batch_size}"
        f" --num_epochs {num_epochs}"
        f" --lr {lr}"
        f" --loss_fn {loss_fn}"
        f" --compression_factor {cf}"
    )
    
    with open(job_script_path, "w") as f:
        f.write(slurm_header + "\n" + train_cmd + "\n")
    
    os.system(f"sbatch {job_script_path}")