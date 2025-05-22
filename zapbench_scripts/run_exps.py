import os
import itertools

# Parameters to iterate over
seeds = [0, 1, 2]
contexts = [4, 256]
models = ["poco"]
save_dir = "./experiments"
batch_size = 8
num_epochs = 25
lr_list = [3e-4]
loss_fn_list = ["L1Loss"]
compression_factor_list = [0]
weight_decay_list = [0.0001]
poyo_unit_dropout_list = [0, 0.1]
conditioning_dim_list = [1024]
decoder_context_length_list = [0, ]

# Make sure sbatch/ exists
os.makedirs("sbatch", exist_ok=True)

for seed, context, model, lr, loss_fn, cf, wd, drop, cond_dim, decoder_context_length in itertools.product(
    seeds, contexts, models, lr_list, loss_fn_list,
    compression_factor_list, weight_decay_list,
    poyo_unit_dropout_list, conditioning_dim_list, decoder_context_length_list
):
    
    if decoder_context_length != 0:
        cf = decoder_context_length // 4

    # Build full job name
    job_name = (
        f"{model}_ctx{context}_dctx{decoder_context_length}_lr{lr}_loss{loss_fn}"
        f"_cf{cf}_wd{wd}_drop{drop}_cond{cond_dim}_seed{seed}"
    )
    # Partition selection
    partition = "kempner_h100" if model == "poco" else "kempner"

    slurm_header = f"""#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account kempner_krajan_lab
#SBATCH -p {partition}
#SBATCH -e ./sbatch/{job_name}.err
#SBATCH -o ./sbatch/{job_name}.out

source ~/.bashrc
conda activate zapbench
"""

    # Command to run
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
        f" --weight_decay {wd}"
        f" --poyo_unit_dropout {drop}"
        f" --conditioning_dim {cond_dim}"
        f" --decoder_context_length {decoder_context_length}"
    )

    # Write out the job script
    script_path = f"sbatch/{job_name}.sh"
    with open(script_path, "w") as fh:
        fh.write(slurm_header)
        fh.write("\n")
        fh.write(train_cmd)
        fh.write("\n")

    # Submit it
    os.system(f"sbatch {script_path}")
