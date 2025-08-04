import os
import subprocess
import argparse
import logging
import time

import configs.experiments as experiments
import configs.exp_analysis as exp_analysis
import torch

from utils.config_utils import save_config, configs_dict_unpack
from train import model_train, model_eval
from configs.config_global import LOG_LEVEL, ROOT_DIR
from configs.configs import BaseConfig

def train_cmd(config_):
    arg = '\'' + config_.save_path + '\''
    command = r'''python -c "import train; train.train_from_path(''' + arg + ''')"'''
    return command


def eval_cmd(eval_save_path):
    arg = '\'' + eval_save_path + '\''
    command = r'''python -c "import train; train.eval_from_path(''' + arg + ''')"'''
    return command


def analysis_cmd(exp_name):
    arg = exp_name + '_analysis()'
    command = r'''python -c "import configs.exp_analysis as exp_analysis; exp_analysis.''' + arg + '''"'''
    return command


def get_jobfile(cmd, job_name, dep_ids, 
                sbatch_path='./sbatch/', hours=8, partition='normal', 
                mem=32, cpu=2, account='kempner_krajan_lab'):
    """
    Create a job file.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        cmd: python command to be execute by the cluster
        job_name: str, name of the job file
        dep_ids: list, a list of job ids used for job dependency
        sbatch_path : str, Directory to store SBATCH file in.
        hours : int, number of hours to train
    Returns:
        job_file : str, Path to the job file.
    """
    assert type(dep_ids) is list, 'dependency ids must be list'
    assert all(type(id_) is str for id_ in dep_ids), 'dependency ids must all be strings'

    if len(dep_ids) == 0:
        dependency_line = ''
    else:
        dependency_line = '#SBATCH --dependency=afterok:' \
                          + ':'.join(dep_ids) + '\n'

    os.makedirs(sbatch_path, exist_ok=True)
    job_file = os.path.join(sbatch_path, job_name + '.sh')

    with open(job_file, 'w') as f:
        f.write(
            '#!/bin/bash\n'
            + '#SBATCH -t {}:00:00\n'.format(hours)
            + '#SBATCH -N 1\n'
            + '#SBATCH -n {}\n'.format(cpu)
            + '#SBATCH --mem={}G\n'.format(mem)
            + '#SBATCH --gres=gpu:1\n'
            + f'#SBATCH --account {account}\n'
            + '#SBATCH -p {}\n'.format(partition)
            + '#SBATCH -e ./sbatch/slurm-%j.out\n'
            + '#SBATCH -o ./sbatch/slurm-%j.out\n'
            + dependency_line
            + '\n'
            + 'source ~/.bashrc\n'
            + 'conda activate metafish\n'
            + f'cd {ROOT_DIR}\n'
            + cmd + '\n'
            + '\n'
            )
        print(job_file)
    return job_file

def get_idle_gpu(subprocess_list):
    while True:
        for idx, pipe in enumerate(subprocess_list):
            if pipe is not None and pipe.poll() is not None:
                print("Experiment completed with code", pipe.poll(), flush=True)
                pipe = None
            if pipe == None:
                return idx
        time.sleep(10)

def train_experiment(experiment, on_cluster, on_server, partition, account, overwrite=False, eval=False):
    """Train model across platforms given experiment name.
    adapted from https://github.com/gyyang/olfaction_evolution
    Args:
        experiment: str, name of experiment to be run
            must correspond to a function in experiments.py
        on_cluster: bool, whether to run experiments on cluster
        on_server: bool, if use on server

    Returns:
        return_ids: list, a list of job ids that are last in the dependency sequence
            if not using cluster, return is an empty list
    """
    if experiment in dir(experiments):
        # Get list of configurations from experiment function
        exp_configs = getattr(experiments, experiment)()
    else:
        raise ValueError('Experiment config not found: ', experiment)

    return_ids = []
    
    # exp_configs is a list of configs
    exp_configs = configs_dict_unpack(exp_configs)
    assert isinstance(exp_configs[0], BaseConfig), \
        'exp_configs should be list of configs'

    if not eval:
        count = 0
        for config in exp_configs:
            config.overwrite = overwrite
            if save_config(config, config.save_path):
                count += 1

        # prompt if user wants to continue
        if count == 0:
            logging.info('All experiments already completed for {:s}'.format(experiment))
            return return_ids
        elif input(f'Continue running {count} jobs? (y/n)') != 'y':
            return return_ids
    else:
        for config in exp_configs:
            config.overwrite = False
            if save_config(config, config.save_path, show_message=False):
                logging.info(f'Training not done at {config.save_path}')

    if on_cluster:
        for config in exp_configs:
            if not save_config(config, config.save_path, show_message=False) and not eval:
                continue
            python_cmd = train_cmd(config) if not eval else eval_cmd(config.save_path)
            job_n = config.experiment_name + '_' + config.model_name
            cp_process = subprocess.run(['sbatch', get_jobfile(python_cmd,
                                                                job_n,
                                                                dep_ids=[],
                                                                hours=config.hours, 
                                                                partition=partition,
                                                                mem=config.mem,
                                                                cpu=config.cpu,
                                                                sbatch_path=config.save_path,
                                                                account=account
                                                                )],
                                        capture_output=True, check=True)
            cp_stdout = cp_process.stdout.decode()
            print(cp_stdout)
            job_id = cp_stdout[-9:-1]
            return_ids.append(job_id)

    elif on_server:
        num_gpus = torch.cuda.device_count()
        print(f"Running on {num_gpus} gpu(s) ...", flush=True)
        subprocesses = [None for _ in range(num_gpus)]

        for config in exp_configs:
            if not save_config(config, config.save_path, show_message=False) and not eval:
                continue
            id = get_idle_gpu(subprocesses)
            cmd = f'export CUDA_VISIBLE_DEVICES={id}' + '\n' + (train_cmd(config) if not eval else eval_cmd(config.save_path))
            out_path = os.path.join(config.save_path, 'out.txt')
            out_file = open(out_path, mode='w')
            subprocesses[id] = subprocess.Popen(cmd, stdout=out_file, stderr=out_file, shell=True)

        for pipe in subprocesses:
            if pipe is not None:
                if pipe.poll() is None:
                    print("Experiment completed with code", pipe.wait())
                else:
                    print("Experiment completed with code", pipe.poll())

    else:
        for config in exp_configs:
            if eval or save_config(config, config.save_path, show_message=False):
                model_train(config) if not eval else model_eval(config)

    return return_ids

def analyze_experiment(experiment, prev_ids, on_cluster, partition, account):
    """analyze experiments
     adapted from https://github.com/gyyang/olfaction_evolution

     Args:
         experiment: str, name of experiment to be analyzed
             must correspond to a function in exp_analysis.py
         prev_ids: list, list of previous job ids
         on_cluster: bool, if use on the cluster
     """
    print('Analyzing {:s} experiment'.format(experiment))
    if (experiment + '_analysis') in dir(exp_analysis):
        if on_cluster:
            python_cmd = analysis_cmd(experiment)
            job_n = experiment + '_analysis'
            slurm_cmd = ['sbatch', get_jobfile(python_cmd, job_n,
                                               dep_ids=prev_ids, partition=partition,
                                               hours=24, mem=64, account=account)]
            cp_process = subprocess.run(slurm_cmd, capture_output=True,
                                        check=True)
            cp_stdout = cp_process.stdout.decode()
            print(cp_stdout)
        else:
            getattr(exp_analysis, experiment + '_analysis')()
    else:
        raise ValueError('Experiment analysis not found: ', experiment + '_analysis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', nargs='+', help='Train experiments', default=[])
    parser.add_argument('-e', '--eval', nargs='+', help='Evaluate experiments', default=[])
    parser.add_argument('-a', '--analyze', nargs='+', help='Analyze experiments', default=[])
    parser.add_argument('-c', '--cluster', action='store_true', help='Use batch submission on Slurm cluster')
    parser.add_argument('-s', '--server', action='store_true', help='Run experiments on a Linux server with multiple GPUs')
    parser.add_argument('-p', '--partition', default='kempner', help='Partition of resource on cluster to use')
    parser.add_argument('--acc', default='kempner_krajan_lab', help='Account to use on cluster')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing complete runs')
    
    args = parser.parse_args()
    experiments2train = args.train
    experiments2analyze = args.analyze
    experiments2eval = args.eval
    use_cluster = args.cluster
    use_server = args.server
    assert not (use_cluster and use_server), "Either use cluster or directly run on a server"

    logging.basicConfig(level=LOG_LEVEL)

    # evaluation jobs are executed after training jobs,
    # analysis jobs are executed after training and evaluation jobs.
    train_ids = []
    if experiments2train:
        for exp in experiments2train:
            exp_ids = train_experiment(exp, on_cluster=use_cluster, on_server=use_server, partition=args.partition, account=args.acc, overwrite=args.overwrite)
            train_ids += exp_ids

    eval_ids = []
    if experiments2eval:
        for exp in experiments2eval:
            exp_ids = train_experiment(exp, on_cluster=use_cluster, on_server=use_server, partition=args.partition, account=args.acc, eval=True)
            eval_ids += exp_ids

    if experiments2analyze:
        for exp in experiments2analyze:
            analyze_experiment(exp, prev_ids=train_ids + eval_ids,
                               on_cluster=use_cluster, partition=args.partition, account=args.acc)