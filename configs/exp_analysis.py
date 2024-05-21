import os
import os.path as osp
import copy
import logging
import numpy as np
import pandas as pd
import torch

from configs.config_global import ROOT_DIR, LOG_LEVEL
from utils.config_utils import configs_dict_unpack
from analysis import plots
from configs import experiments
from datasets.zebrafish import get_baseline_performance

def get_curve(cfgs, idx, key='normalized_reward', max_batch=None, num_points=None, start=0):

    num_seeds = len(cfgs)
    
    if max_batch is None:
        max_batch = cfgs[0][idx].max_batch
    
    eval_interval = cfgs[0][idx].log_every
    tot_steps = max_batch // eval_interval
    if num_points is None:
        num_points = tot_steps
    plot_every = tot_steps // num_points
    performance = []

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]

        try:
            file_path = osp.join(cfg.save_path, 'progress.txt')
            exp_data = pd.read_table(file_path)
            acc = exp_data[key][start: tot_steps + 1: plot_every]
            
            if len(acc) >= (tot_steps - start) // plot_every:
                performance.append(acc)
            else:
                raise ValueError
        except:
            print("Incomplete data:", cfg.save_path)

    performance = np.stack(performance, axis=1)
    x_axis = np.arange((start + 1) * eval_interval, max_batch + 1, plot_every * eval_interval)

    return performance, x_axis

def get_performance(
    cfgs, idx, 
    key1='TrainLoss', key2='TestError', 
    check_complete=True, file_name='progress.txt', 
    beg=0, cfgs2=None
):
    """
    Look at the the step where key1 in minimal and return the corresponding key2 value, an example is to
    look at the test accuracy when the training loss is minimal.
    :return: a list of performance values for each seed
    """

    num_seeds = len(cfgs)
    performance = []

    max_batch = cfgs[0][idx].max_batch
    eval_interval = cfgs[0][idx].log_every
    tot_steps = max_batch // eval_interval

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]

        if cfgs2 is not None:
            cfg2 = cfgs2[seed][idx]

        try:
            file_path = osp.join(cfg.save_path, file_name)
            exp_data = pd.read_table(file_path)

            if check_complete and len(exp_data[key1]) < tot_steps:
                raise ValueError(exp_data)

            if cfgs2 is not None:
                max_batch2 = cfgs2[0][idx].max_batch
                eval_interval2 = cfgs2[0][idx].log_every
                tot_steps2 = max_batch2 // eval_interval2
                file_path2 = osp.join(cfg2.save_path, 'progress.txt')
                exp_data2 = pd.read_table(file_path2)

                if len(exp_data2[key1]) < tot_steps2:
                    raise ValueError(exp_data)

            best_t = exp_data[key1][beg: ].argmin()
            performance.append(exp_data[key2][best_t + beg])

        except:
            print("Incomplete data:", cfg.save_path)

    return performance

def get_lr_curve(cfgs, idx, file_name='lr_info.pth'):

    num_seeds = len(cfgs)
    info = []

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]

        try:
            file_path = osp.join(cfg.save_path, file_name)
            mean, std = torch.load(file_path)
            info.append(np.array(mean))
        except:
            print("Incomplete data:", cfg.save_path)

    info = np.stack(info, axis=1)
    return info

def autoregressive_rnns_analysis(cfgs = experiments.autoregressive_rnns(), save_dir = 'autoregressive_rnns', mode_list = ['minmax', 'zscore', ]):
    
    data_mode_list = ['Shared', ]
    model_list = ['S4', 'LSTM', 'Transformer', 'RNN', ]
    performance = []

    for i, mode in enumerate(mode_list):
        val_baseline_performance = get_baseline_performance(cfgs[0][i * len(data_mode_list) * len(model_list)], 'val')

        for j, data_mode in enumerate(data_mode_list):

            if data_mode == 'Individual':
                continue

            performance = []
            train_performance = []

            for k, model in enumerate(model_list):
                test_curve, x_axis = get_curve(cfgs, k + (i * len(data_mode_list) + j) * len(model_list), key='val_end_loss')
                performance.append(test_curve)
                train_curve, x_axis = get_curve(cfgs, k + (i * len(data_mode_list) + j) * len(model_list), key='TrainLoss')
                train_performance.append(train_curve)

                def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
                    plt.plot(x_axis, [val_baseline_performance] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

                # compare train and test curves
                plots.error_plot(
                    x_axis,
                    [train_curve, test_curve],
                    x_label='Training Step',
                    y_label='Train Loss',
                    label_list=['Train', 'Test'],
                    save_dir=save_dir,
                    fig_name=f'all_fish_{mode}_{data_mode}_{model}',
                    figsize=(5, 4),
                    mode='errorshade',
                    extra_lines=draw_baseline
                )

            plots.error_plot(
                x_axis,
                train_performance,
                x_label='Training Step',
                y_label='Train Loss',
                label_list=model_list,
                save_dir=save_dir,
                fig_name=f'all_fish_{mode}_{data_mode}_train',
                figsize=(5, 4),
                mode='errorshade',
            )

            plots.error_plot(
                x_axis,
                performance,
                x_label='Training Step',
                y_label='Validation Loss',
                label_list=model_list,
                save_dir=save_dir,
                fig_name=f'all_fish_{mode}_{data_mode}_val',
                figsize=(5, 4),
                mode='errorshade',
                extra_lines=draw_baseline
            )

def autoregressive_rnns_sim_analysis():
    cfgs = experiments.autoregressive_rnns_sim()
    autoregressive_rnns_analysis(cfgs, 'autoregressive_rnns_sim', mode_list=['minmax'])

def sim_test_analysis():
    cfgs = experiments.sim_test()
    n_neurons =  [200, 500, 1500, 3000, ]
    hidden_sizes = [128, 512, 2048, ]
    for i, n in enumerate(n_neurons):

        val_baseline_performance = get_baseline_performance(cfgs[0][i], 'val')
        def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
            plt.plot(x_axis, [val_baseline_performance] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)
        val_curves = []

        for j, hidden_size in enumerate(hidden_sizes):
            test_curve, x_axis = get_curve(cfgs, i + j * len(n_neurons), key='val_end_loss')
            train_curve, x_axis = get_curve(cfgs, i + j * len(n_neurons), key='train_end_loss')
            val_curves.append(test_curve)
            
            # compare train and test curves
            plots.error_plot(
                x_axis,
                [train_curve, test_curve],
                x_label='Training Step',
                y_label='Train Loss',
                label_list=['Train', 'Test'],
                save_dir='sim_test',
                fig_name=f'{n}_{hidden_size}',
                figsize=(5, 4),
                mode='errorshade',
                extra_lines=draw_baseline
            )

        plots.error_plot(
            x_axis,
            val_curves,
            x_label='Training Step',
            y_label='Validation Loss',
            label_list=hidden_sizes,
            legend_title='Hidden Size',
            save_dir='sim_test',
            fig_name=f'{n}_val',
            figsize=(5, 4),
            mode='errorshade',
            extra_lines=draw_baseline
        )
    
def tca_analysis():
    from analysis.tca import tca
    cfgs = experiments.linear_rnn_test()

    for idx, exp_type in enumerate(['control', 'shocked', 'reshocked']):
        for seed in range(2):
            for j, hidden_size in enumerate([8, 64, 512]):
                if hidden_size < 512:
                    tca(cfgs[seed][j * 3 + idx])

def meta_rnn_compare_test_step_analysis():
    cfgs = experiments.meta_rnn()
    lr_list = [30, 100, ]
    algo_list = ['none', 'maml']
    step_list = [1, 2, 5, 10, 20]

    for i_lr, lr in enumerate(lr_list):

        train_curve_list = []
        val_curve_list = []

        for i_algo, algo in enumerate(algo_list):

            performances = []
            for i, step in enumerate(step_list):
                loss = get_performance(cfgs, i + (i_algo * len(lr_list) + i_lr) * len(step_list), beg=0, key1='val_end_loss', key2='val_end_loss')
                performances.append(loss)
            val_curve_list.append(performances)

        val_baseline_performance = get_baseline_performance(cfgs[0][0], 'val')
        def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
            plt.plot(x_axis, [val_baseline_performance] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

        plots.error_plot(
            step_list,
            val_curve_list,
            label_list=algo_list,
            x_label='Inner Training Step',
            y_label='Prediction Loss',
            save_dir='meta_rnn',
            fig_name=f'compare_test_step_{lr}',
            figsize=(5, 4),
            mode='errorshade',
            extra_lines=draw_baseline
        )

def seqvae_analysis():

    cfgs = experiments.seqvae_test()
    algo_list = ['Shared', 'Seperate']
    coef_list = [0, 0.002, 0.01, 0.1, 0.3]

    train_curve_list = []
    val_curve_list = []

    val_baseline_performance = get_baseline_performance(cfgs[0][0], 'val')
    def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
        plt.plot(x_axis, [val_baseline_performance] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

    for i_algo, algo in enumerate(algo_list):
        performances = []
        for i, coef in enumerate(coef_list):
            index = i + i_algo * len(coef_list)
            loss = get_performance(cfgs, index, beg=0, key1='val_end_loss', key2='val_end_loss')
            performances.append(loss)

            test_curve, x_axis = get_curve(cfgs, index, key='val_end_loss')
            train_curve, x_axis = get_curve(cfgs, index, key='train_end_loss')

            # compare train and test curves
            plots.error_plot(
                x_axis,
                [train_curve, test_curve],
                x_label='Training Step',
                y_label='Train Loss',
                label_list=['Train', 'Test'],
                save_dir='seq_vae',
                fig_name=f'{algo}_{coef}',
                figsize=(5, 4),
                mode='errorshade',
                extra_lines=draw_baseline
            )
        val_curve_list.append(performances)

    plots.error_plot(
        coef_list,
        val_curve_list,
        label_list=algo_list,
        x_label='KL Loss Coef',
        y_label='Prediction Loss',
        save_dir='seq_vae',
        fig_name=f'compare_coef',
        figsize=(5, 4),
        mode='errorshade',
        extra_lines=draw_baseline
    )

def conv_baselines_analysis():
    cfgs = experiments.conv_baselines()

    for i_mode, mode in enumerate(['minmax', 'zscore', ]):
        kernel_size = [1, 2, 3, 5, 10]

        train_performances = []
        performances = []

        for i, length in enumerate(kernel_size):
            loss = get_performance(cfgs, i_mode * len(kernel_size) + i, key2='val_end_loss')
            performances.append(loss)
            train_loss = get_performance(cfgs, i_mode * len(kernel_size) + i, key2='train_end_loss')
            train_performances.append(train_loss)

        x_axis = kernel_size
        val_baseline_performance = get_baseline_performance(cfgs[0][i_mode * len(kernel_size)], 'val')
        def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
            plt.plot(x_axis, [val_baseline_performance] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

        plots.error_plot(
            x_axis,
            [performances, train_performances],
            label_list=['Val', 'Train'],
            x_label='Kernel Size',
            y_label='Prediction Loss',
            save_dir='conv_baseline',
            fig_name=f'TestLoss_{mode}',
            figsize=(5, 4),
            mode='errorshade',
            extra_lines=draw_baseline
        )