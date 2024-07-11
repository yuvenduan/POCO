import os
import os.path as osp
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

def autoregressive_rnns_analysis(
    cfgs = experiments.autoregressive_rnns(), 
    save_dir = 'autoregressive_rnns', 
    mode_list = ['none', ], 
    model_list = ['Transformer', 'S4', 'LSTM', 'RNN', ]
):
    
    data_mode_list = ['Shared', ]
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

def autoregressive_rnns_visual_analysis():
    cfgs = experiments.autoregressive_rnns_visual_with_stimuli()
    autoregressive_rnns_analysis(cfgs, 'autoregressive_rnns_visual', ['stimuli_behavior', 'stimuli', 'behavior', 'none'])

def autoregressive_rnns_average_analysis():
    cfgs = experiments.autoregressive_rnns_average()
    autoregressive_rnns_analysis(cfgs, 'autoregressive_rnns_average')

def autoregressive_rnns_individual_region_analysis():
    cfgs = experiments.autoregressive_rnns_individual_region()
    region_list = ['l_LHb', 'l_MHb', 'l_ctel', 'l_dthal', 'l_gc', 'l_raphe', 'l_tel', 'l_vent', 'l_vthal', ]
    model_list = ['Transformer', 'RNN', ]
    autoregressive_rnns_analysis(cfgs, 'autoregressive_rnns_individual_region', region_list, model_list)

def autoregressive_rnns_sim_analysis():
    cfgs = experiments.autoregressive_rnns_sim()
    model_list = ['Transformer', 'RNN', ]
    autoregressive_rnns_analysis(cfgs, 'autoregressive_rnns_sim', model_list=model_list)

def latent_models_analysis():
    cfgs = experiments.latent_models()
    model_list = ['PLRNN', 'CTRNNCell', ]
    autoregressive_rnns_analysis(cfgs, 'latent_models', [1, 3, 5, 10, 25], model_list)

def latent_models_average_analysis():
    cfgs = experiments.latent_models_average()
    model_list = ['PLRNN', 'CTRNNCell', ]
    autoregressive_rnns_analysis(cfgs, 'latent_models_average', [64, 128, 512, ], model_list=model_list)

def latent_models_sim_analysis():
    cfgs = experiments.latent_model_sim()
    model_list = ['PLRNN', 'CTRNNCell', ]
    autoregressive_rnns_analysis(cfgs, 'latent_models_sim', [1, 3, 5, 10, 25], model_list)

def sim_compare_n_neurons_analysis():
    cfgs = experiments.sim_compare_n_neurons()

    n_regions = [1, 3, ]
    n_neurons = [200, 500, 1500, 3000, ]
    models = ['Transformer', 'S4', 'LSTM', 'RNN', ]

    for i, n in enumerate(n_regions):
        for j, m in enumerate(n_neurons):
            val_baseline_performance = get_baseline_performance(cfgs[0][(i * len(n_neurons) + j) * len(models)], 'val')
            def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
                plt.plot(x_axis, [val_baseline_performance] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

            val_curves = []
            for k, model in enumerate(models):
                test_curve, x_axis = get_curve(cfgs, (i * len(n_neurons) + j) * len(models) + k, key='val_end_loss')
                train_curve, x_axis = get_curve(cfgs, (i * len(n_neurons) + j) * len(models) + k, key='train_end_loss')
                val_curves.append(test_curve)

                # compare train and test curves
                plots.error_plot(
                    x_axis,
                    [train_curve, test_curve],
                    x_label='Training Step',
                    y_label='Train Loss',
                    label_list=['Train', 'Test'],
                    save_dir='sim_compare_n_neurons',
                    fig_name=f'{n}_{m}_{model}',
                    figsize=(5, 4),
                    mode='errorshade',
                    extra_lines=draw_baseline
                )

            plots.error_plot(
                x_axis,
                val_curves,
                x_label='Training Step',
                y_label='Validation Loss',
                label_list=models,
                legend_title='Model',
                save_dir='sim_compare_n_neurons',
                fig_name=f'{n}_{m}_val',
                figsize=(5, 4),
                mode='errorshade',
                extra_lines=draw_baseline
            )

def sim_compare_pca_dim_analysis():
    cfgs = experiments.sim_compare_pca_dim()
    model_list = ['Transformer', 'RNN', ]
    pc_dims = [None, 64, 512, ]

    for i, n in enumerate([500, 3000]):
        for k, dim in enumerate(pc_dims):

            try:
                curves = []

                for j, model in enumerate(model_list):

                    idx = (i * len(model_list) + j) * len(pc_dims) + k
                    test_curve, x_axis = get_curve(cfgs, idx, key='val_end_loss')
                    train_curve, x_axis = get_curve(cfgs, idx, key='train_end_loss')
                    curves.append(test_curve)

                val_baseline_performance = get_baseline_performance(cfgs[0][i * len(model_list) * len(pc_dims) + k], 'val')
                def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
                    plt.plot(x_axis, [val_baseline_performance] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)
            except:
                continue

            plots.error_plot(
                x_axis,
                curves,
                label_list=model_list,
                x_label='Training Step',
                y_label='Prediction Loss',
                save_dir='sim_compare_pca_dim',
                fig_name=f'{n}_{dim}',
                ylim=(0, None),
                figsize=(5, 4),
                mode='errorshade',
                extra_lines=draw_baseline
            )

def sim_compare_param_analysis(cfgs, param_list, model_list=['Transformer', 'RNN', ], param_name='noise'):

    curves = []
    for i, model in enumerate(model_list):
        train_performances = []
        performances = []
        for j, param in enumerate(param_list):    
            idx = i * len(param_list) + j
            loss = get_performance(cfgs, idx, key1='val_end_loss', key2='val_end_loss')
            performances.append(loss)
            train_loss = get_performance(cfgs, idx, key1='val_end_loss', key2='train_end_loss')
            train_performances.append(train_loss)

        curves.append(performances)

    val_baseline_performance = [get_baseline_performance(cfgs[0][j], 'val') for j in range(len(param_list))]
    def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
        plt.plot(param_list, val_baseline_performance, color='gray', linestyle='--', linewidth=linewidth)
    
    plots.error_plot(
        param_list,
        curves,
        label_list=model_list,
        x_label=param_name,
        y_label='Prediction Loss',
        save_dir='sim_compare_param',
        fig_name=f'TestLoss_{param_name}',
        figsize=(5, 4),
        mode='errorshade',
        extra_lines=draw_baseline,
        errormode='sem' if len(cfgs) > 1 else 'none'
    )

def sim_compare_noise_analysis():
    cfgs = experiments.sim_compare_noise()
    param_list = [0, 0.01, 0.03, 0.05, 0.1, 0.2, ]
    sim_compare_param_analysis(cfgs, param_list, param_name='noise')

def sim_compare_ga_analysis():
    cfgs = experiments.sim_compare_ga()
    param_list = [1.4, 1.6, 1.8, 2.0, 2.2, ]
    sim_compare_param_analysis(cfgs, param_list, param_name='ga')

def sim_partial_obervable_analysis():
    cfgs = experiments.sim_partial_obervable()
    param_list = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01]
    sim_compare_param_analysis(cfgs, param_list, param_name='portion_observed')

def sim_compare_sparsity_analysis():
    cfgs = experiments.sim_compare_sparsity()
    param_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, ]
    log_param_list = np.log(param_list)
    sim_compare_param_analysis(cfgs, log_param_list, param_name='log(sparsity)')

def sim_compare_train_length_analysis():
    cfgs = experiments.sim_compare_train_length()
    n_neurons = [64, 128, 256, 512, 768, 1024, 1536, ]
    train_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, ]
    model_list = ['Transformer', 'RNN', 'S4']
    mean_performance = {key: [] for key in model_list}
    baseline_performances = [
        get_baseline_performance(cfgs[0][i * len(model_list) * len(train_lengths)], 'val') 
            for i in range(len(n_neurons))]
    
    for i, n in enumerate(n_neurons):
        loss_lists = []
        for j, model in enumerate(model_list):
            loss_list = []
            mean_loss_list = []
            for k, length in enumerate(train_lengths):
                idx = (i * len(model_list) + j) * len(train_lengths) + k
                loss = get_performance(cfgs, idx, key1='val_end_loss', key2='val_end_loss')
                mean_loss_list.append(np.mean(loss))
                loss_list.append(loss)
            
            loss_lists.append(loss_list)
            mean_performance[model].append(mean_loss_list)

        def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
            plt.plot(x_axis, [baseline_performances[i]] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

        plots.error_plot(
            np.log2(train_lengths),
            loss_lists,
            label_list=model_list,
            x_label='log(Training Length)',
            y_label='Prediction Loss',
            save_dir='sim_compare_train_length',
            fig_name=f'{n}',
            ylim=(0, None),
            figsize=(5, 4),
            mode='errorshade',
            extra_lines=draw_baseline
        )

    threshold = 0.5
    curves = []

    for model in model_list:
        min_lengths = []
        for i, n in enumerate(n_neurons):
            # find the first length that is below the threshold
            min_length = 65536
            for j, length in enumerate(train_lengths):
                if mean_performance[model][i][j] < baseline_performances[i] * threshold:
                    min_length = length
                    break
            min_lengths.append(min_length)

        curves.append(min_lengths)

    def draw_fit(plt, x_axis, linewidth, capsize, capthick):
        # draw y = 2x - 6 for x in [7, 10.5]
        x_axis = np.linspace(7, 10.5, 100)
        plt.plot(x_axis, [2 * x - 6.1 for x in x_axis], color='gray', linestyle='--', linewidth=linewidth, label='y = 2x - 6')

    # for each model, plot the minimum length that reaches the threshold vs. the number of neurons
    plots.error_plot(
        np.log2(n_neurons),
        [np.log2(curve) for curve in curves],
        label_list=model_list,
        errormode='none',
        x_label='log(Number of Neurons)',
        y_label='log(Length of Training Data)',
        save_dir='sim_compare_train_length',
        fig_name=f'datalength_vs_neurons',
        figsize=(5, 4),
        mode='errorshade',
        extra_lines=draw_fit
    )

def meta_rnn_sim_analysis():
    cfgs = experiments.meta_rnn_sim()
    n_regions = [1, ]
    n_neurons = [512, ]
    algorithms = ['maml', 'none', 'fixed', ]

    for i, n in enumerate(n_regions):
        for j, m in enumerate(n_neurons):
            val_baseline_performance = get_baseline_performance(cfgs[0][(i * len(n_neurons) + j) * len(algorithms)], 'val')
            def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
                plt.plot(x_axis, [val_baseline_performance] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

            val_curves = []
            x_axis_ = None
            for k, algo in enumerate(algorithms):
                test_curve, x_axis = get_curve(cfgs, (i * len(n_neurons) + j) * len(algorithms) + k, key='val_end_loss')
                train_curve, x_axis = get_curve(cfgs, (i * len(n_neurons) + j) * len(algorithms) + k, key='train_end_loss')

                val_curves.append(test_curve)
                x_axis_ = x_axis

                # compare train and test curves
                plots.error_plot(
                    x_axis,
                    [train_curve, test_curve],
                    x_label='Training Step',
                    y_label='Train Loss',
                    label_list=['Train', 'Test'],
                    save_dir='meta_rnn_sim',
                    fig_name=f'{n}_{m}_{algo}',
                    figsize=(5, 4),
                    mode='errorshade',
                    extra_lines=draw_baseline
                )

            # print(val_curves)
            plots.error_plot(
                x_axis_,
                val_curves,
                x_label='Training Step',
                y_label='Validation Loss',
                label_list=algorithms,
                legend_title='Algorithm',
                save_dir='meta_rnn_sim',
                fig_name=f'{n}_{m}_val',
                figsize=(5, 4),
                mode='errorshade',
                extra_lines=draw_baseline
            )

def meta_rnn_analysis():
    cfgs = experiments.meta_rnn()
    modes = ['shared', 'seperate']
    lr_list = [3, 10, 100, ]
    algorithms = ['maml', 'none', 'fixed', ]

    for i, mode in enumerate(modes):
        for j, lr in enumerate(lr_list):
            val_baseline_performance = get_baseline_performance(cfgs[0][i * len(lr_list) + j], 'val')
            def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
                plt.plot(x_axis, [val_baseline_performance] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

            val_curves = []
            x_axis_ = None
            for k, algo in enumerate(algorithms):
                test_curve, x_axis = get_curve(cfgs, k + (i * len(lr_list) + j) * len(algorithms), key='val_end_loss')
                train_curve, x_axis = get_curve(cfgs, k + (i * len(lr_list) + j) * len(algorithms), key='train_end_loss')

                val_curves.append(test_curve)
                x_axis_ = x_axis

                # compare train and test curves
                plots.error_plot(
                    x_axis,
                    [train_curve, test_curve],
                    x_label='Training Step',
                    y_label='Train Loss',
                    label_list=['Train', 'Test'],
                    save_dir='meta_rnn',
                    fig_name=f'{mode}_{lr}_{algo}',
                    figsize=(5, 4),
                    mode='errorshade',
                    extra_lines=draw_baseline
                )

            # print(val_curves)
            plots.error_plot(
                x_axis_,
                val_curves,
                x_label='Training Step',
                y_label='Validation Loss',
                label_list=algorithms,
                legend_title='Algorithm',
                save_dir='meta_rnn',
                fig_name=f'{mode}_{lr}_val',
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

    for i_mode, mode in enumerate(['none', ]):
        kernel_size = [1, 2, 3, 5, 10]

        train_performances = []
        performances = []

        for i, length in enumerate(kernel_size):
            loss = get_performance(cfgs, i_mode * len(kernel_size) + i, key1='val_end_loss', key2='val_end_loss')
            performances.append(loss)
            train_loss = get_performance(cfgs, i_mode * len(kernel_size) + i, key1='train_end_loss', key2='train_end_loss')
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

def conv_baselines_sim_analysis():
    cfgs = experiments.conv_baselines_sim()

    n_regions = [1, ]
    n_neurons = [512, 1024, 1536]
    kernel_size = [1, 2, 3, 5, 10]

    for i, n in enumerate(n_regions):
        for j, m in enumerate(n_neurons):

            train_performances = []
            performances = []

            for k, length in enumerate(kernel_size):
                loss = get_performance(cfgs, k + (i * len(n_neurons) + j) * len(kernel_size), key1='val_end_loss', key2='val_end_loss')
                performances.append(loss)
                train_loss = get_performance(cfgs, k + (i * len(n_neurons) + j) * len(kernel_size), key1='train_end_loss', key2='train_end_loss')
                train_performances.append(train_loss)

            x_axis = kernel_size
            val_baseline_performance = get_baseline_performance(cfgs[0][i * len(n_neurons) * len(kernel_size) + j * len(kernel_size)], 'val')
            def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
                plt.plot(x_axis, [val_baseline_performance] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

            plots.error_plot(
                x_axis,
                [performances, train_performances],
                label_list=['Val', 'Train'],
                x_label='Kernel Size',
                y_label='Prediction Loss',
                save_dir='conv_baseline_sim',
                fig_name=f'{n}_{m}',
                figsize=(5, 4),
                mode='errorshade',
                extra_lines=draw_baseline
            )

def sim_compare_performance_analysis():
    cfgs = experiments.conv_baselines_sim()
    