import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

from configs.config_global import ROOT_DIR, LOG_LEVEL, FIG_DIR
from utils.config_utils import configs_dict_unpack, configs_transpose
from analysis import plots
from configs import experiments
from datasets.zebrafish import get_baseline_performance, Zebrafish

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
    model_list = ['Transformer', 'S4', 'LSTM', 'RNN', ],
    plot_model_lists = None,
    plot_train_test_curve = True,
):
    
    performance = []

    for i, mode in enumerate(mode_list):
        val_baseline_performance = get_baseline_performance(cfgs[0][i * len(model_list)], 'val')
        performance = []
        train_performance = []

        for k, model in enumerate(model_list):
            test_curve, x_axis = get_curve(cfgs, k + i * len(model_list), key='val_end_loss')
            performance.append(test_curve)
            train_curve, x_axis = get_curve(cfgs, k + i * len(model_list), key='TrainLoss')
            train_performance.append(train_curve)

            def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
                plt.plot(x_axis, [val_baseline_performance] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

            if plot_train_test_curve:
                # compare train and test curves
                plots.error_plot(
                    x_axis,
                    [train_curve, test_curve],
                    x_label='Training Step',
                    y_label='Train Loss',
                    label_list=['Train', 'Test'],
                    save_dir=save_dir,
                    fig_name=f'all_fish_{mode}_{model}',
                    figsize=(5, 4),
                    mode='errorshade',
                    extra_lines=draw_baseline
                )

        if plot_model_lists is None:
            plot_model_lists = {'All': None}
        elif isinstance(plot_model_lists, list):
            plot_model_lists = {'selected': plot_model_lists}
        
        for key, plot_model_list in plot_model_lists.items():
            if plot_model_list is None:
                indices = list(range(len(model_list)))
            else:
                indices = [model_list.index(model) for model in plot_model_list]
            plot_performance = [performance[i] for i in indices]
            plot_train_performance = [train_performance[i] for i in indices]
            plot_model_list = [model_list[i] for i in indices]

            plots.error_plot(
                x_axis,
                plot_train_performance,
                x_label='Training Step',
                y_label='Train Loss',
                label_list=plot_model_list,
                save_dir=save_dir,
                fig_name=f'all_fish_{mode}_train_{key}',
                figsize=(5, 4),
                mode='errorshade',
            )

            plots.error_plot(
                x_axis,
                plot_performance,
                x_label='Training Step',
                y_label='Validation Loss',
                label_list=plot_model_list,
                save_dir=save_dir,
                fig_name=f'all_fish_{mode}_val_{key}',
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
    model_list = ['Transformer', 'S4', 'LSTM', 'RNN', ]
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

def sim_compare_param_analysis(cfgs, param_list, model_list=['Transformer', 'RNN', ], param_name='noise', save_dir='sim_compare_param', plot_model_list=None):

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

    if plot_model_list is not None:
        indices = [model_list.index(model) for model in plot_model_list]
        curves = [curves[i] for i in indices]
        model_list = plot_model_list
    
    plots.error_plot(
        param_list,
        curves,
        label_list=model_list,
        x_label=param_name,
        y_label='Prediction Loss',
        save_dir=save_dir,
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

def tca_analysis():
    from analysis.tca import tca
    cfgs = experiments.linear_rnn_test()

    for idx, exp_type in enumerate(['control', 'shocked', 'reshocked']):
        for seed in range(2):
            for j, hidden_size in enumerate([8, 64, 512]):
                if hidden_size < 512:
                    tca(cfgs[seed][j * 3 + idx])

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

def linear_baselines_analysis():
    cfgs = experiments.linear_baselines()
    params = [1, 2, 3, 5, 10, 20, 40]
    sim_compare_param_analysis(
        cfgs,
        [1, 2, 3, 5, 10, 20, 40],
        ['per_channel', 'individual'],
        param_name='input length',
        save_dir='linear_baselines'
    )

def vqvae_decode_analysis():
    cfgs = experiments.vqvae_decode()
    params = [4, 8, 16]
    sim_compare_param_analysis(
        cfgs,
        params,
        ['MLP', 'Transformer'],
        param_name='Compression Factor',
        save_dir='vqvae_decode'
    )

plot_lists = {
    'AR': ['AR_Transformer', 'AR_S4', 'AR_RNN', 'AR_LSTM'],
    'Latent': ['Latent_PLRNN', 'Latent_LRRNN_4', 'Latent_CTRNN'],
    'TOTEM': ['TOTEM_Transformer', 'TOTEM_MLP'],
    'Selected': ['TOTEM_MLP', 'Latent_CTRNN', 'AR_S4', 'Linear', 'TCN']
}

def pc_compare_models_analysis():
    cfgs = experiments.pc_compare_models()
    model_list = experiments.model_list
    autoregressive_rnns_analysis(
        cfgs, 'pc_compare_models',
        model_list=model_list, 
        plot_model_lists=plot_lists
    )

def compare_models_average_analysis():
    cfgs = experiments.compare_models_average()
    model_list = experiments.model_list
    autoregressive_rnns_analysis(
        cfgs, 'compare_models_average', 
        model_list=model_list, 
        plot_model_lists=plot_lists
    )

def compare_models_individual_region_analysis():
    cfgs = experiments.compare_models_individual_region()
    model_list = experiments.model_list
    autoregressive_rnns_analysis(
        cfgs, 'compare_models_individual_region', 
        model_list=model_list, 
        mode_list=['l_LHb', 'l_MHb', 'l_dthal', 'l_gc', 'l_raphe', 'l_vent', 'l_vthal', ],
        plot_model_lists=plot_lists,
        plot_train_test_curve=False
    )

def sim_compare_models_analysis():
    cfgs = experiments.sim_compare_models()
    model_list = experiments.model_list
    autoregressive_rnns_analysis(
        cfgs, 'sim_compare_models', 
        model_list=model_list, 
        mode_list=[256, 512, ],
        plot_model_lists=plot_lists
    )

def sim_compare_models_train_length_analysis():
    cfgs = experiments.sim_compare_models_train_length()
    model_list = experiments.model_list
    train_length = []
    param_list = np.log2([512, 2048, 8192, 65536, ])

    sim_compare_param_analysis(
        cfgs, param_list, model_list=model_list, param_name='log(Training Length)', 
        save_dir='sim_compare_models',
        plot_model_list=['TOTEM_MLP', 'Latent_CTRNN', 'AR_S4', 'Linear', 'TCN']
    )

    cfgs = configs_transpose(cfgs, (len(model_list), len(param_list)))
    autoregressive_rnns_analysis(
        cfgs, 'sim_compare_models_train_length', 
        model_list=model_list, 
        mode_list=[512, 2048, 8192, 65536, ],
        plot_model_lists=plot_lists
    )

def plot_pcs_analysis():
    from configs.configs import NeuralPredictionConfig
    import matplotlib.pyplot as plt

    config: NeuralPredictionConfig = experiments.pc_compare_models()[0][0]
    config.train_split = 1
    config.val_split = 0
    config.patch_length = 10000
    dataset = Zebrafish(config, 'train')

    for idx, data in enumerate(dataset.neural_data):
        
        length = len(data)
        plt.figure(figsize=(4, 10))
        # a subplot for each pc
        for i in range(10):
            plt.subplot(10, 1, i + 1)
            plt.plot(data[i])
            smoothed = np.convolve(data[i], np.ones(250) / 250, mode='same')
            plt.plot(smoothed, color='red')
            plt.title(f'PC{i + 1}')

        save_dir = osp.join(FIG_DIR, 'visualization_pc')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(osp.join(save_dir, f'pc_{idx}'))
        plt.close()