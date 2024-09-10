import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

from configs.config_global import ROOT_DIR, LOG_LEVEL, FIG_DIR
from utils.config_utils import configs_dict_unpack, configs_transpose
from analysis import plots, analyze_performance
from configs import experiments
from configs.configs import NeuralPredictionConfig
from datasets.zebrafish import get_baseline_performance, Zebrafish, Simulation, VisualZebrafish

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

    
    x_axis = np.arange((start + 1) * eval_interval, max_batch + 1, plot_every * eval_interval)
    if len(performance) == 0:
        performance = np.zeros_like(x_axis).reshape(-1, 1)
    else:
        performance = np.stack(performance, axis=1)

    return performance, x_axis

def get_performance(
    cfgs, idx, 
    key1='val_mse', key2='val_mse',
    file_name='test.txt', 
    beg=0,
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
    check_complete = file_name == 'progress.txt' # only check training is complete for progress.txt

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]

        try:
            file_path = osp.join(cfg.save_path, file_name)
            exp_data = pd.read_table(file_path)

            if check_complete and len(exp_data[key1]) < tot_steps:
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

def compare_model_training_curves(
    cfgs, 
    save_dir = 'autoregressive_rnns', 
    mode_list = ['none', ], 
    model_list = ['Transformer', 'S4', 'LSTM', 'RNN', ],
    plot_model_lists = None,
    plot_train_test_curve = False,
    show_test_performance = True
):
    """
    Compare the train/test loss curves of different models, also save the train/val loss curves for each model

    :param cfgs: a dictionary of configurations, each key corresponds to a seed, and each value is a list of configurations
        The list should be of length len(mode_list) * len(model_list)
    :param save_dir: the directory to save the figures
    :param mode_list: a list of modes, this function will compare the models in each mode
    :param model_list: a list of models to compare
    :param plot_model_lists: If None, plot all models in model_list; 
        otherwise should be a dictionary of lists, each entry corresponds to a different plot, the key is the plot name
    :param plot_train_test_curve: whether to plot the train/test curves
    """
    
    performance = []

    for i, mode in enumerate(mode_list):
        val_baseline_performance = get_baseline_performance(cfgs[0][i * len(model_list)], 'val')['avg_copy_mse']
        performance = []
        train_performance = []
        test_mse = []
        test_mae = []

        for k, model in enumerate(model_list):
            test_curve, x_axis = get_curve(cfgs, k + i * len(model_list), key='val_mse')
            performance.append(test_curve)
            train_curve, x_axis = get_curve(cfgs, k + i * len(model_list), key='train_mse')
            train_performance.append(train_curve)

            if show_test_performance:
                test_mse.append(get_performance(cfgs, k + i * len(model_list), key2='val_mse'))
                test_mae.append(get_performance(cfgs, k + i * len(model_list), key2='val_mae'))

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
                    fig_name=f'{mode}_{model}',
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
                fig_name=f'{mode}_train_{key}',
                figsize=(5, 4),
                mode='errorshade',
            )

            if show_test_performance:
                # attach the mean +- std of the test performance to the label string
                labels = []
                for i, label in zip(indices, plot_model_list):
                    mean_mse = np.mean(test_mse[i])
                    std_mse = np.std(test_mse[i])
                    mean_mae = np.mean(test_mae[i])
                    std_mae = np.std(test_mae[i])
                    labels.append(f'{label} (MSE: {mean_mse:.3f} ± {std_mse:.3f}, MAE: {mean_mae:.3f} ± {std_mae:.3f})')
                plot_model_list = labels

            plots.error_plot(
                x_axis,
                plot_performance,
                x_label='Training Step',
                y_label='Validation Loss',
                label_list=plot_model_list,
                save_dir=save_dir,
                fig_name=f'{mode}_val_{key}',
                figsize=(8 + show_test_performance * 4, 4),
                mode='errorshade',
                extra_lines=draw_baseline,
                legend_bbox_to_anchor=(1.05, 0), # move the legend to the right
                legend_loc='lower left'
            )

def compare_param_analysis(
    cfgs, param_list, 
    model_list, 
    param_name, 
    mode_list=[''],
    save_dir='compare_param', 
    plot_model_list=None,
    key='val_mse',
    show_chance=True,
    logarithm=False
):
    """
    For each model, plot the best test loss vs. the parameter

    :param cfgs: a dictionary of configurations, each key corresponds to a seed, and each value is a list of configurations
        The list should be of length len(mode_list) * len(model_list) * len(param_list)
    :param param_list: a list of parameters to compare, this will be the x-axis of the plot
    :param model_list: the list of models to compare, this will become legend of the plot
    """

    for k, mode in enumerate(mode_list):

        curves = []
        for i, model in enumerate(model_list):
            train_performances = []
            performances = []
            for j, param in enumerate(param_list):    
                idx = (k * len(model_list) + i) * len(param_list) + j
                loss = get_performance(cfgs, idx, key1=key, key2=key)
                if logarithm:
                    loss = np.log(loss)
                performances.append(loss)
                train_loss = get_performance(cfgs, idx)
                train_performances.append(train_loss)

            curves.append(performances)

        if show_chance:
            val_baseline_performance = [get_baseline_performance(
                    cfgs[0][k * len(model_list) * len(param_list) + j], 'test')[f'avg_copy_{key[-3: ]}'] for j in range(len(param_list))]
            if logarithm:
                val_baseline_performance = np.log(val_baseline_performance)
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
            y_label=key if not logarithm else f'log({key})',
            save_dir=save_dir,
            fig_name=f'compare_{param_name}_{mode}_{key}' if not logarithm else f'compare_{param_name}_{mode}_log_{key}',
            figsize=(5, 4),
            mode='errorshade',
            extra_lines=draw_baseline if show_chance else None,
            errormode='sem' if len(cfgs) > 1 else 'none'
        )

def test_analysis():
    cfgs = experiments.test()
    compare_model_training_curves(
        cfgs, 'autoregressive_rnns_test',
        model_list=['S4', 'LSTM', ],
    )

def sim_compare_noise_analysis():
    cfgs = experiments.sim_compare_noise()
    param_list = [0, 0.01, 0.03, 0.05, 0.1, 0.2, ]
    compare_param_analysis(cfgs, param_list, param_name='noise')

def sim_compare_ga_analysis():
    cfgs = experiments.sim_compare_ga()
    param_list = [1.4, 1.6, 1.8, 2.0, 2.2, ]
    compare_param_analysis(cfgs, param_list, param_name='ga')

def sim_partial_obervable_analysis():
    cfgs = experiments.sim_partial_obervable()
    param_list = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01]
    compare_param_analysis(cfgs, param_list, param_name='portion_observed')

def sim_compare_sparsity_analysis():
    cfgs = experiments.sim_compare_sparsity()
    param_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, ]
    log_param_list = np.log(param_list)
    compare_param_analysis(cfgs, log_param_list, param_name='log(sparsity)')

def sim_compare_train_length_analysis():
    cfgs = experiments.sim_compare_train_length()
    n_neurons = [64, 128, 256, 512, 768, 1024, 1536, ]
    train_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, ]
    model_list = ['Transformer', 'RNN', 'S4']
    mean_performance = {key: [] for key in model_list}
    baseline_performances = [
        get_baseline_performance(cfgs[0][i * len(model_list) * len(train_lengths)], 'val')['avg_copy_mse'] 
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

def conv_encoder_analysis():
    cfgs = experiments.conv_encoder_test()
    compare_model_training_curves(
        cfgs, 'conv_encoder_test',
        mode_list=['spontaneous', 'spontaneous_pc',  ],
        model_list=[
            'POYO_cnn_64_4', 'POYO_cnn_64_16', 'POYO_cnn_512_4', 'POYO_cnn_512_16', 
            'Linear_cnn_64_4', 'Linear_cnn_64_16', 'Linear_cnn_512_4', 'Linear_cnn_512_16', ],
    )

def population_token_analysis():
    cfgs = experiments.compare_population_token_dim()
    compare_model_training_curves(
        cfgs, 'population_token_dim',
        mode_list=['4_128', '4_512', '16_128', '16_512'],
        model_list=['Transformer', 'Linear', 'MLP', ],
    )

plot_lists = {
    'AR': ['AR_Transformer', 'AR_S4', 'AR_RNN', 'AR_LSTM'],
    'Latent': ['Latent_PLRNN', 'Latent_LRRNN_4', 'Latent_CTRNN'],
    # 'TOTEM': ['Transformer_TOTEM', 'MLP_TOTEM', 'Linear_TOTEM', 'POYO_TOTEM'],
    'Decoder': ['Transformer', 'MLP', 'Linear', 'POYO'],
    'Selected': ['Linear', 'Transformer', 'POYO', 'AR_S4', 'Latent_CTRNN'],
}

def compare_models_fc_analysis():
    cfgs = experiments.compare_models_fc()
    model_list = experiments.model_list
    compare_model_training_curves(
        cfgs, 'compare_models_fc',
        model_list=model_list[4: ],
        mode_list=['spontaneous_fc', ],
        plot_model_lists=plot_lists,
        plot_train_test_curve=False
    )

def compare_models_pc_analysis():
    cfgs = experiments.compare_models_pc()
    model_list = experiments.model_list
    plot_lists['Selected'] = ['Linear', 'Transformer', 'POYO', 'POYO_TOTEM', 'Latent_CTRNN', 'TCN']
    compare_model_training_curves(
        cfgs, 'compare_models', 
        model_list=model_list, 
        mode_list=['spontaneous_pc', 'visual_pc', ],
        plot_model_lists=plot_lists,
        plot_train_test_curve=False
    )

def compare_models_sim_analysis():
    cfgs = experiments.compare_models_sim()
    model_list = experiments.model_list
    plot_lists['Selected'] = ['Linear', 'Transformer', 'POYO', 'POYO_TOTEM', 'Latent_CTRNN', 'AR_S4']
    compare_model_training_curves(
        cfgs, 'compare_models_sim', 
        model_list=model_list, 
        mode_list=['sim256', 'sim1024', ],
        plot_model_lists=plot_lists,
        plot_train_test_curve=False
    )

def compare_models_single_neuron_analysis():
    cfgs = experiments.compare_models_single_neuron()
    model_list = experiments.single_neuron_model_list
    plot_lists = {'Selected': ['POYO_TOTEM', 'POYO', 'Linear',] }
    compare_model_training_curves(
        cfgs, 'compare_models_single_neuron', 
        model_list=model_list, 
        mode_list=['visual', 'spontaneous', ],
        plot_model_lists=plot_lists,
        plot_train_test_curve=False
    )

def compare_models_individual_region_analysis():
    cfgs = experiments.compare_models_individual_region()
    model_list = experiments.model_list
    compare_model_training_curves(
        cfgs, 'compare_models_individual_region', 
        model_list=model_list, 
        mode_list=['l_LHb', 'l_MHb', 'l_dthal', 'l_gc', 'l_raphe', 'l_vent', 'l_vthal', ],
        plot_model_lists=plot_lists,
        plot_train_test_curve=False
    )

def compare_models_long_time_scale_analysis():
    cfgs = experiments.compare_models_long_time_scale()
    model_list = experiments.model_list
    compare_model_training_curves(
        cfgs, 'compare_models_longer_timescale', 
        model_list=model_list, 
        mode_list=['spontaneous_pc_2', 'spontaneous_pc_5', 'visual_2', 'visual_5', ],
        plot_model_lists=plot_lists,
        plot_train_test_curve=False
    )

def compare_unit_dropout_analysis():
    cfgs = experiments.compare_unit_dropout()
    for key in ['val_mse', 'val_mae']:
        compare_param_analysis(
            cfgs, 
            [0, 0.1, 0.3, 0.5, 0.7, ],
            ['POYO', 'POYO_TOTEM', ],
            param_name='unit_dropout',
            save_dir='poyo_params',
            key=key,
            mode_list=['spontaneous', 'spontaneous_pc', ],
        )

def compare_num_layers_analysis():
    cfgs = experiments.compare_num_layers()
    for key in ['val_mse', 'val_mae']:
        compare_param_analysis(
            cfgs, 
            [1, 2, 4, 8],
            ['POYO', 'POYO_TOTEM', ],
            param_name='num_layers',
            save_dir='poyo_params',
            key=key,
            mode_list=['spontaneous', 'spontaneous_pc', ],
        )

def compare_hidden_size_analysis():
    cfgs = experiments.compare_hidden_size()
    for key in ['val_mse', 'val_mae']:
        compare_param_analysis(
            cfgs, 
            [64, 256, 1024, ],
            ['POYO', 'POYO_TOTEM', ],
            param_name='hidden_size',
            save_dir='poyo_params',
            key=key,
            mode_list=['spontaneous', 'spontaneous_pc', ]
        )

def poyo_compare_num_latents_analysis():
    cfgs = experiments.poyo_compare_num_latents()
    for key in ['val_mse', 'val_mae']:
        compare_param_analysis(
            cfgs, 
            [4, 16, ],
            ['POYO', 'POYO_TOTEM', ],
            param_name='num_latents',
            save_dir='poyo_params',
            key=key,
            mode_list=['spontaneous', 'spontaneous_pc', ]
        )
    
def poyo_compare_mu_std_module_analysis():
    cfgs = experiments.poyo_compare_mu_std_module()
    compare_model_training_curves(
        cfgs, 'poyo_compare_mu_std_module',
        mode_list=['combined_mu_only', 'combined', 'original', 'learned', 'none', ],
        model_list=['norm_proj', 'norm', 'proj', 'none'],
    )

    cfgs = experiments.poyo_compare_mu_std_module_neuron()
    compare_model_training_curves(
        cfgs, 'poyo_compare_mu_std_module_neuron',
        mode_list=['combined_mu_only', 'combined', 'original', 'learned', 'none', ],
        model_list=['norm_proj', 'norm', 'proj', 'none'],
    )

def linear_compare_mu_std_module_analysis():
    cfgs = experiments.linear_test()
    compare_model_training_curves(
        cfgs, 'linear_compare_mu_std_module',
        mode_list=['pc_none', 'pc_combined', 'single_none', 'single_combined'],
        model_list=['norm_proj', 'norm', 'proj', 'none'],
    )

def poyo_compare_tmax_analysis():
    cfgs = experiments.poyo_compare_tmax()
    for key in ['val_mse', 'val_mae']:
        compare_param_analysis(
            cfgs, 
            [4, 20, 100, ],
            ['multi_m', 'multi', 'single', ],
            param_name='tmax',
            save_dir='poyo_params',
            key=key,
            mode_list=['spontaneous_pc', ]
        )

def compare_compression_factor_analysis():
    cfgs = experiments.compare_compression_factor()
    for key in ['val_mse', 'val_mae']:
        compare_param_analysis(
            cfgs, 
            [4, 16],
            ['POYO', 'POYO_TOTEM', ],
            param_name='compression_factor',
            save_dir='poyo_params',
            key=key,
            mode_list=['spontaneous', 'spontaneous_pc', ]
        )

def compare_lr_analysis():
    cfgs = experiments.compare_lr()
    for key in ['val_mse', 'val_mae']:
        compare_param_analysis(
            cfgs, 
            np.log10([5e-5, 3e-4, 1e-3, 5e-3, ]),
            ['POYO_cos', 'POYO', 'POYO_TOTEM_cos', 'POYO_TOTEM', 'Linear_cos', 'Linear', ],
            param_name='log10(lr)',
            save_dir='poyo_params',
            key=key,
            mode_list=['spontaneous', 'spontaneous_pc', ]
        )

def compare_wd_analysis():
    cfgs = experiments.compare_wd()
    for key in ['val_mse', 'val_mae']:
        compare_param_analysis(
            cfgs, 
            np.log10([1e-3, 1e-2, 0.1, 0.5, 1, 5, 10, ]),
            ['POYO', 'POYO_TOTEM', 'Linear', ],
            param_name='log10(wd)',
            save_dir='poyo_params',
            key=key,
            mode_list=['spontaneous', 'spontaneous_pc', ]
        )

def poyo_ablations_analysis():
    cfgs = experiments.poyo_ablations()
    model_list = [
        'POYO (direct channel-wise readout)', 
        'POYO (direct readout)', 
        'POYO (Perceiver IO, channel-wise readout)', 
        'POYO (Perceiver IO)', 
        'Transformer (channel-wise)', 
        'Transformer (population)', 
    ]
    compare_model_training_curves(
        cfgs, 
        model_list=model_list,
        save_dir='poyo_ablations',
        mode_list=['spontaneous_pc', ]
    )

def compare_train_length_analysis():
    cfgs = experiments.compare_train_length_pc()
    for key in ['val_mse', 'val_mae']:
        compare_param_analysis(
            cfgs, 
            np.log2([128, 256, 512, 1024, 2048, ]),
            experiments.selected_model_list,
            param_name='log2(train_data_length)',
            save_dir='train_length',
            key=key,
            mode_list=['spontaneous', ],
            logarithm=True
        )

    cfgs = experiments.compare_train_length_sim()
    for key in ['val_mse', 'val_mae']:
        compare_param_analysis(
            cfgs, 
            np.log2([256, 1024, 4096, 16384, 65536, ]),
            experiments.selected_model_list,
            param_name='log2(train_data_length)',
            save_dir='train_length',
            key=key,
            mode_list=['sim512', ],
            logarithm=True
        )

def plot_traces(config: NeuralPredictionConfig, sub_save_dir='', titles=None, do_dmd=True, dmd_delay=20):
    import matplotlib.pyplot as plt
    from scipy import signal
    from pydmd import DMD, BOPDMD
    from pydmd.plotter import plot_eigs, plot_summary, plot_modes_2D
    from pydmd.preprocessing import hankel_preprocessing

    config.train_split = 1
    config.val_split = 0
    config.patch_length = 500000
    config.show_chance = False
    
    if config.dataset == 'zebrafish':
        dataset = Zebrafish(config, 'train')
    elif config.dataset == 'simulation':
        dataset = Simulation(config, 'train')
    elif config.dataset == 'zebrafish_visual':
        dataset = VisualZebrafish(config, 'train')

    all_dynamics = []

    for title, data in zip(titles, dataset.neural_data):
        length = len(data)
        plt.figure(figsize=(10, 10))

        # a subplot for each pc
        if length < 10:
            print(f'{title} has less than 10 channels')
            continue

        for i in range(10):
            ax = plt.subplot(10, 2, i * 2 + 1)
            ax.plot(data[i])
            smoothed = np.convolve(data[i], np.ones(250) / 250, mode='same')
            # ax.plot(smoothed, color='red')
            ax.set_ylabel(f'Dim {i + 1}')

            # plot the power spectrum
            ax = plt.subplot(10, 2, i * 2 + 2)
            f, Pxx = signal.periodogram(data[i], fs=1)
            ax.plot(f[1: ], Pxx[1: ])
            ax.set_ylabel('Power')

        save_dir = osp.join(FIG_DIR, 'traces_visualization', sub_save_dir)
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(osp.join(save_dir, f'{title}.png'))
        plt.close()

        # plot 2 sample patches of lenth 64 for each channel
        plt.figure(figsize=(10, 10))
        for i in range(10):
            for j in range(2):
                length = len(data[i])
                
                start = np.random.randint(0, length - 64)
                ax = plt.subplot(10, 2, i * 2 + j + 1)
                ax.plot(data[i][start: start + 64])
                ax.set_ylabel(f'Dim {i + 1}')

        plt.tight_layout()
        plt.savefig(osp.join(save_dir, f'{title}_patch.png'))
        plt.close()

        if not do_dmd:
            continue

        # plot 3: mods found by DMD
        # Build an exact DMD model with 12 spatiotemporal modes.
        d = dmd_delay # delay might be required for noisy data
        optdmd = BOPDMD(svd_rank=12)
        dmd = hankel_preprocessing(optdmd, d=d)
        t = np.arange(0, len(data[0]))
        delayt = t[: -d + 1]
        dmd.fit(data, t=delayt)
        
        # Plot a summary of the key spatiotemporal modes.
        plot_summary(dmd, filename=osp.join(save_dir, f'{title}_dmd.png'), index_modes=(0, 2, 4))

        plt.figure(figsize=(10, 12))
        mode_order = np.argsort(-np.abs(dmd.amplitudes))
        lead_eigs = dmd.eigs[mode_order]
        lead_modes = dmd.modes[:, mode_order]
        lead_dynamics = dmd.dynamics[mode_order]
        lead_amplitudes = np.abs(dmd.amplitudes[mode_order])

        all_dynamics.append((lead_dynamics, lead_eigs))

        for i in range(12):
            ax = plt.subplot(6, 2, i + 1)
            ax.plot(lead_dynamics[i])
            ax.set_ylabel(f'Mode {i + 1}')
            ax.title.set_text(f'Eigenvalue: {lead_eigs[i].real:.4f} + {lead_eigs[i].imag:.4f}j')

        plt.tight_layout()
        plt.savefig(osp.join(save_dir, f'{title}_dmd_mode_dynamics.png'))
        plt.close()

    # a single plot for all the dynamics
    if len(all_dynamics) <= 1:
        return
    
    n_modes = 10
    plt.figure(figsize=(n_modes * 2 + 1, len(all_dynamics)))
    for i, (lead_dynamics, lead_eigs) in enumerate(all_dynamics):
        for j in range(n_modes):
            ax = plt.subplot(len(all_dynamics), n_modes, i * n_modes + j + 1)
            ax.plot(lead_dynamics[j], label=f'{j + 1} ({lead_eigs[j].real:.4f} + {lead_eigs[j].imag:.4f}j)')
            if j == 0:
                ax.set_ylabel(titles[i])
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, f'all_dynamics.png'))
    plt.close()


def plot_traces_analysis():
    from utils.data_utils import get_exp_names, get_subject_ids
    from copy import deepcopy

    exp_names = get_exp_names()
    config: NeuralPredictionConfig = experiments.compare_models_fc()[0][0]
    for key in exp_names.keys():
        all_titles = [f'{key}_{i}' for i in range(1, len(exp_names[key]) + 1)]
        config.exp_types = [key]
        plot_traces(deepcopy(config), f'spontaneous_FCs_{key}', all_titles, dmd_delay=20)

    config = experiments.compare_models_pc()[0][0]
    for key in exp_names.keys():
        all_titles = [f'{key}_{i}' for i in range(1, len(exp_names[key]) + 1)]
        config.exp_types = [key]
        plot_traces(deepcopy(config), f'spontaneous_PCs_{key}', all_titles, dmd_delay=20)    

    config = experiments.compare_models_single_neuron()[0][len(experiments.single_neuron_model_list)]
    for key in exp_names.keys():
        all_titles = [f'{key}_{i}' for i in range(1, len(exp_names[key]) + 1)]
        config.exp_types = [key]
        plot_traces(deepcopy(config), f'spontaneous_{key}', all_titles, dmd_delay=2)

    config = experiments.compare_models_pc()[0][len(experiments.model_list)]
    exp_names = get_subject_ids()
    exp_names = [f'visual_{name}' for name in exp_names]
    plot_traces(config, 'visual_PCs', exp_names, do_dmd=False)

    for i, n_neurons in enumerate([512, 1024]):
        config = experiments.compare_models_sim()[0][i * len(experiments.model_list)]
        config.train_data_length = 3000
        config.pc_dim = n_neurons
        exp_names = [f'sim_{n_neurons}']
        plot_traces(config, 'sim', exp_names, dmd_delay=2)

    config = experiments.compare_models_single_neuron()[0][0]
    exp_names = get_subject_ids()
    exp_names = [f'visual_{name}' for name in exp_names]
    plot_traces(config, 'visual', exp_names, do_dmd=False)
    

def detailed_performance_analysis():
    cfgs = experiments.compare_models_pc()
    for seed, cfg_list in cfgs.items():
        for cfg in cfg_list:
            if cfg.model_label in experiments.selected_model_list:
                analyze_performance.analyze_predictivity(cfg)

    cfgs = experiments.compare_models_sim_pc()
    for seed, cfg_list in cfgs.items():
        for cfg in cfg_list:
            if cfg.model_label in experiments.selected_model_list:
                analyze_performance.analyze_predictivity(cfg)