import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

from configs.config_global import FIG_DIR, N_CELEGANS_SESSIONS, N_MICE_SESSIONS, N_ZEBRAFISH_SESSIONS, N_CELEGANS_FLAVELL_SESSIONS
from utils.config_utils import configs_dict_unpack, configs_transpose
from analysis import plots, analyze_performance, analyze_embedding
from configs import experiments
from configs.configs import NeuralPredictionConfig
from datasets.datasets import get_baseline_performance, Zebrafish, Simulation

def get_curve(cfgs, idx, key='TestLoss', max_batch=None, num_points=None, start=0):
    """
    Read the training curve from the configuration
    
    :param cfgs: a dictionary of configurations, each key corresponds to a seed, and each value is a list of configurations
    :param idx: the index of the configuration to read
    :param key: the key to read from the configuration
    """

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
            
            if len(acc) == (tot_steps - start) // plot_every:
                performance.append(acc)
            else:
                raise ValueError
        except:
            print("Incomplete data:", cfg.save_path)

    
    x_axis = np.arange((start + 1) * eval_interval, max_batch + 1, plot_every * eval_interval)
    if len(performance) == 0:
        print("No data for", cfg.save_path, " Returning zeros")
        performance = np.zeros_like(x_axis).reshape(-1, 1)
    else:
        performance = np.stack(performance, axis=1)

    return performance, x_axis

def get_performance(
    cfgs, idx, key='TestLoss',
    file_name='test.txt', 
):
    """
    Read the test performance
    :return: a list of performance values for each seed
    """

    num_seeds = len(cfgs)
    performance = []

    for seed in range(num_seeds):
        cfg: NeuralPredictionConfig = cfgs[seed][idx]
        file_path = osp.join(cfg.save_path, file_name)
        if not osp.exists(file_path):
            print(f"File {file_path} does not exist", flush=True)
            continue
        exp_data = pd.read_table(file_path)
        performance.append(exp_data[key].iloc[-1])

    return performance

def compare_model_training_curves(
    cfgs, 
    save_dir = None, 
    mode_list = ['', ], 
    model_list = ['POYO', 'Linear', ],
    plot_model_lists = None,
    plot_train_test_curve = False,
    show_test_performance = False,
    datasets = None
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
    :param show_test_performance: whether to show the test performance on the legend
    :param datasets: a list of datasets to plot results, if None, use all datasets in the configuration
    """
    
    performance = []

    for i, mode in enumerate(mode_list):

        cfg: NeuralPredictionConfig = cfgs[0][i * len(model_list)]
        if datasets is None:
            all_datasets = cfg.dataset_label
        else:
            all_datasets = datasets

        for dataset in all_datasets:
            assert dataset in cfg.dataset_config, f"Dataset {dataset} not found in the configuration"

            val_baseline_performance = get_baseline_performance(cfg.dataset_config[dataset], 'val')['avg_copy_mse']
            performance = []
            train_performance = []
            test_mse = []
            test_mae = []

            for k, model in enumerate(model_list):
                test_curve, x_axis = get_curve(cfgs, k + i * len(model_list), key=f'{dataset}_val_mse')
                performance.append(test_curve)
                train_curve, x_axis = get_curve(cfgs, k + i * len(model_list), key=f'{dataset}_train_mse')
                train_performance.append(train_curve)

                if show_test_performance:
                    test_mse.append(get_performance(cfgs, k + i * len(model_list), key=f'{dataset}_val_mse'))
                    test_mae.append(get_performance(cfgs, k + i * len(model_list), key=f'{dataset}_val_mae'))

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
                        fig_name=f'{dataset}_{mode}_{model}',
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

                colors = plots.get_model_colors(plot_model_list)
                plots.error_plot(
                    x_axis,
                    plot_train_performance,
                    x_label='Training Step',
                    y_label='Train Loss',
                    label_list=plot_model_list,
                    save_dir=save_dir,
                    fig_name=f'{dataset}_{mode}_train_{key}',
                    figsize=(4, 3),
                    mode='errorshade',
                    colors=colors,
                )

                if show_test_performance:
                    # attach the mean +- std of the test performance to the label string
                    labels = []
                    for i, label in zip(indices, plot_model_list):
                        mean_mse = np.mean(test_mse[i])
                        std_mse = np.std(test_mse[i])
                        mean_mae = np.mean(test_mae[i])
                        std_mae = np.std(test_mae[i])
                        labels.append(f'{label} (MSE: {mean_mse:.3f} ± {std_mse:.3f})')
                    plot_model_list = labels

                plots.error_plot(
                    x_axis,
                    plot_performance,
                    x_label='Training Step',
                    y_label='Validation Loss',
                    label_list=plot_model_list,
                    save_dir=save_dir,
                    fig_name=f'{dataset}_{mode}_val_{key}',
                    figsize=(6 + show_test_performance * 4, 3),
                    mode='errorshade',
                    extra_lines=draw_baseline,
                    legend_bbox_to_anchor=(1.05, 0), # move the legend to the right
                    legend_loc='lower left',
                    colors=colors,
                )

def compare_param_analysis(
    cfgs, param_list, 
    model_list, 
    param_name, 
    mode_list=[''],
    save_dir='compare_param', 
    plot_model_list=None,
    key='val_score',
    show_chance=False,
    logarithm=False,
    key_name=None,
    dataset=None,
    transpose=False
):
    """
    For each model, plot the best test loss vs. the parameter

    :param cfgs: a dictionary of configurations, each key corresponds to a seed, and each value is a list of configurations
        The list should be of length len(mode_list) * len(model_list) * len(param_list)
        If transpose is True, the list should be of length len(mode_list) * len(param_list) * len(model_list)
    :param param_list: a list of parameters to compare, this will be the x-axis of the plot
    :param model_list: the list of models to compare, this will become legend of the plot
    :param param_name: the name of the parameter, this will be the x-axis label of the plot
    :param mode_list: a list of modes, this function will compare the models in each mode
    :param save_dir: the directory to save the figures
    :param plot_model_list: If None, plot all models in model_list; 
        otherwise should be a list of models to plot
    :param key: the key to get the performance from the configuration
    :param show_chance: whether to show the chance performance on the plot
    :param logarithm: whether to plot the logarithm of the performance
    :param key_name: the name of the key, this will be the y-axis label of the plot
    :param dataset: the dataset to plot results, if None, use all datasets in the configuration
    :param transpose: whether the configuration list is transposed
    """

    key_name = key_name if key_name is not None else key

    for k, mode in enumerate(mode_list):

        curves = []
        for i, model in enumerate(model_list):
            performances = []
            base_idx = k * len(model_list) * len(param_list)

            if dataset is None:
                dataset_name = cfgs[0][base_idx].dataset_label[0]
            else:
                dataset_name = dataset

            for j, param in enumerate(param_list):    
                idx = i * len(param_list) + j + base_idx
                if transpose:
                    idx = j * len(model_list) + i + base_idx

                print("Getting performance for", model, param, mode, "at ", cfgs[0][idx].save_path)
                
                loss = get_performance(cfgs, idx, key=f'{dataset_name}_{key}')

                print("Performance:", loss)

                if logarithm:
                    loss = np.log(loss)
                performances.append(loss)

            curves.append(performances)

        if show_chance:
            val_baseline_performance = [get_baseline_performance(
                    cfgs[0][base_idx], 'test')[f'avg_copy_{key[-3: ]}'] for j in range(len(param_list))]
            if logarithm:
                val_baseline_performance = np.log(val_baseline_performance)
            def draw_baseline(plt, x_axis, linewidth, capsize, capthick):
                plt.plot(param_list, val_baseline_performance, color='gray', linestyle='--', linewidth=linewidth)

        if plot_model_list is not None:
            indices = [model_list.index(model) for model in plot_model_list]
            curves = [curves[i] for i in indices]
            model_list = plot_model_list

        colors = plots.get_model_colors(model_list)
        
        plots.error_plot(
            param_list,
            curves,
            label_list=model_list,
            x_label=param_name,
            y_label=key_name if not logarithm else f'log({key_name})',
            save_dir=save_dir,
            fig_name=f'compare_{param_name}_{mode}_{key}' if not logarithm else f'compare_{param_name}_{mode}_log_{key}',
            figsize=(4, 3),
            mode='errorshade',
            extra_lines=draw_baseline if show_chance else None,
            errormode='sem' if len(cfgs) > 1 else 'none',
            colors=colors,
        )

def get_weighted_performance(cfgs, n_models, n_exps, exp_list, transpose=False, key='val_score', weighted=False, dataset=None):
    """
    Get the weighted average performance from multiple experiments (weighted by the size of test set, or just average)
    
    :param cfgs: a dictionary of lists, each key corresponds to a seed, each list has length n_models * n_exps
    :param n_models: the number of models to compare
    :param n_exps: the number of experiments
    :param exp_list: a list of experiments to take the weighted average, each element in the list is a integer in [0, n_exps)

    :return: a list of length n_models: weighted performances
    """
    performances = []

    for i in range(n_models):

        total = [0 for _ in range(len(cfgs))]
        sum_pred_num = 0
        valid_seeds = len(cfgs)

        for j in exp_list:
            idx = i * n_exps + j
            if transpose:
                idx = j * n_models + i

            if dataset is None:
                dataset_name = cfgs[0][idx].dataset_label[0]
            else:
                dataset_name = dataset
            performance = get_performance(cfgs, idx, key=f'{dataset_name}_{key}')

            if weighted:
                pred_num = get_performance(cfgs, idx, key=f'{dataset_name}_pred_num')
            else:
                pred_num = [1 for _ in range(len(performance))]

            valid_seeds = min(valid_seeds, len(performance))
            for k in range(valid_seeds):
                total[k] += performance[k] * pred_num[k]

            if len(pred_num) > 1:
                assert pred_num[-1] == pred_num[0], "Different number of predictions (test data size) for different seeds, is this expected?"
            elif len(pred_num) == 0:
                print("No data for", cfgs[0][idx].save_path)
                continue

            sum_pred_num += pred_num[0]

        total = [x / sum_pred_num for x in total]
        performances.append(total[: valid_seeds])

    return performances

def poyo_compare_compression_factor_analysis():
    cfgs = experiments.poyo_compare_compression_factor()
    compare_param_analysis(
        cfgs, [4, 8, 16, 24, 48],
        model_list=['POYO', ],
        mode_list=['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ],
        param_name='token size',
        transpose=True
    )

def poyo_compare_model_size_analysis():
    cfgs = experiments.poyo_compare_hidden_size()
    compare_param_analysis(
        cfgs, [32, 128, 256, 512, 768, 1024, 1280, 1536, ],
        model_list=['POYO', ],
        mode_list=['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ],
        param_name='hidden size',
    )

    cfgs = experiments.poyo_compare_num_layers()
    compare_param_analysis(
        cfgs, [1, 2, 4, 8, 12],
        model_list=['POYO', ],
        mode_list=['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ],
        param_name='num layers',
    )

    cfgs = experiments.poyo_compare_num_latents()
    compare_param_analysis(
        cfgs, [1, 2, 4, 8, 16, ],
        model_list=['POYO', ],
        mode_list=['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ],
        param_name='num latents',
    )

    cfgs = experiments.poyo_compare_num_heads()
    compare_param_analysis(
        cfgs, [1, 2, 4, 8, 16, ],
        model_list=['POYO', ],
        mode_list=['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ],
        param_name='num heads',
    )

def compare_models_multi_session_analysis():
    cfgs = experiments.compare_models_multi_session()
    compare_model_training_curves(
        cfgs, 'compare_models_multi_session',
        model_list=experiments.multi_session_model_list, 
        mode_list=[''] * 5,
        plot_train_test_curve=False,
    )

    cfgs = experiments.compare_models_zebrafish_single_neuron()
    compare_model_training_curves(
        cfgs, 'compare_models_zebrafish_single_neuron',
        model_list=experiments.single_neuron_model_list,
        mode_list=[''],
        plot_train_test_curve=False,
    )

def get_split_performance(cfgs, n_split_list, dataset_name, model_list=['POYO', 'Linear']):
    sum_n = sum(n_split_list)
    model_list = ['POYO', 'Linear', ]
    performance = [[] for _ in range(len(model_list))]
    pc_performance = [[] for _ in range(len(model_list))]
    
    for i, n_split in enumerate(n_split_list):
        l = sum(n_split_list[: i])
        r = sum(n_split_list[: i + 1])
        for j, model in enumerate(model_list):
            performance[j].append(get_weighted_performance(cfgs, len(model_list), sum_n * 2, range(l, r)))
            pc_performance[j].append(get_weighted_performance(cfgs, len(model_list), sum_n * 2, range(sum_n + l, sum_n + r)))

    x_axis = range(len(n_split_list))
    x_tick_labels = [f'{n_split}' for n_split in n_split_list]
    
    def draw_chance(plt, x_axis, linewidth, capsize, capthick):
        plt.plot(x_axis, [0] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

    plots.error_plot(
        x_axis, performance, x_label='Number of splits', y_label='Prediction Score',
        label_list=model_list, save_dir='compare_training_set_size', fig_name=f'splits_vs_score_{dataset_name}',
        xticks=x_axis, xtick_labels=x_tick_labels, extra_lines=draw_chance, 
        title=f'{dataset_name} ({max(n_split_list)} Sessions)'
    )
    plots.error_plot(
        x_axis, pc_performance, x_label='Number of splits', y_label='Prediction Score',
        label_list=model_list, save_dir='compare_training_set_size', fig_name=f'splits_vs_score_{dataset_name}_pc',
        xticks=x_axis, xtick_labels=x_tick_labels, extra_lines=draw_chance, 
        title=f'{dataset_name} {cfgs[0][-1].pc_dim} PCs ({max(n_split_list)} Sessions)'
    )

def compare_models_multiple_splits_analysis():
    cfgs = experiments.compare_models_multiple_splits_celegans_flavell()
    n_split_list = [1, 2, 3, 5, 8, 12, 20, 40]
    get_split_performance(cfgs, n_split_list, 'celegansflavell')
    
    cfgs = experiments.compare_models_multiple_splits_mice()
    n_split_list = [1, 2, 3, 4, 6, 12]
    get_split_performance(cfgs, n_split_list, 'mice')

    cfgs = experiments.compare_models_multiple_splits_zebrafish()
    n_split_list = [1, 2, 3, 5, 10, 19]
    get_split_performance(cfgs, n_split_list, 'zebrafish')

def compare_train_length_analysis():
    cfgs = experiments.compare_train_length_celegans_flavell()
    train_length_list = [128, 256, 512, 768, 1024, 1536, ]
    compare_param_analysis(
        cfgs, np.log2(train_length_list), ['POYO', 'Linear'], 'log2(training recording length)', 
        mode_list=['celegansflavell', 'celegansflavell_pc', ], save_dir='compare_train_length'
    )

    cfgs = experiments.compare_train_length_mice()
    train_length_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16394]
    compare_param_analysis(
        cfgs, np.log2(train_length_list), ['POYO', 'Linear'], 'log2(training recording length)',
        mode_list=['mice', 'mice_pc', ], save_dir='compare_train_length'
    )

    cfgs = experiments.compare_train_length_zebrafish()
    train_length_list = [128, 256, 512, 1024, 2048, 3072]
    compare_param_analysis(
        cfgs, np.log2(train_length_list), ['POYO', 'Linear'], 'log2(training recording length)',
        mode_list=['zebrafish_pc', ], save_dir='compare_train_length'
    )

def compare_models_multi_species_analysis():
    cfgs = experiments.compare_models_multi_species()
    model_list = ['POYO', 'POYO_logloss', 'Linear', 'Linear_logloss']
    compare_model_training_curves(
        cfgs, 'compare_models_multi_species', 
        model_list=model_list,
    )

def compare_models_analysis():
    # compare single session models & multi session models
    cfgs = experiments.compare_models_multi_species()
    all_species_model_list = ['All_POYO', 'All_Linear']
    performances = {}
    dataset_list = [
        'zebrafish', 'celegans', 'celegansflavell', 'mice', 
        'zebrafish_pc', 'celegans_pc', 'celegansflavell_pc', 'mice_pc',
    ]
    for i_data, dataset in enumerate(dataset_list):
        performance_list = get_weighted_performance(
            cfgs, len(all_species_model_list), 
            1, [0], dataset=dataset
        )
        performances[dataset] = performance_list
    
    cfgs = experiments.compare_models_multi_session()
    multi_session_model_list = ['MS_POYO', 'MS_Linear', 'MS_Latent_PLRNN', 'MS_AR_Transformer', ]
    dataset_list = ['celegansflavell', 'celegansflavell_pc', 'zebrafish_pc', 'celegans', 'celegans_pc', 'mice', 'mice_pc', ]

    for i_data, dataset in enumerate(dataset_list):
        performance_list = get_weighted_performance(
            cfgs, len(multi_session_model_list), 
            len(dataset_list), [i_data], transpose=True,
        )
        performances[dataset] += performance_list

    cfgs = experiments.compare_models_celegans_single_session()
    single_session_model_list = experiments.single_session_model_list
    performances['celegans'] += get_weighted_performance(
        cfgs, len(single_session_model_list), 
        N_CELEGANS_SESSIONS * 2, range(N_CELEGANS_SESSIONS), transpose=True
    )
    performances['celegans_pc'] += get_weighted_performance(
        cfgs, len(single_session_model_list),
        N_CELEGANS_SESSIONS * 2, range(N_CELEGANS_SESSIONS, 2 * N_CELEGANS_SESSIONS), transpose=True
    )

    cfgs = experiments.compare_models_celegans_flavell_single_session()
    performances['celegansflavell'] += get_weighted_performance(
        cfgs, len(single_session_model_list),
        N_CELEGANS_FLAVELL_SESSIONS, range(N_CELEGANS_FLAVELL_SESSIONS), transpose=True
    )

    cfgs = experiments.compare_models_mice_single_session()
    performances['mice'] += get_weighted_performance(
        cfgs, len(single_session_model_list),
        N_MICE_SESSIONS * 2, range(N_MICE_SESSIONS), transpose=True, 
    )
    performances['mice_pc'] += get_weighted_performance(
        cfgs, len(single_session_model_list),
        N_MICE_SESSIONS * 2, range(N_MICE_SESSIONS, 2 * N_MICE_SESSIONS), transpose=True,
    )

    cfgs = experiments.compare_models_zebrafish_pc_single_session()
    performances['zebrafish_pc'] += get_weighted_performance(
        cfgs, len(single_session_model_list),
        N_ZEBRAFISH_SESSIONS, range(N_ZEBRAFISH_SESSIONS), transpose=True, 
    )

    # compare all models
    all_models = all_species_model_list + multi_session_model_list + single_session_model_list
    plot_datasets = ['zebrafish_pc', 'celegans', 'celegansflavell', 'mice', 'mice_pc', ]
    performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(all_models))]
    plots.grouped_plot(
        performance, group_labels=plot_datasets,
        bar_labels=all_models, colors=plots.get_model_colors(all_models),
        x_label='Dataset', y_label='Prediction Score',
        ylim=[0, 0.6], save_dir='compare_models_all', fig_name='compare_models_all',
        fig_size=(15, 5), style='bar',
        legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
    )

def compare_models_sim_analysis():
    cfgs = experiments.compare_models_sim_multi_session()
    model_list = ['MS_POYO', 'MS_Linear', 'MS_Latent_PLRNN', 'MS_AR_Transformer', ]
    performances = {}
    datasets = ['sim128', 'sim512', ]
    for i_data, dataset in enumerate(datasets):
        performance_list = get_weighted_performance(
            cfgs, len(model_list), len(datasets), [i_data], transpose=True,
        )
        performances[dataset] = performance_list

    cfgs = experiments.compare_models_sim()
    single_session_model_list = experiments.single_session_model_list
    performances['sim128'] += get_weighted_performance(
        cfgs, len(single_session_model_list), 2, [0], transpose=True,
    )
    performances['sim512'] += get_weighted_performance(
        cfgs, len(single_session_model_list), 2, [1], transpose=True,
    )

    model_list += single_session_model_list
    plot_datasets = ['sim128', 'sim512', ]
    performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(model_list))]
    plots.grouped_plot(
        performance, group_labels=plot_datasets,
        bar_labels=model_list, colors=plots.get_model_colors(model_list),
        x_label='Dataset', y_label='Prediction Score',
        ylim=[0, 1], save_dir='compare_models_sim', fig_name='compare_models_sim',
        fig_size=(8, 5), style='bar',
        legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
    )

def poyo_compare_embedding_mode_analysis():
    datasets = ['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ]
    model_list = ['base', 'base + latent', 'session embedding', 'session embedding + latent', 'session + region embedding', 'session + region embedding + latent']
    cfgs = experiments.poyo_compare_embedding_mode()
    performances = {}
    for i_data, dataset in enumerate(datasets):
        performance_list = get_weighted_performance(
            cfgs, len(model_list), len(datasets), [i_data], transpose=True,
        )
        performances[dataset] = performance_list

    compare_model_training_curves(
        cfgs, 'poyo_compare_embedding_mode',
        mode_list=['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ],
        model_list=model_list,
    )

    plot_datasets = ['celegans', 'zebrafish', 'mice', ]
    performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(model_list))]
    plots.grouped_plot(
        performance, group_labels=plot_datasets,
        bar_labels=model_list, colors=plots.get_model_colors(model_list),
        x_label='Dataset', y_label='Prediction Score',
        ylim=[0, 0.6], save_dir='poyo_compare_embedding_mode', fig_name='poyo_compare_embedding_mode',
        fig_size=(8, 5), style='bar',
        legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
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
            param_name='log2(recording length)',
            save_dir='train_length',
            key=key,
            mode_list=['spontaneous', ],
            plot_model_list=['POYO', 'Linear', ],
            logarithm=True,
            key_name='Test MSE' if key == 'val_mse' else 'Test MAE'
        )

    cfgs = experiments.compare_train_length_single_neuron()
    for key in ['val_mse', 'val_mae']:
        compare_param_analysis(
            cfgs, 
            np.log2([128, 256, 512, 1024, 2048, 3000, ]),
            experiments.selected_model_list,
            param_name='log2(recording length)',
            save_dir='train_length',
            key=key,
            mode_list=['ceelgans', ],
            plot_model_list=['POYO', 'Linear', ],
            logarithm=True,
            key_name='Test MSE' if key == 'val_mse' else 'Test MAE'
        )

def detailed_performance_analysis():
    cfgs = experiments.compare_pc_dim()
    dim = [1, 4, 32, 128, 512, 2048, ]
    for i_model, model in enumerate(['Linear', 'Latent_PLRNN', 'POYO']):
        sub_cfgs = {key: cfgs[key][i_model * len(dim): (i_model + 1) * len(dim)] for key in cfgs.keys()}
        analyze_performance.compare_models(sub_cfgs, dim, sub_dir=f'pc_dim_{model}', config_filter='pc_dim')

    cfgs = experiments.compare_models_pc()
    model_list = ['POYO', 'Linear']
    analyze_performance.compare_models(cfgs, model_list, sub_dir='spontaneous_pc')

    cfgs = experiments.compare_models_sim_pc()
    analyze_performance.compare_models(cfgs, model_list, sub_dir='sim_pc')

def poyo_embedding_analysis():
    """
    cfgs = experiments.compare_models_pc()
    model_list = ['POYO', ]
    for seed in cfgs.keys():
        for cfg in cfgs[seed]:
            if cfg.model_label in model_list:
                analyze_embedding.visualize_embedding(cfg)

    cfgs = experiments.compare_models_sim_pc()
    for seed in cfgs.keys():
        for cfg in cfgs[seed]:
            if cfg.model_label in model_list:
                analyze_embedding.visualize_embedding(cfg)
    """

    cfgs = experiments.compare_models_single_neuron()
    model_list = ['POYO', ]
    for seed in cfgs.keys():
        for cfg in cfgs[seed]:
            if cfg.model_label in model_list:
                analyze_embedding.visualize_embedding(cfg, methods=['PCA', 'UMAP'])