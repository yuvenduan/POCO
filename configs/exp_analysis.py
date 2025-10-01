import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

from configs.config_global import FIG_DIR, N_CELEGANS_SESSIONS, N_MICE_SESSIONS, N_ZEBRAFISH_SESSIONS, N_CELEGANS_FLAVELL_SESSIONS, N_ZEBRAFISH_AHNRENS_SESSIONS, DATA_DIR
from utils.config_utils import configs_dict_unpack, configs_transpose
from analysis import plots, analyze_performance, analyze_embedding, tables
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
            
            if len(acc) >= (tot_steps - start) // plot_every:
                performance.append(acc[: (tot_steps - start) // plot_every])
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
        try:
            exp_data = pd.read_table(file_path)
            performance.append(exp_data[key].iloc[-1])
        except:
            print("Incomplete data:", cfg.save_path)

    return performance

def compare_model_training_curves(
    cfgs, 
    save_dir = None, 
    mode_list = ['', ], 
    model_list = ['POYO', 'Linear', ],
    plot_model_lists = None,
    plot_train_test_curve = False,
    show_test_performance = False,
    datasets = None,
    max_batch = None,
    colors = None,
    draw_baseline = True
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
    :param max_batch: the maximum batch to plot
    :param draw_baseline: whether to draw the baseline performance on the plot
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

            if draw_baseline:
                val_baseline_performance = get_baseline_performance(cfg.dataset_config[dataset], 'val')['avg_copy_mse']
            performance = []
            train_performance = []
            test_mse = []
            test_mae = []

            for k, model in enumerate(model_list):
                test_curve, x_axis = get_curve(cfgs, k + i * len(model_list), key=f'{dataset}_val_mse', max_batch=max_batch)
                performance.append(test_curve)
                train_curve, x_axis = get_curve(cfgs, k + i * len(model_list), key=f'{dataset}_train_mse', max_batch=max_batch)
                train_performance.append(train_curve)

                if show_test_performance:
                    test_mse.append(get_performance(cfgs, k + i * len(model_list), key=f'{dataset}_val_mse'))
                    test_mae.append(get_performance(cfgs, k + i * len(model_list), key=f'{dataset}_val_mae'))

                def baseline(plt, x_axis, linewidth, capsize, capthick):
                    if draw_baseline:
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
                        extra_lines=baseline
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

                colors = plots.get_model_colors(plot_model_list) if colors is None else colors
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
                    figsize=(7 + show_test_performance * 4, 3),
                    mode='errorshade',
                    extra_lines=baseline,
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
    transpose=False,
    ylim_list=None,
):
    """
    For each model, plot the best test loss vs. the parameter

    :param cfgs: a dictionary of configurations, each key corresponds to a seed, and each value is a list of configurations. 
        The list should be of len(mode_list) * len(model_list) * len(param_list), flattened from (len(mode_list), len(model_list), len(param_list). 
        If transpose is True, the list should be flattened from (len(mode_list), len(param_list), len(model_list)). 
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

                if logarithm:
                    loss = np.log(loss)
                performances.append(loss)

            curves.append(performances)

        if show_chance:
            if key == 'val_score':
                val_baseline_performance = [0] * len(param_list)
            else:
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
            figsize=(5, 3),
            extra_lines=draw_baseline if show_chance else None,
            errormode='sem',
            colors=colors,
            title=f'{mode}' if mode != '' else None,
            ylim=ylim_list[k] if ylim_list is not None else None,
            legend_loc='upper left',
            legend_bbox_to_anchor=(1, 0.8), # move the legend to the right
        )

def get_weighted_performance(cfgs, n_models, n_exps, exp_list, transpose=False, key='val_score', weighted=None, dataset=None, verbose=False):
    """
    Get the weighted average performance from multiple experiments (weighted by the size of test set, or just average)
    
    :param cfgs: a dictionary of lists, each key corresponds to a seed, each list has length n_models * n_exps (flattened from (n_models, n_exps)).
        If transpose is True, the list should be flattened from (n_exps, n_models)
    :param n_models: the number of models to compare
    :param n_exps: the number of experiments
    :param exp_list: a list of experiments to take the weighted average, each element in the list is a integer in [0, n_exps)
    :param transpose: whether the configuration list is transposed
    :param key: the key to get the performance from the configuration
    :param weighted: if None, average the performance; if 'pred_num', weight the performance by the number of predictions; 
        if 'session', weight the performance by the number of sessions in the training set

    :return: a list of length n_models: weighted performances
    """
    performances = []

    for i in range(n_models):

        total = [0 for _ in range(len(cfgs))]
        sum_weights = 0
        valid_seeds = len(cfgs)

        for j in exp_list:
            idx = i * n_exps + j
            if transpose:
                idx = j * n_models + i

            if dataset is None:
                dataset_name = cfgs[0][idx].dataset_label[0]
            else:
                dataset_name = dataset

            if verbose:
                print("Getting performance for model", i, "exp ", j, "at ", cfgs[0][idx].save_path)
            performance = get_performance(cfgs, idx, key=f'{dataset_name}_{key}')

            if weighted == 'pred_num':
                weight = get_performance(cfgs, idx, key=f'{dataset_name}_val_pred_num')
            elif weighted == 'session':
                id_list = cfgs[0][idx].dataset_config[dataset_name].session_ids
                weight = len(id_list) if id_list is not None else 1
                weight = [weight] * len(performance)
            else:
                weight = [1] * len(performance)

            valid_seeds = min(valid_seeds, len(performance))
            for k in range(valid_seeds):
                total[k] += performance[k] * weight[k]

            if len(weight) > 1:
                assert weight[-1] == weight[0], "Different weight for different seeds, is this expected?"
            elif len(weight) == 0:
                print("No data for", cfgs[0][idx].save_path)
                continue

            sum_weights += weight[0]

        if valid_seeds == 0:
            print("No valid seeds for", i, "model", cfgs[0][i * n_exps].model_label, "returning zeros")
            performances.append([0])
            continue

        total = [x / sum_weights for x in total]
        performances.append(total[: valid_seeds])

    return performances

# ['zebrafish_pc', 'zebrafishahrens_pc', 'celegansflavell', 'celegans', 'mice', 'mice_pc', ]
ylim_list = [(0.5, 0.53), (0.42, 0.45), (0, 0.3), (0.3, 0.4), (0.41, 0.43), (0.37, 0.39)]

def poyo_compare_compression_factor_analysis():
    cfgs = experiments.poyo_compare_compression_factor()
    compare_param_analysis(
        cfgs, [1, 4, 16, 48],
        model_list=['POCO', ],
        mode_list=get_dataset_labels(experiments.dataset_list),
        param_name='Token Length',
        key_name='Prediction Score',
        ylim_list=ylim_list
    )

def compare_context_window_length_analysis():
    cfgs = experiments.compare_context_window_length()
    compare_param_analysis(
        cfgs, [1, 3, 12, 24, 48],
        model_list=['POCO', 'NLinear', 'TexFilter'],
        mode_list=get_dataset_labels(['zebrafish_pc', 'mice', ]),
        param_name='Context Length (C)',
        key_name='Prediction Score',
    )

def compare_model_size_analysis():
    cfgs = experiments.compare_hidden_size()
    compare_param_analysis(
        cfgs, [8, 32, 128, 512, 1024, 1536, ],
        model_list=['POCO', ],
        mode_list=get_dataset_labels(experiments.dataset_list),
        param_name='Hidden Size',
        key_name='Prediction Score',
        ylim_list=ylim_list,
    )

    cfgs = experiments.poyo_compare_num_layers()
    compare_param_analysis(
        cfgs, [1, 4, 8,],
        model_list=['POCO', ],
        mode_list=get_dataset_labels(experiments.dataset_list),
        param_name='Number of Layers',
        key_name='Prediction Score',
        ylim_list=ylim_list,
    )

    cfgs = experiments.poyo_compare_num_latents()
    compare_param_analysis(
        cfgs, [1, 2, 4, 8, 16, ],
        model_list=['POCO', ],
        mode_list=get_dataset_labels(experiments.dataset_list),
        param_name='Number of Latents',
        key_name='Prediction Score',
        ylim_list=ylim_list,
    )

def compare_lr_wd_analysis():
    cfgs = experiments.compare_wd()
    wd_list = np.log10([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 10])
    compare_param_analysis(
        cfgs, wd_list, experiments.training_set_size_models, 'log10(Weight Decay)',
        mode_list=get_dataset_labels(experiments.dataset_list),
        save_dir='compare_lr_wd', key_name='Validation Score',
    )
    
    cfgs = experiments.compare_lr()
    lr_list = np.log10([1e-4, 3e-4, 1e-3, 5e-3, ])
    compare_param_analysis(
        cfgs, lr_list, experiments.training_set_size_models, 'log10(Learning Rate)',
        mode_list=get_dataset_labels(experiments.dataset_list),
        save_dir='compare_lr_wd', key_name='Validation Score',
    )

def get_split_performance(cfgs, n_split_list, dataset_name, modes=['', 'PC'], model_list=['POCO', 'TexFilter']):
    sum_n = sum(n_split_list)

    for i_mode, mode in enumerate(modes):
        performance = [[] for _ in range(len(model_list))]
    
        for i, n_split in enumerate(n_split_list):
            l = sum(n_split_list[: i]) + i_mode * sum_n
            r = sum(n_split_list[: i + 1]) + i_mode * sum_n
            data = get_weighted_performance(
                cfgs, len(model_list), sum_n * len(modes), 
                range(l, r), transpose=True, weighted='session'
            )

            for j, model in enumerate(model_list):
                performance[j].append(data[j])

        x_axis = list(range(len(n_split_list) - 1, -1, -1))
        xtick_labels = [f'{n_split}' for n_split in n_split_list]
        
        def draw_chance(plt, x_axis, linewidth, capsize, capthick):
            plt.plot(x_axis, [0] * len(x_axis), color='gray', linestyle='--', linewidth=linewidth)

        colors = plots.get_model_colors(model_list)
        plots.error_plot(
            x_axis, performance, x_label='Number of splits', y_label='Prediction Score',
            label_list=model_list, save_dir='compare_training_set_size', fig_name=f'splits_vs_score_{dataset_name}_{mode}',
            xticks=x_axis, xticks_labels=xtick_labels, colors=colors, # extra_lines=draw_chance, 
            title=f'{dataset_name} {mode} ({max(n_split_list)} Sessions)'
        )

def compare_models_multiple_splits_analysis():
    cfgs = experiments.compare_models_multiple_splits_celegans_flavell()
    n_split_list = [1, 2, 3, 5, 8, 12, 20, 40]
    get_split_performance(cfgs, n_split_list, 'Worm (Flavell)', ['', 'lowpass'])
    
    cfgs = experiments.compare_models_multiple_splits_mice()
    n_split_list = [1, 2, 3, 4, 6, 12]
    get_split_performance(cfgs, n_split_list, 'Mice', ['', 'lowpass'])

    cfgs = experiments.compare_models_multiple_splits_zebrafish()
    n_split_list = [1, 2, 3, 5, 10, 19]
    get_split_performance(cfgs, n_split_list, 'Fish (Deisseroth)', ['', 'PC'])

    cfgs = experiments.compare_models_multiple_splits_zebrafish_ahrens()
    n_split_list = [1, 2, 3, 5, 8, 15]
    get_split_performance(cfgs, n_split_list, 'Fish (Ahrens)', ['', 'PC'])

    cfgs = experiments.compare_models_multiple_splits_zebrafish_ahrens_lowpass()
    get_split_performance(cfgs, n_split_list, 'Fish (Ahrens)', ['lowpass', 'lowpass PC'])

def compare_train_length_analysis():
    cfgs = experiments.compare_train_length_celegans_flavell()
    train_length_list = [256, 512, 768, 1024, 1536, ]
    compare_param_analysis(
        cfgs, np.log2(train_length_list), experiments.training_set_size_models, 'log2(training recording length)', 
        mode_list=['Worm (Flavell)', 'Worm (Flavell), Low-pass Filtered', ], save_dir='compare_training_set_size', key_name='Validation Score',
    )

    cfgs = experiments.compare_train_length_mice()
    train_length_list = [256, 512, 1024, 2048, 4096, 8192, 16394]
    compare_param_analysis(
        cfgs, np.log2(train_length_list), experiments.training_set_size_models, 'log2(training recording length)',
        mode_list=['Mice', 'Mice, Low-pass Filtered', ], save_dir='compare_training_set_size', key_name='Validation Score',
    )

    cfgs = experiments.compare_train_length_zebrafish() 
    train_length_list = [256, 512, 1024, 1536, 2048, 3072]
    compare_param_analysis(
        cfgs, np.log2(train_length_list), experiments.training_set_size_models, 'log2(training recording length)',
        mode_list=['Fish (Deisseroth)', 'Fish PC (Deisseroth)',], save_dir='compare_training_set_size', key_name='Validation Score',
    )

    cfgs = experiments.compare_train_length_zebrafish_ahrens()
    train_length_list = [256, 512, 1024, 1536, 2048, 3072]
    compare_param_analysis(
        cfgs, np.log2(train_length_list), experiments.training_set_size_models, 'log2(training recording length)',
        mode_list=['Fish PC (Ahrens)', 'Fish PC (Ahrens), Low-pass Filtered',], save_dir='compare_training_set_size', key_name='Validation Score',
    )

def compare_dataset_filter_analysis():
    cfgs = experiments.compare_dataset_filter()
    cfgs_large = experiments.compare_large_dataset_filter()

    from scipy.stats import sem

    filter_types = ['lowpass', 'bandpass', 'none']
    n_exp = len(experiments.dataset_list)

    for i, dataset in enumerate(experiments.dataset_list + ['zebrafishahrens']):
        performance = {}
        model_list = experiments.multi_session_model_list
        if dataset == 'zebrafishahrens':
            model_list = experiments.single_neuron_model_list

        for j, filter_type in enumerate(filter_types):

            if dataset == 'zebrafishahrens':
                performance[filter_type] = get_weighted_performance(
                    cfgs_large, len(model_list), 3, [j], transpose=True
                )
            else:
                performance[filter_type] = get_weighted_performance(
                    cfgs, len(model_list), 3 * n_exp,  [i + j * n_exp], transpose=True
                )

        plot_datasets = filter_types
        performance = [[performance[filter_type][i] for filter_type in filter_types] for i in range(len(model_list))]

        """
        poco = performance[0]
        poco_mean = np.mean(poco, axis=1)
        poco_95ci = 1.96 * sem(poco, axis=1)

        print(f"Dataset: {dataset}, POCO Performance: {poco_mean}, 95% CI: {poco_95ci}")

        texfilter = performance[4]
        texfilter_mean = np.mean(texfilter, axis=1)
        texfilter_95ci = 1.96 * sem(texfilter, axis=1)

        print(f"Dataset: {dataset}, TexFilter Performance: {texfilter_mean}, 95% CI: {texfilter_95ci}")
        """

        plots.grouped_plot(
            performance, group_labels=plot_datasets,
            bar_labels=model_list, colors=plots.get_model_colors(model_list),
            x_label='Filter Type', y_label='Prediction Score',
            ylim=[0, 0.7], save_dir='compare_dataset_filter', fig_name=f'compare_dataset_filter_{dataset}',
            fig_size=(8, 4), style='bar', title='Compare Filters for Dataset: ' + get_dataset_labels([dataset])[0],
            legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
        )

        tables.summarize_model_performance_latex(
            performance, model_list, plot_datasets,
            save_dir='compare_dataset_filter', table_name=f'compare_dataset_filter_{dataset}', precision=3
        )

def ablations_analysis():
    cfgs = experiments.ablations()
    model_list = ['POCO', 'POYO only', 'MLP only', 'Replace POYO w/ Vanilla Transformer', 
                  'Unit-Embedding + FiLM', 'MLP only (large)', 'Unit-Embedding + FiLM (large)']
    dataset_list = experiments.dataset_list
    
    performances = {}
    for i, dataset in enumerate(dataset_list):
        performances[dataset] = get_weighted_performance(
            cfgs, len(model_list), len(dataset_list), [i], transpose=True
        )

    plot_datasets = ['zebrafish_pc', 'zebrafishahrens_pc', 'mice', ]
    performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(model_list))]
    colors = plots.get_model_colors(['POCO', 'POYO', 'MLP', 'NetFormer', 'TexFilter', 'All_MLP', 'All_TexFilter'])
    plots.grouped_plot(
        performance, group_labels=get_dataset_labels(plot_datasets),
        bar_labels=model_list, colors=colors,
        x_label='Dataset', y_label='Prediction Score',
        ylim=[0, 0.7], save_dir='compare_models_all', fig_name='ablations',
        fig_size=(10, 4), style='bar', title='Ablation Analysis',
        legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
    )

    tables.summarize_model_performance_latex(
        performance, model_list, get_dataset_labels(plot_datasets),
        save_dir='compare_models_all', table_name='ablations', precision=3
    )

    tables.summarize_model_performance_markdown(
        performance, model_list, get_dataset_labels(plot_datasets),
        save_dir='rebuttal', table_name='ablations', precision=3
    )

def get_dataset_labels(dataset_list):
    name_dict = {
        'zebrafish': 'Fish (Deisseroth)',
        'zebrafish_pc': 'Fish PC (Deisseroth)',
        'zebrafishahrens': 'Fish (Ahrens)',
        'zebrafishahrens_pc': 'Fish PC (Ahrens)',
        'zebrafishviren': 'Fish (Viren)',
        'zebrafishviren_pc': 'Fish PC (Viren)',
        'mice': 'Mice',
        'mice_pc': 'Mice PC',
        'celegans': 'Worms (Zimmer)',
        'celegans_pc': 'Worms PC (Zimmer)',
        'celegansflavell': 'Worms (Flavell)',
        'celegansflavell_pc': 'Worms PC (Flavell)',
    }
    return [name_dict[dataset] for dataset in dataset_list]

def poco_test_analysis():
    cfgs = experiments.poco_test()
    model_list = ['POCOtest', 'POCO', 'NLinear', 'Linear']
    dataset_list = experiments.dataset_list

    performances = {}
    for i, dataset in enumerate(dataset_list):
        performances[dataset] = get_weighted_performance(
            cfgs, len(model_list), len(dataset_list), [i], transpose=True
        )
    plot_datasets = dataset_list
    performance = [[performances[dataset][i] for dataset in dataset_list] for i in range(len(model_list))]
    colors = plots.get_model_colors(['POCO', 'POCOtest', 'NLinear', 'Linear'])
    plots.grouped_plot(
        performance, group_labels=get_dataset_labels(dataset_list),
        bar_labels=model_list, colors=colors,
        x_label='Dataset', y_label='Prediction Score',
        ylim=[0, 0.7], save_dir='compare_models_all', fig_name='poco_test',
        fig_size=(10, 4), style='bar', title='POCO Test Analysis',
        legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
    )

def compare_models_analysis():

    for key in ['val_score', 'val_mae', 'val_mse']:

        weighted_mode = None if key == 'val_score' else 'pred_num'
        kwargs = {'key': key, 'weighted': weighted_mode, 'transpose': True}

        performances = {}
        def set_performance(performance_list, model_list, dataset_name):
            for i, model in enumerate(model_list):
                performances[dataset_name + '_' + model] = performance_list[i]

        def retrieve_performance(model_list, dataset_list):
            return [[performances[dataset + '_' + model] for dataset in dataset_list] for model in model_list]

        multi_dataset_model_list = ['POCO', ]
        zebrafish_model_list = [f'{n}Datasets_{name}' for n in [2] for name in multi_dataset_model_list]
        cfgs = experiments.zebrafish_multi_datasets()
        for dataset in ['zebrafish_pc', 'zebrafishahrens_pc']:
            performance_list = get_weighted_performance(
                cfgs, len(zebrafish_model_list), 1, [0], dataset=dataset, **kwargs
            )
            set_performance(performance_list, zebrafish_model_list, dataset)

        cfgs = experiments.compare_models_multi_species()
        all_species_model_list = ['All_' + name for name in multi_dataset_model_list]
        dataset_list = [
            'zebrafish_pc', 'zebrafishahrens_pc', 'mice_pc', 
            'mice', 'celegans', 'celegansflavell', 
        ]

        for i_data, dataset in enumerate(dataset_list):
            performance_list = get_weighted_performance(
                cfgs, len(all_species_model_list), 1, [0], dataset=dataset, **kwargs
            )
            set_performance(performance_list, all_species_model_list, dataset)
        
        cfgs = experiments.compare_models_multi_session()
        multi_session_model_list = ['MS_' + model_name.replace('Multi', '') for model_name in experiments.multi_session_model_list]
        assert len(multi_session_model_list) == len(experiments.multi_session_model_list)
        dataset_list = experiments.dataset_list

        for i_data, dataset in enumerate(dataset_list):
            performance_list = get_weighted_performance(
                cfgs, len(multi_session_model_list), len(dataset_list), [i_data], **kwargs
            )
            set_performance(performance_list, multi_session_model_list, dataset)

        """
        cfgs = experiments.compare_models_celegans_single_session()
        single_session_model_list = experiments.single_session_model_list
        set_performance(
            get_weighted_performance(
                cfgs, len(single_session_model_list), 
                N_CELEGANS_SESSIONS * 2, range(N_CELEGANS_SESSIONS), **kwargs
            ),
            single_session_model_list, 'celegans'
        )

        set_performance(
            get_weighted_performance(
                cfgs, len(single_session_model_list),
                N_CELEGANS_SESSIONS * 2, range(N_CELEGANS_SESSIONS, 2 * N_CELEGANS_SESSIONS), **kwargs
            ),
            single_session_model_list, 'celegans_pc'
        )

        cfgs = experiments.compare_models_celegans_flavell_single_session()
        set_performance(
            get_weighted_performance(
                cfgs, len(single_session_model_list),
                N_CELEGANS_FLAVELL_SESSIONS, range(N_CELEGANS_FLAVELL_SESSIONS), **kwargs
            ),
            single_session_model_list, 'celegansflavell'
        )

        cfgs = experiments.compare_models_mice_single_session()
        set_performance(
            get_weighted_performance(
                cfgs, len(single_session_model_list),
                N_MICE_SESSIONS * 2, range(N_MICE_SESSIONS), **kwargs
            ),
            single_session_model_list, 'mice'
        )
        set_performance(
            get_weighted_performance(
                cfgs, len(single_session_model_list),
                N_MICE_SESSIONS * 2, range(N_MICE_SESSIONS, 2 * N_MICE_SESSIONS), **kwargs
            ),
            single_session_model_list, 'mice_pc'
        )

        cfgs = experiments.compare_models_zebrafish_pc_single_session()
        set_performance(
            get_weighted_performance(
                cfgs, len(single_session_model_list),
                N_ZEBRAFISH_SESSIONS, range(N_ZEBRAFISH_SESSIONS), **kwargs
            ),
            single_session_model_list, 'zebrafish_pc'
        )
        cfgs = experiments.compare_models_zebrafish_ahrens_single_session()
        set_performance(
            get_weighted_performance(
                cfgs, len(single_session_model_list),
                N_ZEBRAFISH_AHNRENS_SESSIONS, range(N_ZEBRAFISH_AHNRENS_SESSIONS), **kwargs
            ),
            single_session_model_list, 'zebrafishahrens_pc'
        )
        """
        
        plot_datasets = ['zebrafish_pc', 'zebrafishahrens_pc', 'mice', 'celegans', 'celegansflavell',]
        plot_models = multi_session_model_list
        performance = retrieve_performance(plot_models, plot_datasets)

        if key == 'val_score':
            plots.grouped_plot(
                performance, group_labels=get_dataset_labels(plot_datasets),
                bar_labels=plot_models, colors=plots.get_model_colors(plot_models),
                x_label='Dataset', y_label='Prediction Score',
                ylim=[0, 0.7], save_dir='compare_models_all', fig_name='compare_models_all',
                fig_size=(20, 6), style='bar',
                legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
            )

        tables.summarize_model_performance_latex(
            performance, plot_models, get_dataset_labels(plot_datasets), 
            save_dir='compare_models_all', table_name=f'compare_models_all_{key}', 
            precision=3, best='max' if key == 'val_score' else 'min',
        )

        if key == 'val_score':
            # zebrafish only
            plot_datasets = ['zebrafish_pc', 'zebrafishahrens_pc', ]
            plot_models = ['All_POCO', '2Datasets_POCO', 'MS_POCO', ]
            performance = retrieve_performance(plot_models, plot_datasets)
            plots.grouped_plot(
                performance, group_labels=['Fish PC (Deisseroth)', 'Fish PC (Ahrens)', ],
                bar_labels=plot_models, colors=['#272932', '#4d7ea8'] + plots.get_model_colors(plot_models[2: ]),
                x_label='Dataset', y_label='Prediction Score',
                ylim=[0, 0.7], save_dir='compare_models_all', fig_name='compare_poyo_zebrafish',
                fig_size=(5, 4), style='bar',
                legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
            )
            tables.summarize_model_performance_latex(
                performance, plot_models, ['Fish PC (Deisseroth)', 'Fish PC (Ahrens)', ],
                save_dir='compare_models_all', table_name='compare_poyo_zebrafish', 
                precision=3, best='max'
            )

            plot_datasets = ['zebrafish_pc', 'zebrafishahrens_pc', 'mice', 'mice_pc', 'celegans', 'celegansflavell',]
            plot_models = multi_session_model_list
            performance = retrieve_performance(plot_models, plot_datasets)
            plots.grouped_plot(
                performance, group_labels=get_dataset_labels(plot_datasets),
                bar_labels=plot_models, colors=plots.get_model_colors(plot_models),
                x_label='Dataset', y_label='Prediction Score',
                ylim=[0, 0.7], save_dir='compare_models_all', fig_name='compare_models_all_multi_session',
                fig_size=(15, 5), style='bar',
                legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
            )

            plot_datasets = ['zebrafish_pc', 'zebrafishahrens_pc', 'mice', 'celegans', 'celegansflavell',]
            plot_models = ['MS_POCO', 'All_POCO']
            performance = retrieve_performance(plot_models, plot_datasets)
            tables.summarize_model_performance_latex(
                performance, plot_models, get_dataset_labels(plot_datasets),
                save_dir='compare_models_all', table_name='compare_models_all_multi_session',
                precision=3, best='max'
            )

def compare_models_controlled_time_scale_analysis():
    cfgs = experiments.compare_models_controlled_time_scale()
    cfgs2 = experiments.compare_models_multi_session()

    model_list = experiments.multi_session_model_list
    dataset_list = experiments.dataset_list[1: ]

    performances = {}
    performances['zebrafish_pc'] = get_weighted_performance(
        cfgs2, len(model_list), len(dataset_list) + 1, [0], transpose=True
    )
    for dataset in dataset_list:
        performances[dataset] = get_weighted_performance(
            cfgs, len(model_list), len(dataset_list), [dataset_list.index(dataset)], transpose=True
        )

    plot_datasets = ['zebrafish_pc', 'zebrafishahrens_pc', 'mice', ]
    plot_model_list = model_list
    performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(plot_model_list))]

    plots.grouped_plot(
        performance, group_labels=get_dataset_labels(plot_datasets),
        bar_labels=plot_model_list, colors=plots.get_model_colors(plot_model_list),
        x_label='Dataset', y_label='Prediction Score',
        ylim=[0, 0.7], save_dir='rebuttal', fig_name='compare_models_controlled_time_scale',
        fig_size=(10, 4), style='bar',
        legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
    )

    tables.summarize_model_performance_markdown(
        performance, plot_model_list, get_dataset_labels(plot_datasets),
        save_dir='rebuttal', table_name='compare_models_controlled_time_scale',
        precision=3, best='max',
    )

def compare_models_zebrafish_single_neuron_analysis():
    cfgs = experiments.compare_models_zebrafish_single_neuron()
    model_list = experiments.single_neuron_model_list
    dataset_list = experiments.large_dataset_list
    performances = {}

    for key in ['val_score', 'val_mae', 'val_mse']:

        for i_data, dataset in enumerate(dataset_list):
            performance_list = get_weighted_performance(
                cfgs, len(model_list), len(dataset_list), [i_data], transpose=True, key=key
            )
            performances[dataset] = performance_list

        plot_datasets = dataset_list
        plot_model_list = experiments.single_neuron_model_list
        performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(plot_model_list))]
        
        if key == 'val_score':
            plots.grouped_plot(
                performance, group_labels=get_dataset_labels(plot_datasets),
                bar_labels=model_list, colors=plots.get_model_colors(['MS_' + model for model in plot_model_list]),
                x_label='Dataset', y_label='Prediction Score',
                ylim=[0, 0.6], save_dir='compare_models_all', fig_name='compare_models_zebrafish_single_neuron',
                fig_size=(4.5, 4), style='bar',
                legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
            )

        tables.summarize_model_performance_latex(
            performance, ['MS_' + model for model in plot_model_list], get_dataset_labels(plot_datasets),
            save_dir='compare_models_all', table_name=f'compare_models_zebrafish_single_neuron_{key}',
            precision=3, best='max' if key == 'val_score' else 'min',
        )

def compare_models_sim_analysis():
    performances = {}
    def set_performance(performance_list, model_list, dataset_name):
        for i, model in enumerate(model_list):
            performances[dataset_name + '_' + model] = performance_list[i]

    def retrieve_performance(model_list, dataset_list):
        return [[performances[dataset + '_' + model] for dataset in dataset_list] for model in model_list]

    cfgs = experiments.compare_models_sim_multi_session()
    model_list = experiments.selected_models
    model_list = ['MS_' + model_name.replace('Multi', '') for model_name in model_list]
    noise_list = [0, 0.05, 0.5, 1]
    datasets = [f'sim{noise}_{n}' for noise in noise_list for n in [300]]
    num_seeds = 16

    for i_data, dataset in enumerate(datasets):
        performance_list = get_weighted_performance(
            cfgs, len(model_list), len(datasets), [i_data], transpose=True,
        )
        set_performance(performance_list, model_list, dataset)

    cfgs = experiments.compare_models_sim()
    single_session_model_list = [model_name.replace('Multi', '') for model_name in experiments.selected_models]
    for i_data, dataset in enumerate(datasets):
        performance_list = get_weighted_performance(
            cfgs, len(single_session_model_list), len(datasets) * num_seeds, 
            list(range(i_data * num_seeds, i_data * num_seeds + num_seeds)), transpose=True,
        )
        set_performance(performance_list, single_session_model_list, dataset)

    for i_noise, noise in enumerate(noise_list):
        plot_model_list = single_session_model_list
        plot_datasets = [f'sim{noise}_{n}' for n in [300]]
        performance = retrieve_performance(plot_model_list, plot_datasets)

        plot_model_list = model_list + single_session_model_list
        performance = retrieve_performance(plot_model_list, plot_datasets)
        plots.grouped_plot(
            performance, group_labels=plot_datasets,
            bar_labels=plot_model_list, colors=plots.get_model_colors(plot_model_list),
            x_label='Dataset', y_label='Prediction Score',
            ylim=[0, 1], save_dir='compare_models_sim', fig_name=f'compare_models_sim_multi_session_{noise}',
            fig_size=(4, 4), style='bar',
            legend_loc='upper left', legend_bbox_to_anchor=(1, 0.9), error_mode='sem'
        )

    # plot 2: for 8 different seeds, plot MS_POCO - POCO, with x-axis as the noise level
    for basemodel in ['POCO',]:
        n = 300
        datasets = [f'sim{noise}_{n}' for noise in noise_list]
        performance = retrieve_performance([f'MS_{basemodel}', f'{basemodel}'], datasets)
        
        delta_performance = np.array(performance[0]) - np.array(performance[1]) # n_datasets x n_seeds
        plots.grouped_plot(
            [delta_performance], group_labels=noise_list,
            x_label='$\\eta$', y_label='Prediction Score Difference', title='MS_POCO - POCO', 
            bar_labels=None, legend=None, error_mode='sem', 
            save_dir='compare_models_sim', fig_name=f'multi_session_gain_{basemodel}',
            ylim=[0, 0.1], capsize=8
        )

    # plot 3: compare all single session models
    cfgs = experiments.compare_models_sim_all()
    single_session_model_list = experiments.single_session_model_list
    dataset_list = ['n = 150', 'n = 300']
    performances = []
    for i_data, dataset in enumerate(dataset_list):
        performance_list = get_weighted_performance(
            cfgs, len(single_session_model_list), len(dataset_list), [i_data], transpose=True,
        )
        performances.append(performance_list)
    performances = [[performances[i][j] for i in range(len(dataset_list))] for j in range(len(single_session_model_list))]

    plots.grouped_plot(
        performances, group_labels=dataset_list,
        bar_labels=single_session_model_list, colors=plots.get_model_colors(single_session_model_list),
        x_label='Dataset', y_label='Prediction Score',
        ylim=[0, 1], save_dir='compare_models_sim', fig_name='compare_models_sim_single_session',
        fig_size=(6, 4), style='bar', capsize=4,
        legend_loc='upper left', legend_bbox_to_anchor=(1, 0.9), error_mode='sem'
    )

def finetuning_analysis():

    finetuning_cfgs = experiments.finetuning()
    zero_shot_cfgs = experiments.zeroshot()

    for mode, cfgs in [('finetuning', finetuning_cfgs), ('zero_shot', zero_shot_cfgs)]:

        model_list = [
            'Pre-POCO (full finetune)', 'Pre-POCO (embedding only)', 'Pre-POCO (unit embedding + MLP)', 
            'Pre-MLP', 'Pre-TMLP', 'Pre-UMLP (full finetune)', 'Pre-UMLP (embedding only)',
            'Pre-MLP (large)', 'Pre-UMLP (large, full finetune)', 'Pre-UMLP (large, embedding only)',
            'POCO', 'TMLP', 'UMLP', 'NLinear', 'MLP',  
        ]
        performances = {}

        performances['zebrafish_pc'] = get_weighted_performance(
            cfgs, len(model_list), 10, [0, 1, 2, 3], transpose=True,
        )
        performances['zebrafishahrens_pc'] = get_weighted_performance(
            cfgs, len(model_list), 10, [4, 5, 6, 7], transpose=True,
        )
        performances['mice'] = get_weighted_performance(
            cfgs, len(model_list), 10, [8, 9], transpose=True,
        )

        plot_datasets = ['zebrafish_pc', 'zebrafishahrens_pc', 'mice' ]
        plot_model_list = [
            'Pre-POCO (full finetune)', 'Pre-POCO (embedding only)', 'Pre-MLP (large)', 'POCO', 'MLP',
        ]
        idx = [model_list.index(model) for model in plot_model_list]
        performance = [[performances[dataset][i] for dataset in plot_datasets] for i in idx]
        plots.grouped_plot(
            performance, group_labels=get_dataset_labels(plot_datasets),
            bar_labels=plot_model_list, colors=plots.get_model_colors(plot_model_list),
            x_label='Dataset', y_label='Prediction Score',
            ylim=[0, 0.7], save_dir=mode, fig_name=f'{mode}_summary',
            fig_size=(8, 4.5), style='bar',
            legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
        )

        if mode == 'finetuning':
            # also plot the training curves
            compare_model_training_curves(
                cfgs, 'finetuning',
                mode_list=['Deisseroth{}'.format(i) for i in range(4)] + ['ahrens{}'.format(i) for i in range(4)] + ['mice{}'.format(i) for i in range(2)],
                model_list=model_list, max_batch=200, 
                colors=['#34623F', '#B39C4D'] + plots.get_model_colors(['TexFilter', 'POCO', 'MLP']),
                plot_model_lists=plot_model_list,
                draw_baseline=False
            )
        
        performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(model_list))]
        tables.summarize_model_performance_latex(
            performance, model_list, get_dataset_labels(plot_datasets),
            save_dir=mode, table_name=f'{mode}_summary', precision=3, best='max',
        )

        tables.summarize_model_performance_markdown(
            performance, model_list, get_dataset_labels(plot_datasets),
            save_dir='rebuttal', table_name=f'{mode}_summary', precision=3, best='max',
        )

def compare_mlp_size_analysis():
    cfgs = experiments.compare_mlp_hidden_size()
    poco_cfgs = experiments.poco_compare_conditioning_layers()

    model_list = [f'MLP ({n}, {m})' for n in [2, 3, 4] for m in [256, 1024, 2048, ]]
    poco_model_list = [f'POCO ({n}, {m})' for n in [3, 4] for m in [1024]]

    performances = {}

    for i, dataset in enumerate(experiments.dataset_list):
        performances[dataset] = get_weighted_performance(
            cfgs, len(model_list), len(experiments.dataset_list), [i], transpose=True
        ) + get_weighted_performance(
            poco_cfgs, len(poco_model_list), len(experiments.dataset_list), [i], transpose=True
        )

    model_list += poco_model_list
    plot_datasets = ['zebrafish_pc', 'zebrafishahrens_pc', 'mice', 'celegans', 'celegansflavell',][: 3]
    performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(model_list))]

    tables.summarize_model_performance_markdown(
        performance, model_list, get_dataset_labels(plot_datasets),
        save_dir='rebuttal', table_name='compare_mlp_size', precision=3, best='max',
    )

def compare_multi_species_poco_model_size_analysis():
    cfgs = experiments.compare_models_multi_species_model_size()
    datasets = experiments.dataset_list
    model_list = [f'POCO ({n}, {m})' for m in [128, 256, 512] for n in [1, 2, 4]]

    performances = {}
    for i, dataset in enumerate(datasets):
        performances[dataset] = get_weighted_performance(
            cfgs, len(model_list), 1, [0], transpose=True, dataset=dataset
        )

    plot_datasets = [
        'zebrafish_pc', 'zebrafishahrens_pc', 'mice', 'celegans', 'celegansflavell', 
    ]
    performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(model_list))]
    
    plot_model_list = ['POCO (1, 128)', 'POCO (2, 256)', 'POCO (4, 512)', ]
    plot_model_idx = [model_list.index(model) for model in plot_model_list]
    performance = [performance[i] for i in plot_model_idx]

    tables.summarize_model_performance_markdown(
        performance, plot_model_list, get_dataset_labels(plot_datasets),
        save_dir='rebuttal', table_name='compare_multi_species_poco_model_size',
        precision=3, best='max',
    )

def compare_baseline_model_size_analysis():
    tsmixer_cfgs = experiments.tsmixer_compare_ffdim()
    model_list = [f'TSMixer {n}' for n in [64, 128, 256, 512]]
    dataset_list = ['zebrafish_pc', 'zebrafishahrens_pc', 'mice', ]
    n_total_sessions = 19 + 12 + 15

    performances = {}
    performances['zebrafish_pc'] = get_weighted_performance(
        tsmixer_cfgs, len(model_list), n_total_sessions, range(19), transpose=True,
    )
    performances['zebrafishahrens_pc'] = get_weighted_performance(
        tsmixer_cfgs, len(model_list), n_total_sessions, range(19, 19 + 15), transpose=True,
    )
    performances['mice'] = get_weighted_performance(
        tsmixer_cfgs, len(model_list), n_total_sessions, range(19 + 15, n_total_sessions), transpose=True,
    )

    other_cfgs = experiments.compare_baseline_size()
    models = ['TexFilter', 'AR_Transformer', 'Latent_PLRNN', 'NetFormer']
    sizes = [64, 128, 256, 512, 1024]
    other_model_names = [f'{model} {size}' for model in models for size in sizes]

    for i, dataset in enumerate(dataset_list):
        performances[dataset] += get_weighted_performance(
            other_cfgs, len(other_model_names), len(dataset_list), [i], transpose=True
        )

    model_list = model_list + other_model_names
    plot_datasets = dataset_list
    performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(model_list))]

    tables.summarize_model_performance_latex( # need to change the model names to show the actual parameter
        performance, model_list, get_dataset_labels(plot_datasets),
        save_dir='rebuttal', table_name='compare_baseline_model_size',
        precision=3, best='max',
    )

def compare_pretraining_dataset_analysis():
    cfgs = experiments.compare_pretraining_dataset()
    model_list = ['Pre-POCO(Ahrens)', 'Pre-POCO(Deisseroth)', 'Pre-POCO(Both Datasets)', 'POCO', 'NLinear', 'MLP']
    performances = {}

    performances['zebrafish_pc_jain'] = get_weighted_performance(
        cfgs, len(model_list), 9, [0], transpose=True,
    )
    performances['zebrafish_pc_Deisseroth'] = get_weighted_performance(
        cfgs, len(model_list), 9, [1, 2, 3, 4], transpose=True,
    )
    performances['zebrafish_pc_ahrens'] = get_weighted_performance(
        cfgs, len(model_list), 9, [5, 6, 7, 8], transpose=True,
    )

    plot_datasets = ['zebrafish_pc_Deisseroth', 'zebrafish_pc_ahrens', ]
    plot_model_list = ['Pre-POCO(Deisseroth)', 'Pre-POCO(Ahrens)', 'Pre-POCO(Both Datasets)', 'POCO', 'NLinear', 'MLP']
    idx = [0, 1, 2, 3, 4, 5]
    performance = [[performances[dataset][i] for dataset in plot_datasets] for i in idx]
    plots.grouped_plot(
        performance, group_labels=['Ahrens', 'Deisseroth', 'Jain', ],
        bar_labels=plot_model_list, colors=['#34623F', '#B39C4D'] + plots.get_model_colors(['MS_POCO', 'POCO', 'NLinear', 'MLP']),
        x_label='Finetuning Dataset', y_label='Prediction Score',
        ylim=[0, 0.7], save_dir='finetuning', fig_name='pretraining_dataset_summary',
        fig_size=(8, 4.5), style='bar',
        legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem', title='Zebrafish PCs'
    )

    tables.summarize_model_performance_latex(
        performance, model_list, plot_datasets,
        save_dir='finetuning', table_name='pretraining_dataset_summary', precision=3, best='max',
    )

def simple_dataset_label(dataset_list):
    names = []
    for dataset in dataset_list:
        name: str = dataset
        name = name.replace('zebrafish', 'ze')
        name = name.replace('celegans', 'ce')
        name = name.replace('mice', 'mi')
        name = name.replace('ahrens', 'ah')
        name = name.replace('flavell', 'fl')
        names.append(name)
    return names

def poyo_compare_conditioning_analysis():
    _datasets = experiments.dataset_list + experiments.large_dataset_list
    datasets = _datasets + ['lowpass_' + d for d in _datasets]
    cfgs = experiments.poyo_compare_conditioning()
    model_list = [0, 4, 16, 128, 1024]
    performances = {}

    for i_data, dataset in enumerate(datasets):
        performance_list = get_weighted_performance(
            cfgs, len(model_list), len(datasets), [i_data], transpose=True,
        )
        performances[dataset] = performance_list

    for mode in ['', 'lowpass_']:
        plot_datasets = [mode + d for d in _datasets]
        performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(model_list))]
        plots.grouped_plot(
            performance, group_labels=simple_dataset_label(plot_datasets),
            bar_labels=model_list, colors=plots.get_model_colors(model_list),
            x_label='Dataset', y_label='Prediction Score', legend_title='Conditioning Dim',
            ylim=[0, 0.6], save_dir='poyo_compare_conditioning', fig_name=f'poyo_compare_conditioning{mode}',
            fig_size=(10, 5), style='bar',
            legend_loc='upper left', legend_bbox_to_anchor=(1, 0.8), error_mode='sem'
        )

def sample_prediction_analysis():
    from analysis.find_samples import find_samples
    cfg_list = experiments.compare_models_multi_session()[0]
    dataset_list = ['zebrafish_pc', 'zebrafishahrens_pc', 'celegans', 'mice', ]
    target_model = ['POCO']
    alt_models = ['MLP_L', 'TexFilter']

    for dataset_label in dataset_list:
        cfgs = [cfg for cfg in cfg_list if cfg.dataset_label[0] == dataset_label]

        cfg = [cfg for cfg in cfgs if cfg.model_label in target_model][0]
        alt_cfgs = [cfg for cfg in cfgs if cfg.model_label in alt_models]
        
        save_path = os.path.join(FIG_DIR, 'sample_prediction', dataset_label)
        os.makedirs(save_path, exist_ok=True)
        
        find_samples(cfg, alt_cfgs, save_path)

def zebrafish_region_avg_analysis():
    cfgs = experiments.zebrafish_region_avg()
    model_list = ['ARMLP', 'Latent_PLRNN', 'POCO', 'MLP', 'AR_Transformer']

    compare_model_training_curves(
        cfgs, 'zebrafish_region_avg',
        mode_list=[f'c{x}p{y}' for x in [1, 48] for y in [1, 16]],
        model_list=model_list,
        plot_model_lists=model_list[: 4],
        max_batch=5000
    )

def virtual_perturbation_analysis():
    cfgs = experiments.zebrafish_region_avg()
    models = ['ARMLP', 'Latent_PLRNN', 'POCO', 'MLP', 'AR_Transformer']
    from analysis.stim import analyze_interaction_maps
    analyze_interaction_maps(cfgs, modes=[f'c{x}p{y}' for x in [1, 48] for y in [1, 16]], models=models)

def compare_model_size_celegans_lowpass_analysis():
    cfgs = experiments.compare_model_size_celegans_lowpass()
    datasets = [f'{dataset}' for dataset in ['celegansflavell', 'celegans']]
    model_list = [f'POCO ({lr}, {layer}, {size})' for lr in [3e-5, 1e-4, 1e-3] for layer in [2, 3] for size in [16, 32, 128]]

    performances = {}
    for i_data, dataset in enumerate(datasets):
        performance_list = get_weighted_performance(
            cfgs, len(model_list), len(datasets), [i_data], transpose=True,
        )
        performances[dataset] = performance_list

    plot_datasets = datasets
    performance = [[performances[dataset][i] for dataset in plot_datasets] for i in range(len(model_list))]
    tables.summarize_model_performance_markdown(
        performance, model_list, plot_datasets,
        save_dir='rebuttal', table_name='compare_model_size_celegans_lowpass',
        precision=3, best='max',
    )

def detailed_performance_analysis():
    cfgs = experiments.compare_models_multi_session()
    model_list = experiments.multi_session_model_list
    selected_model_list = ['POCO', 'NLinear', 'TexFilter', ]
    selected_model_names = ['MS_' + name for name in selected_model_list]
    datasets = experiments.dataset_list

    session_performances = {}
    for i_data, dataset in enumerate(datasets):

        sub_cfgs = {key: cfgs[key][i_data * len(model_list): (i_data + 1) * len(model_list)] for key in cfgs.keys()}
        performance_dict_list = analyze_performance.compare_models(
            sub_cfgs, selected_model_list, 
            sub_dir=dataset, plot_phases=['test']
        )
        for model_name, performance_dict in zip(selected_model_names, performance_dict_list):
            session_performances[dataset + '_' + model_name] = np.mean(performance_dict['test']['session_mse_improvement'], axis=-1)

    cfgs = experiments.compare_models_zebrafish_single_neuron()
    model_list = experiments.single_neuron_model_list
    dataset_list = experiments.large_dataset_list
    for i_data, dataset in enumerate(dataset_list):
        if dataset != 'zebrafish':
            continue

        sub_cfgs = {key: cfgs[key][i_data * len(model_list): (i_data + 1) * len(model_list)] for key in cfgs.keys()}
        performance_dict_list = analyze_performance.compare_models(
            sub_cfgs, selected_model_list,
            sub_dir=dataset, plot_phases=['test']
        )
        for model_name, performance_dict in zip(selected_model_names, performance_dict_list):
            session_performances[dataset + '_' + model_name] = np.mean(performance_dict['test']['session_mse_improvement'], axis=-1)

    cfgs = experiments.compare_models_zebrafish_pc_single_session()
    model_list = experiments.single_session_model_list
    # print the performance for each session
    for i_model, model in enumerate(model_list):
        performance = get_weighted_performance(
            cfgs, N_ZEBRAFISH_SESSIONS, len(model_list), [i_model]
        )
        session_performances['zebrafish_pc_' + model] = np.mean(performance, axis=1)

    cfgs = experiments.compare_models_zebrafish_ahrens_single_session()
    model_list = experiments.single_session_model_list

    for i_model, model in enumerate(model_list):
        performance = get_weighted_performance(
            cfgs, N_ZEBRAFISH_AHNRENS_SESSIONS, len(model_list), [i_model]
        )
        session_performances['zebrafishahrens_pc_' + model] = np.mean(performance, axis=1)

    cfgs = experiments.compare_models_mice_single_session()
    model_list = experiments.single_session_model_list

    for i_model, model in enumerate(model_list):
        performance = get_weighted_performance(
            cfgs, N_MICE_SESSIONS, len(model_list), [i_model]
        )
        session_performances['mice_' + model] = np.mean(performance, axis=1)

    data_dir = os.path.join(DATA_DIR, 'analysis_results', 'session_performances.npy')
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)
    np.save(data_dir, session_performances)

def compare_per_session_performance_analysis():
    # need to run detailed_performance_analysis first
    data_dir = os.path.join(DATA_DIR, 'analysis_results', 'session_performances.npy')
    session_performances = np.load(data_dir, allow_pickle=True).item()

    for dataset in ['mice', 'zebrafish_pc', 'zebrafishahrens_pc']:
        save_dir = os.path.join('compare_detailed_performance', dataset)
        dataset_name = get_dataset_labels([dataset])[0]
        for model_x, model_y in [['MS_POCO', 'POCO'], ['MS_POCO', 'NLinear']]:
            x = session_performances[dataset + '_' + model_x]
            y = session_performances[dataset + '_' + model_y]

            plots.scatter_plot(
                x, y, colors='blue', sizes=50, x_label=model_x + ' Score', y_label=model_y + ' Score',
                save_dir=save_dir, fig_name=f'per_session_performance_{model_x}_vs_{model_y}',
                title=f'{dataset_name}: {model_x} vs {model_y}', figsize=(4.5, 4.5),
                diag_line=True
            )

def unit_embedding_analysis():
    model_list = ['POCO', ]
    dataset_list = ['zebrafish_pc', 'zebrafishahrens_pc', 'mice_pc', 'zebrafish', 'mice']

    cfgs = experiments.compare_models_multi_session()
    for idx in range(len(cfgs[0])):
        cfg: NeuralPredictionConfig = cfgs[0][idx]
        if cfg.model_label in model_list and cfg.dataset_label[0] in dataset_list:
            analyze_embedding.analyze_embedding_all_seeds(cfgs, idx, methods=['PCA', 'UMAP'])

    cfgs = experiments.compare_models_zebrafish_single_neuron()
    for idx in range(len(cfgs[0])):
        cfg: NeuralPredictionConfig = cfgs[0][idx]
        if cfg.model_label in model_list and cfg.dataset_label[0] in dataset_list:
            analyze_embedding.analyze_embedding_all_seeds(cfgs, idx, methods=['PCA', 'UMAP'])

from analysis import stim

def stim_analysis():
    cfgs = experiments.zebrafish_stim()

    for seed in range(2):
        for cfg in cfgs[seed]:
            stim.virtual_stimulation(cfg)