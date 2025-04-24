import numpy as np
import os

from analysis.plots import error_plot, grouped_plot, get_model_colors
from configs.configs import NeuralPredictionConfig, DatasetConfig
from datasets.datasets import get_baseline_performance
from analysis.plots import grouped_plot, errorbar_plot
from configs.config_global import FIG_DIR, MICE_BRAIN_AREAS, ZEBRAFISH_BRAIN_AREAS
from datasets.datasets import Zebrafish, Mice

import matplotlib.pyplot as plt

def analyze_predictivity(config: NeuralPredictionConfig, phases=['train', 'val', 'test']):
    """
    Analyze the prediction performance of the model
    Plot 1: contribution of mse and mae for each PC
    Plot 2: (baseline_mse - mse) / baseline_mse, use copy baseline
        Plot 2.5: (baseline_mse - mse) / baseline_mse, use mean baseline
    Plot 3: violin plot of relative mse and mae for each session
    Plot 4: analyze the performance over time

    Return dict includes:
        - mse_contribution, mae_contribution: contribution of mse/mae for each PC, shape [d], only for pc datasets
        - mse_improvement, mae_improvement: improvement over copy baseline for each PC, shape [d], only for pc datasets
        - mse_improvement_chance, mae_improvement_chance: improvement over mean baseline for each PC, shape [d], only for pc datasets
        - session_mse, session_mae: mse/mae for each session, shape [n_session]
        - session_mse_improvement, session_mae_improvement: improvement over copy baseline for each session, shape [n_session]
        - session_baseline: baseline mse/mae for each session, shape [n_session]
        - mean_mse_time, mean_mae_time: mean mse/mae over time, shape [pred_step]
    """

    ret = {}
    fig_dir = os.path.join(config.save_path, 'performance_plots')
    os.makedirs(fig_dir, exist_ok=True)

    for phase in phases:
        try:
            info_dict = np.load(os.path.join(config.save_path, f'{phase}_best_info.npy'), allow_pickle=True).item()
        except:
            print(f'No {phase} info file found at {config.save_path}')
            continue        

        mse = info_dict['mse'] # list of arrays, each for one session, each array has shape [pred_step, d], sum of mse for all trials
        mae = info_dict['mae'] # similar to mse
        pred_num = info_dict['pred_num'] # list of arrays, each has shape [d]
        return_dict = {}

        datum_size = [x.shape[1] for x in mse]
        pred_step = config.pred_length
        data_config: DatasetConfig = config.dataset_config[config.dataset_label[0]]
        baseline_loss_dict = get_baseline_performance(data_config, phase)
    
        # calculate the mean prediction performance for different PCs
        if config.dataset_label[0].endswith('_pc'):
            data_pc_dim = config.dataset_config[config.dataset_label[0]].pc_dim
            sum_pred_num = np.zeros(data_pc_dim)
            sum_mse = np.zeros(data_pc_dim)
            sum_mae = np.zeros(data_pc_dim)

            # weighted average of mse and mae for each PC across sessions
            for idx in range(len(datum_size)):
                sum_pred_num += pred_num[idx]
                sum_mse += mse[idx].mean(axis=0)
                sum_mae += mae[idx].mean(axis=0)

            mean_mse = sum_mse / sum_pred_num # D
            mean_mae = sum_mae / sum_pred_num
            
            baseline_mse = baseline_loss_dict['mean_copy_mse']
            baseline_mae = baseline_loss_dict['mean_copy_mae']

            # plot 1: contribution of mse and mae for each PC
            plt.figure(figsize=(4, 3))
            plt.plot(mean_mse / np.sum(mean_mse), label='MSE')
            plt.plot(mean_mae / np.sum(mean_mae), label='MAE')
            plt.legend()
            plt.xlabel('PC')
            plt.ylabel('Contribution of Loss')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'{phase}_PC_loss_contribution.pdf'))
            plt.close()

            return_dict['mse_contribution'] = mean_mse / np.sum(mean_mse)
            return_dict['mae_contribution'] = mean_mae / np.sum(mean_mae)

            # plot 2: (baseline_mse - mse) / baseline_mse, use copy baseline
            plt.figure(figsize=(4, 3))
            plt.plot((baseline_mse - mean_mse) / baseline_mse, label='MSE')
            plt.plot((baseline_mae - mean_mae) / baseline_mae, label='MAE')
            plt.hlines(0, 0, data_pc_dim, linestyles='dashed', color='gray')
            plt.legend()
            plt.xlabel('PC')
            plt.ylabel('Improvement over copy baseline')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'{phase}_PC_loss_improvement.pdf'))
            plt.close()

            return_dict['mse_improvement'] = (baseline_mse - mean_mse) / baseline_mse
            return_dict['mae_improvement'] = (baseline_mae - mean_mae) / baseline_mae

            # plot 2.5 use mean baseline
            baseline_mse = baseline_loss_dict['mean_chance_mse']
            baseline_mae = baseline_loss_dict['mean_chance_mae']

            plt.figure(figsize=(4, 3))
            plt.plot((baseline_mse - mean_mse) / baseline_mse, label='MSE')
            plt.plot((baseline_mae - mean_mae) / baseline_mae, label='MAE')
            plt.hlines(0, 0, data_pc_dim, linestyles='dashed', color='gray')
            plt.legend()
            plt.xlabel('PC')
            plt.ylabel('Improvement over baseline')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'{phase}_PC_loss_improvement_chance.pdf'))
            plt.close()

            return_dict['mse_improvement_chance'] = (baseline_mse - mean_mse) / baseline_mse
            return_dict['mae_improvement_chance'] = (baseline_mae - mean_mae) / baseline_mae
        
        elif config.dataset_label[0] in ['zebrafish', 'mice']:
            
            region_ids = []
            if config.dataset_label[0] == 'zebrafish':
                dataset = Zebrafish(config.dataset_config['zebrafish'])
                region_names = ZEBRAFISH_BRAIN_AREAS
            elif config.dataset_label[0] == 'mice':
                dataset = Mice(config.dataset_config['mice'])
                region_names = MICE_BRAIN_AREAS

            region_ids = np.concatenate(dataset.unit_types)
            if config.dataset_label[0] == 'mice':
                region_ids += 1
            elif config.dataset_label[0] == 'zebrafish':
                region_ids[region_ids > 9] = region_ids[region_ids > 9] - 9
            
            all_mse = np.concatenate(mse, axis=1).mean(axis=0) / np.concatenate(pred_num)
            assert all_mse.shape[0] == region_ids.shape[0]
            all_mae = np.concatenate(mae, axis=1).mean(axis=0) / np.concatenate(pred_num)
            assert all_mae.shape[0] == region_ids.shape[0]

            n_regions = len(region_names)
            region_avg_mse = np.zeros((n_regions))
            region_avg_mae = np.zeros((n_regions))

            for i in range(n_regions):
                region_mask = region_ids == i + 1
                region_avg_mse[i] = all_mse[region_mask].mean()
                region_avg_mae[i] = all_mae[region_mask].mean()
            
            return_dict['region_avg_mse'] = region_avg_mse
            return_dict['region_avg_mae'] = region_avg_mae

        # plot 3: violin plot of relative mse and mae for each session
        baseline_mse = np.array(baseline_loss_dict['session_copy_mse'])
        baseline_mae = np.array(baseline_loss_dict['session_copy_mae'])

        session_mse = np.array([x.mean(axis=0).sum() / n.sum() for x, n in zip(mse, pred_num)])
        session_mae = np.array([x.mean(axis=0).sum() / n.sum() for x, n in zip(mae, pred_num)])

        mse_improvement = (baseline_mse - session_mse) / baseline_mse
        mae_improvement = (baseline_mae - session_mae) / baseline_mae

        return_dict['session_mse'] = session_mse
        return_dict['session_mae'] = session_mae
        return_dict['session_mse_improvement'] = mse_improvement
        return_dict['session_mae_improvement'] = mae_improvement
        return_dict['session_baseline'] = baseline_mse

        grouped_plot(
            [[mse_improvement, mae_improvement]],
            ['MSE', 'MAE'],
            save_dir=fig_dir,
            fig_name=f'{phase}_PC_loss_improvement_per_session',
            legend=False,
            x_label='Metric',
            y_label='Improvement over baseline',
            fig_size=(3, 3),
            ylim=(0, None),
            violin_alpha=0.3,
            colors=['#17BEBB'],
            yticks=[0, 0.5, 1],
        )

        # plot 4: analyze the performance over time
        sum_pred_num = 0
        sum_mse = np.zeros(pred_step)
        sum_mae = np.zeros(pred_step)

        for idx in range(len(datum_size)):
            sum_pred_num += pred_num[idx].sum()
            sum_mse += mse[idx].sum(axis=1)
            sum_mae += mae[idx].sum(axis=1)

        mean_mse = sum_mse / sum_pred_num
        mean_mae = sum_mae / sum_pred_num

        baseline_mse = baseline_loss_dict['avg_chance_mse']
        baseline_mae = baseline_loss_dict['avg_chance_mae']

        plt.figure(figsize=(8, 3))
        ax = plt.subplot(1, 2, 1)
        x_axis = np.arange(1, pred_step + 1)
        ax.plot(x_axis, mean_mse, label='MSE')
        ax.hlines(baseline_mse, x_axis[0], x_axis[-1], linestyle='dashed', color='gray', label='Chance MSE')
        ax.legend()
        ax.set_xlabel('Prediction Step')
        ax.set_ylabel('Mean Squared Error')

        ax = plt.subplot(1, 2, 2)
        ax.plot(mean_mae, label='MAE')
        ax.hlines(baseline_mae, x_axis[0], x_axis[-1], linestyle='dashed', color='gray', label='Chance MAE')
        ax.legend()
        ax.set_xlabel('Prediction Step')
        ax.set_ylabel('Mean Absolute Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'{phase}_time_performance.pdf'))
        plt.close()

        return_dict['mean_mse_time'] = mean_mse
        return_dict['mean_mae_time'] = mean_mae
        ret[phase] = return_dict

    return ret

def analyze_predictivity_all(cfgs: dict, plot_phases=['train', 'val', 'test']):
    info_list = [] # list of dicts, each dict is the return value of analyze_predictivity for one model

    for i in range(len(cfgs[0])):
        result_list = []

        for seed, cfg_list in cfgs.items():
            cfg: NeuralPredictionConfig = cfg_list[i]
            ret = analyze_predictivity(cfg, phases=plot_phases)
            result_list.append(ret)
        
        # stack the results slong the last axis
        info_dict = {}
        for phase in plot_phases:
            info_dict[phase] = {}
            for key in result_list[0][phase].keys():
                info_dict[phase][key] = np.stack([x[phase][key] for x in result_list], axis=-1)
        info_list.append(info_dict)

    return info_list

def compare_models(
    cfgs: dict, model_list: list, sub_dir='', 
    save_dir='compare_detailed_performance', plot_phases=['train', 'val', 'test'], 
    config_filter='model_label'
):
    """
    Compare models on their performance over time and performance on different PCs (when analyze_pcs is True).
    Return a list of dicts, each dict is the return value of analyze_predictivity for one model which contains detailed performance info, see analyze_predictivity for details.
        Here we also merge the performance info for different seeds, indicated by the last axis of the arrays
    """
    
    save_dir = os.path.join(FIG_DIR, save_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)

    info_list = [] # list of dicts, each dict is the return value of analyze_predictivity for one model
    dataset_label: str = cfgs[0][0].dataset_label[0]

    for model in model_list:
        result_list = []
        for seed, cfg_list in cfgs.items():
            for cfg in cfg_list:
                cfg: NeuralPredictionConfig
                # print(cfg.__getattribute__(config_filter), model, type(cfg.__getattribute__(config_filter)), type(model))
                if cfg.__getattribute__(config_filter) == model:
                    ret = analyze_predictivity(cfg, phases=plot_phases)
                    if dataset_label in ['mice', 'zebrafish']:
                        print(model, ret['test']['region_avg_mse'])

                    valid = True
                    for phase in plot_phases:
                        if phase not in ret:
                            print(f'No {phase} info found for {cfg.save_path}')
                            valid = False
                    
                    if valid:
                        result_list.append(ret)
                    break

        # stack the results slong the last axis
        info_dict = {}
        for phase in plot_phases:
            info_dict[phase] = {}
            for key in result_list[0][phase].keys():
                info_dict[phase][key] = np.stack([x[phase][key] for x in result_list], axis=-1)
        info_list.append(info_dict)

    colors = get_model_colors(model_list)

    for phase in plot_phases:
        if dataset_label.endswith('_pc'):
            # plot 1: (baseline_mse - mse) / baseline_mse for different models
            # average over [2^k, 2^(k + 1)), k = 0, 1, 2, ... to get a smoother curve

            for key in ['mse_improvement', 'mse_improvement_chance']:

                x_axis_list = []
                data_list = []

                for info in info_list:
                    pc_dim = info[phase][key].shape[0]
                    x_axis = []
                    data = []

                    for k in range(int(np.log2(pc_dim)) + 1):
                        x_axis.append(2 ** k)
                        mean = info[phase][key][int(2 ** (k - 1)): 2 ** k].mean(axis=0)
                        data.append(mean)

                    x_axis_list.append(x_axis)
                    data_list.append(data)

                error_plot(
                    x_axis_list,
                    data_list, 
                    label_list=model_list,
                    xscale='log',
                    save_dir=save_dir,
                    fig_name=f'{phase}_{dataset_label}_{key}',
                    x_label='PC',
                    y_label=f'Improvement over {"chance" if key.endswith("chance") else "copy"} baseline',
                    figsize=(4.5, 3.5),
                    mode='errorshade',
                    colors=colors,
                )

        elif dataset_label in ['zebrafish', 'mice']:
            region_names = MICE_BRAIN_AREAS if dataset_label == 'mice' else ZEBRAFISH_BRAIN_AREAS
            
            for key in ['region_avg_mse', 'region_avg_mae']:
                data_list = [info[phase][key] for info in info_list] # n_models * n_regions * n_seeds

                # bar plot
                grouped_plot(
                    data_list,
                    fig_size=(len(region_names) + 2, 3.5),
                    group_labels=region_names,
                    bar_labels=model_list,
                    save_dir=save_dir,
                    fig_name=f'{phase}_{dataset_label}_{key}',
                    x_label='Region',
                    y_label='Mean Squared Error' if key == 'region_avg_mse' else 'Mean Absolute Error',
                    style='violin',
                    colors=colors
                )

        
        # plot 2: compare the performance over time
        x_axis = np.arange(1, cfgs[0][0].pred_length + 1)
        data_list = [info[phase]['mean_mse_time'] for info in info_list]

        error_plot(
            x_axis,
            data_list, 
            label_list=model_list,
            save_dir=save_dir,
            fig_name=f'{phase}_time_performance_mse',
            x_label='Prediction Step',
            y_label='Mean Squared Error',
            figsize=(5.5, 3),
            colors=colors,
            xticks=(4, 8, 12, 16),
            legend_bbox_to_anchor=(1.05, 0.8),
            legend_loc='upper left',
        )

    return info_list