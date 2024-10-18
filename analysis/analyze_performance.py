import numpy as np
import os

from analysis.plots import error_plot, grouped_plot
from configs.configs import NeuralPredictionConfig
from datasets.zebrafish import get_baseline_performance
import matplotlib.pyplot as plt

def analyze_predictivity(config: NeuralPredictionConfig):

    for phase in ['val']:

        try:
            info_dict = np.load(os.path.join(config.save_path, f'{phase}_best_info.npy'), allow_pickle=True).item()
        except:
            print(f'No {phase} info file found at {config.save_path}')
            continue

        mse = info_dict['mse'] # list of tensors, each tensor has shape [pred_step, d], sum of mse for all trials
        mae = info_dict['mae'] # similar to mse
        pred_num = info_dict['pred_num'] # list of tensors, each tensor has shape [d]

        datum_size = [x.shape[1] for x in mse]
        pred_step = config.pred_length
        baseline_loss_dict = get_baseline_performance(config, phase)
    
        # calculate the mean prediction performance for different PCs
        if config.pc_dim is not None:

            sum_pred_num = np.zeros(config.pc_dim)
            sum_mse = np.zeros(config.pc_dim)
            sum_mae = np.zeros(config.pc_dim)

            # weighted average of mse and mae for each PC across animals
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
            plt.savefig(os.path.join(config.save_path, f'{phase}_PC_loss_contribution.pdf'))
            plt.close()

            # plot 2: (baseline_mse - mse) / baseline_mse
            plt.figure(figsize=(4, 3))
            plt.plot((baseline_mse - mean_mse) / baseline_mse, label='MSE')
            plt.plot((baseline_mae - mean_mae) / baseline_mae, label='MAE')
            plt.hlines(0, 0, config.pc_dim, linestyles='dashed', color='gray')
            plt.legend()
            plt.xlabel('PC')
            plt.ylabel('Improvement over copy baseline')
            plt.tight_layout()
            plt.savefig(os.path.join(config.save_path, f'{phase}_PC_loss_improvement.pdf'))
            plt.close()

            # plot 2.5 use chance mse and mae as baseline
            baseline_mse = baseline_loss_dict['mean_chance_mse']
            baseline_mae = baseline_loss_dict['mean_chance_mae']

            plt.figure(figsize=(4, 3))
            plt.plot((baseline_mse - mean_mse) / baseline_mse, label='MSE')
            plt.plot((baseline_mae - mean_mae) / baseline_mae, label='MAE')
            plt.hlines(0, 0, config.pc_dim, linestyles='dashed', color='gray')
            plt.legend()
            plt.xlabel('PC')
            plt.ylabel('Improvement over baseline')
            plt.tight_layout()
            plt.savefig(os.path.join(config.save_path, f'{phase}_PC_loss_improvement_chance.pdf'))
            plt.close()


        # plot 3: violin plot of relative mse and mae for each animal
        baseline_mse = np.array(baseline_loss_dict['animal_copy_mse'])
        baseline_mae = np.array(baseline_loss_dict['animal_copy_mae'])

        animal_mse = np.array([x.mean(axis=0).sum() / n.sum() for x, n in zip(mse, pred_num)])
        animal_mae = np.array([x.mean(axis=0).sum() / n.sum() for x, n in zip(mae, pred_num)])

        print(baseline_mse[: 10], baseline_mae[: 10], animal_mse[: 10], animal_mae[: 10])

        mse_improvement = (baseline_mse - animal_mse) / baseline_mse
        mae_improvement = (baseline_mae - animal_mae) / baseline_mae

        grouped_plot(
            [[mse_improvement, mae_improvement]],
            ['MSE', 'MAE'],
            save_dir=config.save_path,
            fig_name=f'{phase}_PC_loss_improvement_per_animal',
            legend=False,
            x_label='Metric',
            y_label='Improvement over baseline',
            fig_size=(3, 3),
            ylim=(0, None),
            violin_alpha=0.3,
            colors=['#17BEBB'],
            yticks=[0, 0.5, 1],
        )