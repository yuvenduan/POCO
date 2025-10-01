import logging
import os.path as osp
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import os

from configs.config_global import LOG_LEVEL, NP_SEED, TCH_SEED, DATA_DIR, DEVICE, MODEL_COLORS
from configs.configs import BaseConfig, SupervisedLearningBaseConfig, NeuralPredictionConfig
from datasets.dataloader import DatasetIters
from tasks.taskfunctions import TaskFunction
from models.model_utils import model_init
from utils.logger import Logger
from utils.train_utils import task_init
from train import get_full_input

import matplotlib.pyplot as plt

def data_batch_to_device(data_b, device=DEVICE):
    if type(data_b) is torch.Tensor:
        return data_b.to(device)
    elif type(data_b) is tuple or type(data_b) is list:
        return [data_batch_to_device(data, device) for data in data_b]
    else:
        raise NotImplementedError("input type not recognized")

@torch.no_grad()
def find_samples(config: NeuralPredictionConfig, alternative_configs, save_path):

    np.random.seed(NP_SEED)
    torch.manual_seed(TCH_SEED)
    random.seed(config.seed)

    config.dataset_config[config.dataset_label[0]].shuffle_test = True
    test_data = DatasetIters(config, 'test')
    
    os.makedirs(save_path, exist_ok=True)
    net = model_init(config, test_data.input_sizes, test_data.unit_types)
    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth'), weights_only=True))
    net.eval()
    net.to(DEVICE)

    alternative_nets = []
    for alt_config in alternative_configs:
        alt_net = model_init(alt_config, test_data.input_sizes, test_data.unit_types)
        alt_net.load_state_dict(torch.load(osp.join(alt_config.save_path, 'net_best.pth'), weights_only=True))
        alt_net.eval()
        alt_net.to(DEVICE)
        alternative_nets.append(alt_net)

    for i_tloader, test_iter in enumerate(test_data.data_iters):
        total = 0
        test_loss = 0.0
        fig_idx = 0

        while True:
            try:
                t_data = next(test_iter)
                t_data = get_full_input(t_data, i_tloader, config, test_data.input_sizes)

                input, target, info_list = t_data
                input, target = data_batch_to_device((input, target))
                
                output = net(input)
                alt_outputs = [alt_net(input) for alt_net in alternative_nets]

                i_plotted = 0  # Total number of trials plotted
                plt.close('all')  # Close any existing figures
                fig, axs = plt.subplots(10, 1, figsize=(5, 12))  # Prepare the first figure
                axs = axs.flatten()

                for i_session in range(len(output)):

                    tar = target[i_session][-config.pred_length:]
                    out = output[i_session][-config.pred_length:]
                    if np.prod(tar.shape) == 0:
                        continue

                    alt_out = [alt_output[i_session][-config.pred_length:] for alt_output in alt_outputs]

                    # calculate l1 loss
                    loss = torch.abs(out - tar).mean(dim=0)  # B * D
                    alt_loss = [torch.abs(a_out - tar).mean(dim=0) for a_out in alt_out]

                    # find the trial and neuron where loss - max(alt_loss) is largest or smallest
                    loss_diff = loss - torch.stack(alt_loss, dim=0).mean(dim=0)  # B * D

                    for i_trial in range(loss_diff.shape[0]):
                        loss_diff_trial = loss_diff[i_trial]  # D

                        # find the neuron with the largest loss difference
                        min_idx = loss_diff_trial.abs().argmax().item()

                        # compare the prediction of different models with the target
                        pred = out[:, i_trial, min_idx]
                        inp = input[i_session][:, i_trial, min_idx]
                        alt_preds = [a_out[:, i_trial, min_idx] for a_out in alt_out]

                        inp_x = np.arange(inp.shape[0])
                        pred_x = np.arange(pred.shape[0]) + inp.shape[0]

                        ax = axs[i_plotted % 10]  # Choose subplot
                        ax.clear()

                        ax.plot(pred_x, pred.cpu().numpy(), label=config.model_label, color=MODEL_COLORS[config.model_label])
                        for i_alt, alt_pred in enumerate(alt_preds):
                            ax.plot(pred_x, alt_pred.cpu().numpy(), label=alternative_configs[i_alt].model_label, color=MODEL_COLORS[alternative_configs[i_alt].model_label])

                        ax.plot(pred_x, tar[:, i_trial, min_idx].cpu().numpy(), label='ground truth', color='black', linestyle='--')
                        ax.plot(inp_x, inp.cpu().numpy(), color='black', linestyle='--')

                        y_min = min(
                            inp.min().item(), 
                            tar[:, i_trial, min_idx].min().item(), 
                            *[alt_pred.min().item() for alt_pred in alt_preds],
                            out[:, i_trial, min_idx].min().item()
                        )
                        y_max = max(
                            inp.max().item(), 
                            tar[:, i_trial, min_idx].max().item(), 
                            *[alt_pred.max().item() for alt_pred in alt_preds],
                            out[:, i_trial, min_idx].max().item()
                        )
                        ax.vlines(inp_x[-1], y_min, y_max, color='gray', linestyle='--')

                        ax.set_title(f'Session {i_session}, Neuron {min_idx}')
                        ax.set_xlabel('Time Step')
                        ax.set_ylabel('Activity')
                        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='x-small')

                        i_plotted += 1

                        # Save figure every 10 plots
                        if i_plotted % 10 == 0:
                            plt.tight_layout()
                            plt.savefig(osp.join(save_path, f'comparison_batch_{fig_idx}.pdf'))
                            plt.close()
                            fig_idx += 1

                            # Prepare a new figure for the next batch
                            fig, axs = plt.subplots(10, 1, figsize=(5, 12))
                            axs = axs.flatten()

                        if fig_idx >= 20:
                            return

            except StopIteration:
                break