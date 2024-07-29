import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import analysis.plots as plots

from configs.config_global import DEVICE
from configs.configs import SupervisedLearningBaseConfig, NeuralPredictionConfig
from utils.logger import Logger
from models import MultiFishSeqVAE, vqvae, Decoder
from typing import List

def data_batch_to_device(data_b, device=DEVICE):
    if type(data_b) is torch.Tensor:
        return data_b.to(device)
    elif type(data_b) is tuple or type(data_b) is list:
        return [data_batch_to_device(data, device) for data in data_b]
    else:
        raise NotImplementedError("input type not recognized")

class TaskFunction:

    def __init__(self, config: SupervisedLearningBaseConfig):
        """
        initailize information from config
        """
        pass

    def roll(self, model: nn.Module, data_batch: tuple, phase: str = 'train'):
        """
        Phase can be 'train', 'val', or 'test'
        In the training mode (train=True), this method should return a scalar representing the training loss
        In the eval or test mode, this method should return a tuple of at least 3 elements: 
        (
            test loss, 
            number of predictions (often equal to batchsize), 
            number of correct prediction (in classification tasks) or sum of error (in regression tasks),
            ... (additional information that might be used in callback functions)
        )
        """
        raise NotImplementedError

    def after_testing_callback(self, batch_info: List[tuple], logger: Logger, save_path: str, is_best: bool, batch_num: int):
        """
        An optional function called after the test stage
        :param batch_info: list of return values of task.roll(mode, data_batch, test=True) for batches of data in the test set
        :param is_best: whether the test result is the best one
        :param batch_num: number of batches used for training
        """
        pass

    def after_training_callback(self, config, model):
        """
        An optional function called after training is done
        :param model: the best model
        """
        pass

class NeuralPrediction(TaskFunction):
    """
    Task function for autoregressive neural prediction
    """

    def __init__(self, config: NeuralPredictionConfig):
        self.criterion = nn.MSELoss(reduction='sum')
        self.data = {'train': [], 'val': [], 'test': []}
        self.curves = {'train': [], 'val': [], 'test': []}

        self.do_analysis = config.do_analysis
        self.loss_mode = config.loss_mode
        self.pred_step = config.pred_length
        self.target_mode = config.target_mode
        self.config = config

        self.recon_loss_list = {'train': [], 'val': [], 'test': []}
        self.vae_loss_list = {'train': [], 'val': [], 'test': []}

        assert self.loss_mode in ['prediction', 'autoregressive', 'reconstruction', 'reconstruction_prediction']
        if self.loss_mode == 'reconstruction':
            assert self.pred_step == 0, 'reconstruction mode does not support prediction'
        else:
            assert self.pred_step > 0, 'prediction step should be larger than 0'

    def roll(self, model: nn.Module, data_batch: tuple, phase: str = 'train'):
        train_flag = phase == 'train'
        
        input, target, info_list = data_batch
        input, target = data_batch_to_device((input, target))
        output = model(input, pred_step=self.pred_step)
    
        task_loss = torch.zeros(1).to(DEVICE)
        pred_num = 0

        for inp, out, tar, info in zip(input, output, target, info_list):
            if target is None or np.prod(tar.shape) == 0:
                continue

            if self.loss_mode[: 14] == 'reconstruction':
                tar = torch.cat([inp[: 1], tar], dim=0)
            elif self.loss_mode == 'prediction':
                out = out[-self.pred_step: ]
                tar = tar[-self.pred_step: ]
            else:
                assert self.loss_mode == 'autoregressive'

            loss_array = F.mse_loss(out, tar, reduction='none') # shape: L, B, D
            self.data[phase].append((
                out.detach().cpu().numpy(),
                tar.detach().cpu().numpy(),
                loss_array.detach().cpu().numpy(),
                inp.detach().cpu().numpy()
            ))

            task_loss += self.criterion(out, tar) # shape: L, B, D
            pred_num += np.prod(tar.shape)

        task_loss = task_loss / max(1, pred_num)

        if isinstance(model, MultiFishSeqVAE):
            self.recon_loss_list[phase].append(task_loss.item())
            task_loss += model.kl_loss * self.config.kl_loss_coef / self.config.batch_size
            self.vae_loss_list[phase].append(model.kl_loss.item() / self.config.batch_size)
        elif isinstance(model, vqvae):
            self.recon_loss_list[phase].append(task_loss.item())
            task_loss += model.vq_loss
            self.vae_loss_list[phase].append(model.vq_loss.item())
        # elif isinstance(model, Decoder):
        #    task_loss += F.mse_loss(model.mu,)

        if train_flag:
            return task_loss
        else:
            return task_loss, pred_num, task_loss.item() * pred_num
        
    def after_testing_callback(self, save_path: str, batch_num: int, logger: Logger, is_best: bool, **unused):
        # compute mean loss for the prediction part
        for phase in ['train', 'val']:

            if len(self.recon_loss_list[phase]) > 0:
                mean_recon_loss = np.mean(self.recon_loss_list[phase])
                logger.log_tabular(f'{phase}_recon_loss', mean_recon_loss)
                self.recon_loss_list[phase] = []
            if len(self.vae_loss_list[phase]) > 0:
                mean_vae_loss = np.mean(self.vae_loss_list[phase])
                logger.log_tabular(f'{phase}_vae_loss', mean_vae_loss)
                self.vae_loss_list[phase] = []

            pred_num = 0
            sum_loss = 0
            for data in self.data[phase]:
                end_loss = data[2][-self.pred_step: ]
                sum_loss += end_loss.sum()
                pred_num += np.prod(end_loss.shape)
            mean_loss = sum_loss / pred_num
            logger.log_tabular(f'{phase}_end_loss', mean_loss.item())

            # visualize prediction of the first/last pc for 10 random trials
            for pc in [0, -1]:
                preds = []
                targets = []
                for iter in range(10):
                    batch_idx = np.random.randint(len(self.data[phase]))
                    data = self.data[phase][batch_idx]
                    sample_idx = np.random.randint(data[0].shape[1])

                    if self.target_mode == 'raw':
                        pred = data[0][:, sample_idx, pc]
                        target = data[1][:, sample_idx, pc]
                        if self.loss_mode == 'prediction':
                            inp = data[3][:, sample_idx, pc]
                            pred = np.concatenate([inp, pred])
                            target = np.concatenate([inp, target])
                        preds.append(pred)
                        targets.append(target)
                    else:
                        assert self.loss_mode != 'prediction', 'Not implemented'
                        inp = data[3][:, sample_idx, pc]
                        raw_target = inp[-1] + np.cumsum(data[1][-self.pred_step:, sample_idx, pc])
                        targets.append(np.concatenate([inp[1: ], raw_target]))
                        pred = inp[-1] + np.cumsum(data[0][-self.pred_step:, sample_idx, pc])
                        preds.append(np.concatenate([inp[: -1] + data[0][: -self.pred_step, sample_idx, pc], pred]))

                plots.pred_vs_target_plot(
                    list(zip(preds, targets)), 
                    os.path.join(save_path, f'{phase}_pred_vs_target'), 
                    f'{batch_num}_dim{pc}',
                    len(preds[0]) - self.pred_step
                )

        if self.do_analysis:
            for phase in ['train', 'val']:
                self.curves[phase].append(self.error_dist_plot(self.data[phase], os.path.join(save_path, f'{phase}_error_dist'), batch_num))

        self.data = {'train': [], 'val': [], 'test': []}
    
    def after_training_callback(self, config, model):
        if not self.do_analysis:
            return
        # compare curve for the first and last epoch
        for phase in ['train', 'val']:
            plots.error_plot(
                np.arange(-5, 5, 0.1),
                [self.curves[phase][0], self.curves[phase][-1]], # [[curve1], [curve2]]
                'Delta F', 'Loss',
                label_list=['Initial', 'Final'],
                fig_name=phase,
                save_dir=config.save_path,
                mode='errorshade'
            )

    def error_dist_plot(self, data, save_path, batch_num):
        if len(data) == 0:
            return

        target = np.concatenate([d[1].reshape(-1) for d in data])
        error = np.concatenate([d[2].reshape(-1) for d in data])

        import matplotlib.pyplot as plt
        left = -5
        right = 5
        interval = 0.1

        curve = []
        for x in np.arange(left, right, interval).tolist():
            mask = (target >= x) & (target < x + interval)
            curve.append([(error * mask).sum()])
        x_axis = np.arange(left, right, interval)

        plots.error_plot(
            x_axis, [curve], 'Delta F', 'Loss',
            fig_name=f'{batch_num}',
            save_dir=save_path,
            mode='errorshade'
        )

        return curve