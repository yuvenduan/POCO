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

    def __init__(self, config: NeuralPredictionConfig, datum_size):
        self.criterion = nn.MSELoss(reduction='sum')
        self.do_analysis = config.do_analysis
        self.loss_mode = config.loss_mode
        self.pred_step = config.pred_length
        self.seq_len = config.seq_length
        self.target_mode = config.target_mode
        self.config = config
        self.datum_size = datum_size
        self._init_info()

        assert self.loss_mode in ['prediction', 'autoregressive', 'reconstruction', 'reconstruction_prediction']
        if self.loss_mode == 'reconstruction':
            assert self.pred_step == 0, 'reconstruction mode does not support prediction'
        else:
            assert self.pred_step > 0, 'prediction step should be larger than 0'

    def _init_info(self):
        self.sample_trials = {'train': [], 'val': [], 'test': []}
        self.pred_num = {'train': [], 'val': [], 'test': []}
        self.sum_mse = {'train': [], 'val': [], 'test': []}
        self.sum_mae = {'train': [], 'val': [], 'test': []}
        
        for phase in ['train', 'val', 'test']:
            for size in self.datum_size:
                self.pred_num[phase].append(np.zeros(size))
                L = self.pred_step if self.pred_step > 0 else self.seq_len
                self.sum_mse[phase].append(np.zeros((L, size)))
                self.sum_mae[phase].append(np.zeros((L, size))) # L, D

        self.recon_loss_list = {'train': [], 'val': [], 'test': []}
        self.vae_loss_list = {'train': [], 'val': [], 'test': []}

    def roll(self, model: nn.Module, data_batch: tuple, phase: str = 'train'):
        train_flag = phase == 'train'
        
        input, target, info_list = data_batch
        input, target = data_batch_to_device((input, target))
        output = model(input, pred_step=self.pred_step)
    
        task_loss = torch.zeros(1).to(DEVICE)
        pred_num = 0

        mean_list = []
        std_list = []

        for idx, (inp, out, tar, info) in enumerate(zip(input, output, target, info_list)):
            if target is None or np.prod(tar.shape) == 0:
                continue

            if self.loss_mode[: 14] == 'reconstruction':
                tar = torch.cat([inp[: 1], tar], dim=0)
            elif self.loss_mode == 'prediction':
                out = out[-self.pred_step: ]
                tar = tar[-self.pred_step: ]
            else:
                assert self.loss_mode == 'autoregressive'

            if len(self.sample_trials[phase]) < len(input):
                self.sample_trials[phase].append((
                    out.detach().cpu().numpy(), 
                    tar.detach().cpu().numpy(), 
                    inp.detach().cpu().numpy()
                ))
            
            with torch.no_grad():
                if self.pred_step > 0:
                    loss_array = torch.abs(out - tar)[-self.pred_step: ] # shape: L, B, D
                    assert loss_array.shape[0] == self.pred_step
                else:
                    loss_array = torch.abs(out - tar)

                self.sum_mse[phase][idx] += loss_array.pow(2).sum(dim=1).cpu().numpy()
                self.sum_mae[phase][idx] += loss_array.sum(dim=1).cpu().numpy()
                self.pred_num[phase][idx] += loss_array.shape[1]

            task_loss += self.criterion(out, tar) 
            pred_num += np.prod(tar.shape)

            tar = tar[-self.pred_step: ]
            mean_list.append(torch.mean(tar, dim=0).reshape(-1))
            std_list.append(torch.std(tar, dim=0).reshape(-1))

        task_loss = task_loss / max(1, pred_num)

        if isinstance(model, MultiFishSeqVAE):
            self.recon_loss_list[phase].append(task_loss.item())
            task_loss += model.kl_loss * self.config.kl_loss_coef / self.config.batch_size
            self.vae_loss_list[phase].append(model.kl_loss.item() / self.config.batch_size)
        elif isinstance(model, vqvae):
            self.recon_loss_list[phase].append(task_loss.item())
            task_loss += model.vq_loss
            self.vae_loss_list[phase].append(model.vq_loss.item())
        elif isinstance(model, Decoder):
            # here recon loss / vae loss denotes mean and std prediction loss
            assert self.loss_mode == 'prediction'
            mean_loss = F.mse_loss(model.mu, torch.cat(mean_list))
            std_loss = F.mse_loss(model.std, torch.cat(std_list))
            task_loss += (mean_loss + std_loss) * model.mu_std_loss_coef
            self.recon_loss_list[phase].append(mean_loss.item())
            self.vae_loss_list[phase].append(std_loss.item())

        if train_flag:
            return task_loss
        else:
            return task_loss, pred_num, task_loss.item() * pred_num
        
    def after_testing_callback(self, save_path: str, batch_num: int, logger: Logger, is_best: bool, **unused):

        # compute mean loss for the prediction part
        for phase in ['train', 'val']:

            if self.pred_num[phase] == 0:
                continue

            if len(self.recon_loss_list[phase]) > 0:
                mean_recon_loss = np.mean(self.recon_loss_list[phase])
                logger.log_tabular(f'{phase}_recon_loss', mean_recon_loss)
            if len(self.vae_loss_list[phase]) > 0:
                mean_vae_loss = np.mean(self.vae_loss_list[phase])
                logger.log_tabular(f'{phase}_vae_loss', mean_vae_loss)

            sum_pred_num = 0
            sum_mse = 0
            sum_mae = 0
            for idx in range(len(self.datum_size)):
                sum_pred_num += self.pred_num[phase][idx].sum() * (self.pred_step if self.pred_step > 0 else self.seq_len)
                sum_mse += self.sum_mse[phase][idx].sum()
                sum_mae += self.sum_mae[phase][idx].sum()

            logger.log_tabular(f'{phase}_mse', sum_mse / sum_pred_num)
            logger.log_tabular(f'{phase}_mae', sum_mae / sum_pred_num)
            
            if is_best and len(self.sample_trials[phase]) > 0:

                # visualize prediction of the first/last dimension for 10 random trials
                for pc in [0, 20, 50, -1]:
                    preds = []
                    targets = []

                    for iter in range(10):
                        batch_idx = np.random.randint(len(self.sample_trials[phase]))
                        data = self.sample_trials[phase][batch_idx]
                        sample_idx = np.random.randint(data[0].shape[1])

                        pred = data[0][:, sample_idx, pc]
                        target = data[1][:, sample_idx, pc]
                        inp = data[2][:, sample_idx, pc]

                        if self.target_mode == 'raw':
                            if self.loss_mode == 'prediction': # get the whole trial
                                pred = np.concatenate([inp, pred])
                                target = np.concatenate([inp, target])
                            preds.append(pred)
                            targets.append(target)
                        else:
                            assert self.loss_mode != 'prediction', 'Not implemented'
                            raw_target = inp[-1] + np.cumsum(data[1][-self.pred_step:, sample_idx, pc])
                            targets.append(np.concatenate([inp[1: ], raw_target]))
                            pred = inp[-1] + np.cumsum(data[0][-self.pred_step:, sample_idx, pc])
                            preds.append(np.concatenate([inp[: -1] + data[0][: -self.pred_step, sample_idx, pc], pred]))

                    plots.pred_vs_target_plot(
                        list(zip(preds, targets)), 
                        os.path.join(save_path, f'{phase}_pred_vs_target'), 
                        f'best_dim{pc}',
                        len(preds[0]) - self.pred_step
                    )


        if is_best:
            for phase in ['val']:
                info = {
                    'mse': self.sum_mse[phase],
                    'mae': self.sum_mae[phase],
                    'pred_num': self.pred_num[phase],
                    'sample_trials': self.sample_trials[phase]
                }
                np.save(os.path.join(save_path, f'{phase}_best_info.npy'), info)

        self._init_info()
    
    def after_training_callback(self, config, model):
        
        if not self.do_analysis:
            return