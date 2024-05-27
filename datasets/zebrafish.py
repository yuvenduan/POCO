"""
Zebrafish dataset for neural prediction
"""

__all__ = ['Zebrafish', 'Simulation']

import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os

from configs.configs import NeuralPredictionConfig
from configs.config_global import PROCESSED_DIR, SIM_DIR
from utils.data_utils import get_exp_names

class NeuralDataset(tud.Dataset):

    def load_all_activities(self, config: NeuralPredictionConfig):
        """
        Return:
            a list containing neural activity for each animal, each element is a N * T matrix
        Should be implemented in the subclass
        """
        raise NotImplementedError

    def __init__(self, config: NeuralPredictionConfig, phase='train'):

        self.neural_data = []
        self.animal_id = []
        self.datum_size = []
        self.pred_length = config.pred_length

        animal_idx = 0
        all_activities = self.load_all_activities(config)

        for all_activity in all_activities:
                    
            all_activity: np.ndarray
            # For each fish, split the data into patches
            length = all_activity.shape[1]
            if phase == 'train':
                length = min(length, config.train_data_length)
            patch_length = config.patch_length
            segments = max(1, length // patch_length)
            self.datum_size.append(all_activity.shape[0])

            if config.normalize_mode == 'minmax':
                Min = all_activity.min(axis=1, keepdims=True)
                Max = all_activity.max(axis=1, keepdims=True)
                all_activity = (all_activity - Min) / (Max - Min + 1e-4)
                all_activity = all_activity * 2 - 1
                assert np.all(all_activity >= -1) and np.all(all_activity <= 1)
            elif config.normalize_mode == 'zscore':
                Mean = all_activity.mean(axis=1, keepdims=True)
                Std = all_activity.std(axis=1, keepdims=True)
                all_activity = (all_activity - Mean) / (Std + 1e-4)
            elif config.normalize_mode == 'none':
                pass
            else:
                raise NotImplementedError(config.normalize_mode)

            for patch in range(segments):
                patch_start = patch * patch_length
                patch_end = (patch + 1) * patch_length if patch < segments - 1 else length
                activity = all_activity[:, patch_start: patch_end]
                cur_length = activity.shape[1]

                train_split = int(cur_length * config.train_split)
                val_split = int(cur_length * (config.train_split + config.val_split))

                # split the data into train, val, test
                if phase == 'train':
                    activity = activity[:, : train_split]
                elif phase == 'val':
                    activity = activity[:, train_split: val_split]
                elif phase == 'test':
                    activity = activity[:, val_split: ]
                else:
                    raise NotImplementedError(phase)
                
                self.animal_id.append(animal_idx)
                self.neural_data.append(activity)

            animal_idx += 1
        
        self.seq_length = config.seq_length
        self.data_length = [max(0, data.shape[1] - self.seq_length) for data in self.neural_data]
        self.num_animals = animal_idx

        self.get_chance_performance()
        logging.info(f'Phase {phase}, copy performance: {self.copy_performance:.6f}, chance performance: {self.chance_performance:.6f}')
        logging.info(f'Phase {phase}, num_animals: {self.num_animals}, total patch: {len(self.neural_data)}, datum_size: {self.datum_size}, data_length: {self.data_length}')

        self.target_mode = config.target_mode

    def get_chance_performance(self):
        loss_list = []
        act_list = []
        for idx in range(len(self)):
            data, info = self[idx]
            error = data[-self.pred_length: ] - data[-self.pred_length - 1]
            loss_list.append(error.reshape(-1))
            act_list.append(data[-self.pred_length: ].reshape(-1))

        loss_list = np.concatenate(loss_list)
        act_list = np.concatenate(act_list)
        self.copy_performance = np.mean(np.square(loss_list))
        self.chance_performance = np.mean(np.square(act_list - np.mean(act_list)))

    def __len__(self):
        return sum(self.data_length)

    def __getitem__(self, idx):
        patch_idx = 0
        while idx >= self.data_length[patch_idx]:
            idx -= self.data_length[patch_idx]
            patch_idx += 1

        animal_idx = self.animal_id[patch_idx]
        data = self.neural_data[patch_idx][:, idx: idx + self.seq_length].transpose()
        data = torch.from_numpy(data).float()

        info = {'animal_idx': animal_idx, 'time_idx': idx, }

        return data, info
    
    def collate_fn(self, batch):
        # group the data by fish
        animal_indices = [[] for _ in range(self.num_animals)]
        for idx, (data, info) in enumerate(batch):
            animal_indices[info['animal_idx']].append(idx)

        input_list = []
        target_list = []
        info_list = []
        for animal_idx, indices in enumerate(animal_indices):
            if len(indices) == 0:
                input_list.append(torch.zeros(self.seq_length - self.pred_length, 0, self.datum_size[animal_idx]))
                target_list.append(torch.zeros(self.seq_length - 1, 0, self.datum_size[animal_idx]))
                info_list.append({'animal_idx': animal_idx, 'time_idx': []})
                continue

            data = torch.stack([batch[idx][0] for idx in indices], dim=1) # L * B * D
            input_list.append(data[: -self.pred_length])

            if self.target_mode == 'raw':
                target_list.append(data[1: ])
            elif self.target_mode == 'derivative':
                target_list.append(data[1: ] - data[: -1])
            else:
                raise NotImplementedError

            info_list.append({'animal_idx': animal_idx, 'time_idx': [batch[idx][1]['time_idx'] for idx in indices]})

        return input_list, target_list, info_list

class Zebrafish(NeuralDataset):

    def load_all_activities(self, config: NeuralPredictionConfig):
        exp_names = get_exp_names()
        all_activities = []
        for exp_type in config.exp_types:
            for i_exp, exp_name in enumerate(exp_names[exp_type]):
                if config.animal_ids != 'all' and i_exp not in config.animal_ids:
                    continue

                filename = os.path.join(PROCESSED_DIR, exp_name + '.npz')

                with np.load(filename) as fishdata:
                    if config.pc_dim is None:
                        all_activity: np.ndarray = fishdata['M']
                    else:
                        all_activity: np.ndarray = fishdata['PC'][: config.pc_dim]
                    
                    all_activities.append(all_activity)
        
        return all_activities
    
class Simulation(NeuralDataset):

    def load_all_activities(self, config: NeuralPredictionConfig):
        filename = os.path.join(SIM_DIR, f'sim_{config.n_neurons}_{config.n_regions}.npz')

        data = np.load(filename)
        if config.pc_dim is None:
            all_activity: np.ndarray = data['M']
        else:
            all_activity: np.ndarray = data['PC'][: config.pc_dim]
        
        return [all_activity]
    
def get_baseline_performance(config=None, phase='train'):
    if config.dataset == 'zebrafish':
        dataset = Zebrafish(config, phase=phase)
    elif config.dataset == 'simulation':
        dataset = Simulation(config, phase=phase)
    return min(dataset.copy_performance, dataset.chance_performance)