"""
Zebrafish dataset for neural prediction
"""

__all__ = ['Zebrafish', 'Simulation', 'VisualZebrafish']

import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import time

from tqdm import tqdm
from configs.configs import NeuralPredictionConfig
from configs.config_global import PROCESSED_DIR, SIM_DIR, VISUAL_PROCESSED_DIR
from utils.data_utils import get_exp_names, get_subject_ids

class NeuralDataset(tud.Dataset):

    def load_all_activities(self, config: NeuralPredictionConfig):
        """
        Return:
            a list containing neural activity for each animal, each element is a N * T matrix
        Should be implemented in the subclass
        """
        raise NotImplementedError
    
    def downsample(self, data: np.ndarray):
        if self.sampling_freq == 1:
            return data
        
        if self.sampling_mode == 'point':
            return data[:, ::self.sampling_freq]
        elif self.sampling_mode == 'avg':
            T = data.shape[1]
            new_T = T // self.sampling_freq
            data = data[:, : new_T * self.sampling_freq]
            return np.mean(data.reshape(data.shape[0], -1, self.sampling_freq), axis=2)
        else:
            raise NotImplementedError(self.sampling_mode)

    def __init__(self, config: NeuralPredictionConfig, phase='train'):

        self.neural_data = []
        self.animal_id = []
        self.datum_size = []
        self.pred_length = config.pred_length
        self.stimuli_dim = config.stimuli_dim
        self.normalize_coef = []

        animal_idx = 0
        self.sampling_freq = config.sampling_freq
        self.sampling_mode = 'avg'
        all_activities = self.load_all_activities(config)

        for all_activity in all_activities:
                    
            all_activity: np.ndarray
            # For each fish, split the data into patches
            length = all_activity.shape[1]
            if phase == 'train':
                length = min(length, config.train_data_length)
            patch_length = config.patch_length
            segments = max(1, length // patch_length)
            self.datum_size.append(all_activity.shape[0] - self.stimuli_dim)

            # different normalization methods
            if config.normalize_mode == 'minmax':
                Min = all_activity.min(axis=1, keepdims=True)
                Max = all_activity.max(axis=1, keepdims=True)
                all_activity = (all_activity - Min) / (Max - Min + 1e-4)
                all_activity = all_activity * 2 - 1
                assert np.all(all_activity >= -1) and np.all(all_activity <= 1)
                self.normalize_coef.append((Max - Min).reshape(-1) / 2)
            elif config.normalize_mode == 'zscore':
                Mean = all_activity.mean(axis=1, keepdims=True)
                Std = all_activity.std(axis=1, keepdims=True)
                all_activity = (all_activity - Mean) / (Std + 1e-4)
                self.normalize_coef.append(Std.reshape(-1))
            elif config.normalize_mode == 'none':
                self.normalize_coef.append(np.ones((all_activity.shape[0], )))
            else:
                raise NotImplementedError(config.normalize_mode)
            
            for patch in range(segments):
                patch_start = patch * patch_length
                patch_end = (patch + 1) * patch_length if patch < segments - 1 else length
                activity = all_activity[:, patch_start: patch_end]
                cur_length = activity.shape[1]

                train_split = int(cur_length * config.train_split)
                val_split = int(cur_length * (config.train_split + config.val_split))

                # split the data into train, val, test in each patch
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
        
        self.window_stride = config.test_set_window_stride if phase != 'train' else 1
        self.seq_length = config.seq_length
        self.data_length = [max(0, (data.shape[1] - self.seq_length - 1) // self.window_stride + 1) for data in self.neural_data]
        self.num_animals = animal_idx
        self.target_mode = config.target_mode


        logging.info(f"---- {phase} dataset info ----")
        if phase != 'train' and config.show_chance:
            self.baseline = self.get_detailed_chance_performance()
            logging.info(f"""
    copy mse/mae: {self.baseline['avg_copy_mse']:.6f} {self.baseline['avg_copy_mae']:.6f}
    chance mse/mae: {self.baseline['avg_chance_mse']:.6f} {self.baseline['avg_chance_mae']:.6f}
            """)
        
        logging.info(f"""
    num_animals: {self.num_animals}, total patch: {len(self.neural_data)}, total_length: {len(self)}
    input_size: {self.datum_size}
    data_length: {self.data_length}
------------------------------
        """)
        
    # reimplement get chance performance to reduce memory usage (by keeping only the sum of error and count)
        
    def get_detailed_chance_performance(self):
        assert self.stimuli_dim == 0
        assert self.target_mode == 'raw'

        pred_num = []
        sum_copy_mse = []
        sum_copy_mae = []
        sum_chance_mse = []
        sum_chance_mae = []
        act_list = []

        for size in self.datum_size:
            pred_num.append(np.zeros(size))
            if self.pred_length > 0:
                sum_copy_mse.append(np.zeros((self.pred_length, size)))
                sum_copy_mae.append(np.zeros((self.pred_length, size)))
            else:
                sum_copy_mse.append(np.zeros((self.seq_length, size)))
                sum_copy_mae.append(np.zeros((self.seq_length, size)))
            
            sum_chance_mse.append(np.zeros(size))
            sum_chance_mae.append(np.zeros(size))
            act_list.append(np.zeros(size))

        print('Calculating chance performance...')
        for idx in tqdm(range(len(self))):
            data, info = self[idx]

            data: np.ndarray = data
            animal_idx = info['animal_idx']

            if self.pred_length > 0:
                # for predicition, calculate the performance when copying the last frame
                error = data[-self.pred_length: ] - data[-self.pred_length - 1, :] # L * D
                act_list[animal_idx] += data[-self.pred_length: ].mean(axis=0)
            else:
                # for reconstruction, calculate the performance when copying the mean
                error = data - data.mean(axis=0)
                act_list[animal_idx] += data.mean(axis=0)
            
            pred_num[animal_idx] += 1
            sum_copy_mse[animal_idx] += np.square(error)
            sum_copy_mae[animal_idx] += np.abs(error)

        # get the mean activation for each channel
        for idx in range(self.num_animals):
            act_list[idx] = act_list[idx] / pred_num[idx]

        for idx in tqdm(range(len(self))):
            data, info = self[idx]
            data: np.ndarray = data
            animal_idx = info['animal_idx']
            if self.pred_length > 0:
                error = data[-self.pred_length: ] - act_list[animal_idx] # L * D
            else:
                error = data - act_list[animal_idx]
            sum_chance_mse[animal_idx] += np.square(error).mean(axis=0) # D
            sum_chance_mae[animal_idx] += np.abs(error).mean(axis=0) # D

        copy_mse = []
        copy_mae = []
        chance_mse = []
        chance_mae = []

        animal_chance_mse = []
        animal_chance_mae = []
        animal_copy_mse = []
        animal_copy_mae = []

        all_pred_num = 0
        all_copy_mse = 0
        all_copy_mae = 0
        all_chance_mse = 0
        all_chance_mae = 0

        for animal_idx in range(self.num_animals):
            copy_mse.append(sum_copy_mse[animal_idx].mean(axis=0) / pred_num[animal_idx])
            copy_mae.append(sum_copy_mae[animal_idx].mean(axis=0) / pred_num[animal_idx])
            chance_mse.append(sum_chance_mse[animal_idx] / pred_num[animal_idx])
            chance_mae.append(sum_chance_mae[animal_idx] / pred_num[animal_idx])

            animal_copy_mse.append(copy_mse[-1].mean())
            animal_copy_mae.append(copy_mae[-1].mean())
            animal_chance_mse.append(chance_mse[-1].mean())
            animal_chance_mae.append(chance_mae[-1].mean())

            all_pred_num += pred_num[animal_idx].sum()
            all_copy_mse += sum_copy_mse[animal_idx].mean(axis=0).sum()
            all_copy_mae += sum_copy_mae[animal_idx].mean(axis=0).sum()
            all_chance_mse += sum_chance_mse[animal_idx].sum()
            all_chance_mae += sum_chance_mae[animal_idx].sum()

        return_dict = {
            'pred_num': pred_num,
            'copy_mse': copy_mse,
            'copy_mae': copy_mae,
            'chance_mse': chance_mse,
            'chance_mae': chance_mae,
            'animal_chance_mse': animal_chance_mse,
            'animal_chance_mae': animal_chance_mae,
            'animal_copy_mse': animal_copy_mse,
            'animal_copy_mae': animal_copy_mae,
            'avg_copy_mse': all_copy_mse / all_pred_num,
            'avg_copy_mae': all_copy_mae / all_pred_num,
            'avg_chance_mse': all_chance_mse / all_pred_num,
            'avg_chance_mae': all_chance_mae / all_pred_num,
        }

        # calculate the mean prediction performance for different PCs by doing weighted averaging
        if np.all(np.array(self.datum_size) == self.datum_size[0]):
            sum_pred_num = np.zeros(self.datum_size[0])
            sum_copy_mse = np.zeros(self.datum_size[0])
            sum_copy_mae = np.zeros(self.datum_size[0])
            sum_chance_mse = np.zeros(self.datum_size[0])
            sum_chance_mae = np.zeros(self.datum_size[0])

            for idx in range(len(self.datum_size)):
                sum_pred_num += pred_num[idx]
                sum_copy_mse += copy_mse[idx] * pred_num[idx]
                sum_copy_mae += copy_mae[idx] * pred_num[idx]
                sum_chance_mse += chance_mse[idx] * pred_num[idx]
                sum_chance_mae += chance_mae[idx] * pred_num[idx]

            return_dict['mean_copy_mse'] = sum_copy_mse / sum_pred_num
            return_dict['mean_copy_mae'] = sum_copy_mae / sum_pred_num
            return_dict['mean_chance_mse'] = sum_chance_mse / sum_pred_num
            return_dict['mean_chance_mae'] = sum_chance_mae / sum_pred_num

        return return_dict

    def __len__(self):
        return sum(self.data_length)

    def __getitem__(self, idx):
        patch_idx = 0
        while idx >= self.data_length[patch_idx]:
            idx -= self.data_length[patch_idx]
            patch_idx += 1

        animal_idx = self.animal_id[patch_idx]
        pos = idx * self.window_stride
        data = self.neural_data[patch_idx][:, pos: pos + self.seq_length].transpose()
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
                input_list.append(torch.zeros(self.seq_length - self.pred_length, 0, self.datum_size[animal_idx] + self.stimuli_dim))
                target_list.append(torch.zeros(self.seq_length - 1, 0, self.datum_size[animal_idx]))
                info_list.append({'animal_idx': animal_idx, 'time_idx': [], 'normalize_coef': self.normalize_coef[animal_idx]})
                continue

            data = torch.stack([torch.from_numpy(batch[idx][0]).float() for idx in indices], dim=1) # L * B * D
            input_list.append(data[: -self.pred_length] if self.pred_length > 0 else data)
            target_dim = data.shape[2] - self.stimuli_dim

            if self.target_mode == 'raw':
                target_list.append(data[1:, :, : target_dim])
            elif self.target_mode == 'derivative':
                target_list.append(data[1:, :, : target_dim] - data[: -1, :, : target_dim])
            else:
                raise NotImplementedError

            info_list.append({
                'animal_idx': animal_idx, 
                'time_idx': [batch[idx][1]['time_idx'] for idx in indices],
                'normalize_coef': self.normalize_coef[animal_idx]
            })

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
                    if config.pc_dim is None and config.fc_dim is None:
                        all_regions = [
                            'in_l_LHb', 'in_l_MHb', 'in_l_ctel', 'in_l_dthal', 'in_l_gc', 'in_l_raphe', 'in_l_tel', 'in_l_vent', 'in_l_vthal',
                            'in_r_LHb', 'in_r_MHb', 'in_r_ctel', 'in_r_dthal', 'in_r_gc', 'in_r_raphe', 'in_r_tel', 'in_r_vent', 'in_r_vthal'
                        ]
                        roi_indices = [fishdata[region] for region in all_regions]
                        
                        if config.brain_regions == 'all':
                            # use all brain regions
                            all_activity: np.ndarray = fishdata['M']
                        elif config.brain_regions == 'average':
                            # use average activity of all brain regions
                            all_activity = np.empty((len(all_regions), fishdata['M'].shape[1]))
                            for i, roi_index in enumerate(roi_indices):
                                if np.sum(roi_index) == 0:
                                    all_activity[i] = np.zeros((fishdata['M'].shape[1], ))
                                else:
                                    all_activity[i] = fishdata['M'][roi_index].mean(axis=0)
                            assert config.normalize_mode == 'zscore', 'Recommend using zscore normalization when using averaged activity'
                        else:
                            # use specific brain regions
                            if not isinstance(config.brain_regions, (list, tuple)):
                                config.brain_regions = [config.brain_regions]
                            use_indices = np.zeros_like(roi_indices[0])
                            for region in config.brain_regions:
                                region = 'in_' + region
                                assert region in all_regions, f'Invalid brain region {region}'
                                use_indices |= roi_indices[all_regions.index(region)]
                            all_activity: np.ndarray = fishdata['M'][use_indices]
                    elif config.fc_dim is not None:
                        assert config.brain_regions == 'all', 'Only support using all brain regions when using FC'
                        all_activity: np.ndarray = fishdata[f'FC_{config.fc_dim}']
                    else:
                        assert config.brain_regions == 'all', 'Only support using all brain regions when using PC'
                        all_activity: np.ndarray = fishdata['PC'][: config.pc_dim]
                    
                    all_activity = self.downsample(all_activity)
                    all_activities.append(all_activity)
        
        return all_activities
    
class Simulation(NeuralDataset):

    def load_all_activities(self, config: NeuralPredictionConfig):
        
        name = f'sim_{config.n_neurons}_{config.n_regions}_{config.ga}_{config.sim_noise_std}'
        if config.sparsity != 1:
            name += f'_sparsity_{config.sparsity}'
        filename = os.path.join(SIM_DIR, f'{name}.npz')

        data = np.load(filename)
        if config.pc_dim is None:
            all_activity: np.ndarray = data['M']
            n = all_activity.shape[0]
            if config.portion_observable_neurons < 1:
                n_observable = int(n * config.portion_observable_neurons)
                all_activity = all_activity[: n_observable]
        else:
            all_activity: np.ndarray = data['PC'][: config.pc_dim]
        
        all_activity = self.downsample(all_activity)
        return [all_activity]

class VisualZebrafish(NeuralDataset):

    def load_all_activities(self, config: NeuralPredictionConfig):
        subject_ids = get_subject_ids()
        all_activities = []
        for id in subject_ids:
            if config.animal_ids != 'all' and id not in config.animal_ids:
                continue
            filename = os.path.join(VISUAL_PROCESSED_DIR, f'subject_{id}.npz')

            with np.load(filename) as fishdata:
                if config.pc_dim is None:
                    all_activity: np.ndarray = fishdata['M']
                else:
                    all_activity: np.ndarray = fishdata['PC'][: config.pc_dim]

                if config.use_eye_movements:
                    if 'eye' not in fishdata:
                        print("Subject {} does not have eye movements".format(id))
                        continue
                    eye = fishdata['eye_motorseed']
                    assert np.all(eye != np.nan)
                    all_activity = np.concatenate([all_activity, eye], axis=0)

                if config.use_motor:
                    if 'behavior' not in fishdata:
                        print("Subject {} does not have motor data".format(id))
                        continue
                    motor = fishdata['behavior_motorseed']
                    assert np.all(motor != np.nan)
                    all_activity = np.concatenate([all_activity, motor], axis=0)

                if config.use_stimuli:
                    s = fishdata['stim']
                    # change to one-hot
                    s = np.eye(config.stimuli_dim)[s].T
                    all_activity = np.concatenate([all_activity, s], axis=0)
                    assert config.stimuli_dim == 24

                all_activity = self.downsample(all_activity)
                all_activities.append(all_activity)
        return all_activities
    
def get_baseline_performance(config, phase='train'):
    config.show_chance = True
    if config.dataset == 'zebrafish':
        dataset = Zebrafish(config, phase=phase)
    elif config.dataset == 'simulation':
        dataset = Simulation(config, phase=phase)
    elif config.dataset == 'zebrafish_visual':
        dataset = VisualZebrafish(config, phase=phase)

    return dataset.baseline