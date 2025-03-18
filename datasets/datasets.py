"""
Zebrafish dataset for neural prediction
"""

__all__ = ['Zebrafish', 'ZebrafishAhrens', 'ZebrafishJain', 'Simulation', 'StimZebrafish', 'Celegans', 'CelegansFlavell', 'Mice', 'get_baseline_performance']

import torch
import torch.utils.data as tud
import numpy as np
import logging
import os

from tqdm import tqdm
from configs.configs import DatasetConfig
from configs.config_global import \
    ZEBRAFISH_PROCESSED_DIR, ZEBRAFISH_STIM_PROCESSED_DIR, ZEBRAFISH_AHRENS_PROCESSED_DIR, N_ZEBRAFISH_AHNRENS_SESSIONS, \
    CELEGANS_PROCESSED_DIR, N_CELEGANS_SESSIONS, CELEGANS_FLAVELL_PROCESSED_DIR, N_CELEGANS_FLAVELL_SESSIONS, \
    SIM_DIR, MICE_PROCESSED_DIR, ZEBRAFISH_JAIN_PROCESSED_DIR
from utils.data_utils import get_exp_names, get_stim_exp_names, get_mice_sessions

class NeuralDataset(tud.Dataset):

    def load_all_activities(self, config: DatasetConfig):
        """
        Return: (activity, unit_types)
            activity: a list containing neural activity for each session, each element is a N * T matrix
            unit_type: (optional) a list containing unit types for each session, each element is an array of N integers
            unit type could be brain region, cell type, rank of pc etc.
        Should be implemented in the subclass
        """
        raise NotImplementedError
    
    def downsample(self, data: np.ndarray):
        """
        Downsample the data by the sampling frequency
        data: N * T matrix
        """
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

    def __init__(self, config: DatasetConfig, phase='train'):

        self.neural_data = []
        self.session_id = []
        self.input_size = [] # used in init_single_dataset
        self.pred_length = config.pred_length
        self.normalize_coef = []

        session_idx = 0
        self.sampling_freq = config.sampling_freq
        self.sampling_mode = config.sampling_mode
        if isinstance(config.session_ids, int):
            config.session_ids = [config.session_ids]
        all_activities, unit_types = self.load_all_activities(config)
        self.unit_types = unit_types # used in init_single_dataset

        logging.info(f"Raw Activity Shape: {[activity.shape for activity in all_activities]}")

        for all_activity in all_activities:
                    
            all_activity: np.ndarray
            # For each fish, split the data into patches
            length = all_activity.shape[1]
            if phase == 'train':
                length = min(length, config.train_data_length)
            patch_length = config.patch_length
            segments = max(1, length // patch_length)
            self.input_size.append(all_activity.shape[0])

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
                
                self.session_id.append(session_idx)
                self.neural_data.append(activity)

            session_idx += 1
        
        self.window_stride = config.test_set_window_stride if phase != 'train' else 1
        self.seq_length = config.seq_length
        self.data_length = [max(0, (data.shape[1] - self.seq_length - 1) // self.window_stride + 1) for data in self.neural_data]
        self.num_sessions = session_idx

        logging.info(f"---- {config.dataset} {phase} dataset info ----")
        if phase != 'train':
            self.baseline = self.get_detailed_chance_performance()
            logging.info(f"""
    copy mse/mae: {self.baseline['avg_copy_mse']:.6f} {self.baseline['avg_copy_mae']:.6f}
    chance mse/mae: {self.baseline['avg_chance_mse']:.6f} {self.baseline['avg_chance_mae']:.6f}
            """)
        
        logging.info(f"""
    num_sessions: {self.num_sessions}, total patch: {len(self.neural_data)}, total_length: {len(self)}
    input_size: {self.input_size}
    total data_length: {sum(self.data_length)}
------------------------------
        """)
        
    # reimplement get chance performance to reduce memory usage (by keeping only the sum of error and count)
        
    def get_detailed_chance_performance(self):
        """
        See get_baseline_performance for explanation of the return value
        """

        pred_num = []
        sum_copy_mse = []
        sum_copy_mae = []
        sum_chance_mse = []
        sum_chance_mae = []
        act_list = []

        for size in self.input_size:
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
            session_idx = info['session_idx']

            if self.pred_length > 0:
                # for predicition, calculate the performance when copying the last frame
                error = data[-self.pred_length: ] - data[-self.pred_length - 1, :] # L * D
                act_list[session_idx] += data[-self.pred_length: ].mean(axis=0)
            else:
                # for reconstruction, calculate the performance when copying the mean
                error = data - data.mean(axis=0)
                act_list[session_idx] += data.mean(axis=0)
            
            pred_num[session_idx] += 1
            sum_copy_mse[session_idx] += np.square(error)
            sum_copy_mae[session_idx] += np.abs(error)

        # get the mean activation for each channel
        for idx in range(self.num_sessions):
            act_list[idx] = act_list[idx] / pred_num[idx]

        for idx in tqdm(range(len(self))):
            data, info = self[idx]
            data: np.ndarray = data
            session_idx = info['session_idx']
            if self.pred_length > 0:
                error = data[-self.pred_length: ] - act_list[session_idx] # L * D
            else:
                error = data - act_list[session_idx]
            sum_chance_mse[session_idx] += np.square(error).mean(axis=0) # D
            sum_chance_mae[session_idx] += np.abs(error).mean(axis=0) # D

        copy_mse = []
        copy_mae = []
        chance_mse = []
        chance_mae = []

        session_chance_mse = []
        session_chance_mae = []
        session_copy_mse = []
        session_copy_mae = []

        all_pred_num = 0
        all_copy_mse = 0
        all_copy_mae = 0
        all_chance_mse = 0
        all_chance_mae = 0

        for session_idx in range(self.num_sessions):
            copy_mse.append(sum_copy_mse[session_idx].mean(axis=0) / pred_num[session_idx])
            copy_mae.append(sum_copy_mae[session_idx].mean(axis=0) / pred_num[session_idx])
            chance_mse.append(sum_chance_mse[session_idx] / pred_num[session_idx])
            chance_mae.append(sum_chance_mae[session_idx] / pred_num[session_idx])

            session_copy_mse.append(copy_mse[-1].mean())
            session_copy_mae.append(copy_mae[-1].mean())
            session_chance_mse.append(chance_mse[-1].mean())
            session_chance_mae.append(chance_mae[-1].mean())

            all_pred_num += pred_num[session_idx].sum()
            all_copy_mse += sum_copy_mse[session_idx].mean(axis=0).sum()
            all_copy_mae += sum_copy_mae[session_idx].mean(axis=0).sum()
            all_chance_mse += sum_chance_mse[session_idx].sum()
            all_chance_mae += sum_chance_mae[session_idx].sum()

        return_dict = {
            'pred_num': pred_num,
            'copy_mse': copy_mse,
            'copy_mae': copy_mae,
            'chance_mse': chance_mse,
            'chance_mae': chance_mae,
            'session_chance_mse': session_chance_mse,
            'session_chance_mae': session_chance_mae,
            'session_copy_mse': session_copy_mse,
            'session_copy_mae': session_copy_mae,
            'avg_copy_mse': all_copy_mse / all_pred_num,
            'avg_copy_mae': all_copy_mae / all_pred_num,
            'avg_chance_mse': all_chance_mse / all_pred_num,
            'avg_chance_mae': all_chance_mae / all_pred_num,
        }

        # calculate the mean prediction performance for different PCs by doing weighted averaging
        if np.all(np.array(self.input_size) == self.input_size[0]):
            sum_pred_num = np.zeros(self.input_size[0])
            sum_copy_mse = np.zeros(self.input_size[0])
            sum_copy_mae = np.zeros(self.input_size[0])
            sum_chance_mse = np.zeros(self.input_size[0])
            sum_chance_mae = np.zeros(self.input_size[0])

            for idx in range(len(self.input_size)):
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

        session_idx = self.session_id[patch_idx]
        pos = idx * self.window_stride
        data = self.neural_data[patch_idx][:, pos: pos + self.seq_length].transpose()
        info = {'session_idx': session_idx, 'time_idx': idx, }

        return data, info
    
    def collate_fn(self, batch):
        # group the data by session
        session_indices = [[] for _ in range(self.num_sessions)]
        for idx, (data, info) in enumerate(batch):
            session_indices[info['session_idx']].append(idx)

        input_list = []
        target_list = []
        info_list = []
        for session_idx, indices in enumerate(session_indices):
            if len(indices) == 0:
                input_list.append(torch.zeros(self.seq_length - self.pred_length, 0, self.input_size[session_idx]))
                target_list.append(torch.zeros(self.seq_length - 1, 0, self.input_size[session_idx]))
                info_list.append({'session_idx': session_idx, 'time_idx': [], 'normalize_coef': self.normalize_coef[session_idx]})
                continue

            data = torch.stack([torch.from_numpy(batch[idx][0]).float() for idx in indices], dim=1) # L * B * D
            input_list.append(data[: -self.pred_length] if self.pred_length > 0 else data)
            target_dim = data.shape[2]
            target_list.append(data[1:, :, : target_dim])

            info_list.append({
                'session_idx': session_idx, 
                'time_idx': [batch[idx][1]['time_idx'] for idx in indices],
                'normalize_coef': self.normalize_coef[session_idx]
            })

        return input_list, target_list, info_list

class Zebrafish(NeuralDataset):

    all_regions = [
        'in_l_LHb', 'in_l_MHb', 'in_l_ctel', 'in_l_dthal', 'in_l_gc', 'in_l_raphe', 'in_l_tel', 'in_l_vent', 'in_l_vthal',
        'in_r_LHb', 'in_r_MHb', 'in_r_ctel', 'in_r_dthal', 'in_r_gc', 'in_r_raphe', 'in_r_tel', 'in_r_vent', 'in_r_vthal'
    ]

    def get_activity_unit_id(self, filename, config: DatasetConfig):
        """
        Load the neural activity from the file
        Based on the config, return the activity of specific brain regions, the average activity of all brain regions, or principal components of whole-brain activity
        """
        unit_id = None

        with np.load(filename) as fishdata:
            if config.pc_dim is None and config.fc_dim is None:
                roi_indices = [fishdata[region] for region in self.all_regions]
                
                if config.brain_regions == 'all':
                    # use all brain regions
                    activity: np.ndarray = fishdata['M']
                    unit_id = np.zeros((activity.shape[0], ))
                    for id, region in enumerate(self.all_regions):
                        unit_id[roi_indices[id]] = id + 1

                elif config.brain_regions == 'average':
                    # use average activity of all brain regions
                    activity = np.empty((len(self.all_regions), fishdata['M'].shape[1]))
                    for i, roi_index in enumerate(roi_indices):
                        if np.sum(roi_index) == 0:
                            activity[i] = np.zeros((fishdata['M'].shape[1], ))
                        else:
                            activity[i] = fishdata['M'][roi_index].mean(axis=0)
                    assert config.normalize_mode == 'zscore', 'Recommend using zscore normalization when using averaged activity'
                else:
                    # use specific brain regions
                    if not isinstance(config.brain_regions, (list, tuple)):
                        config.brain_regions = [config.brain_regions]
                    
                    use_indices = np.zeros_like(roi_indices[0])
                    unit_ids = np.zeros_like(roi_indices[0])

                    for id, region in enumerate(config.brain_regions):
                        region = 'in_' + region
                        assert region in self.all_regions, f'Invalid brain region {region}'
                        use_indices |= roi_indices[self.all_regions.index(region)]
                        unit_ids[roi_indices[self.all_regions.index(region)]] = id + 1

                    activity: np.ndarray = fishdata['M'][use_indices]
                    unit_id = unit_ids[use_indices]

            elif config.fc_dim is not None:
                assert config.brain_regions == 'all', 'Only support using all brain regions when using FC'
                activity: np.ndarray = fishdata[f'FC_{config.fc_dim}']
            else:
                assert config.brain_regions == 'all', 'Only support using all brain regions when using PC'
                activity: np.ndarray = fishdata['PC'][: config.pc_dim]

        if unit_id is None:
            unit_id = np.arange(activity.shape[0])
        
        return activity, unit_id.astype(int)

    def load_all_activities(self, config: DatasetConfig):
        exp_names = get_exp_names()
        all_activities = []
        all_unit_types = []
        n_sessions = 0

        for exp_type in config.exp_types:
            for i_exp, exp_name in enumerate(exp_names[exp_type]):
                n_sessions += 1
                if config.session_ids != None and n_sessions - 1 not in config.session_ids:
                    continue
                if config.animal_ids != None and i_exp not in config.animal_ids:
                    continue

                filename = os.path.join(ZEBRAFISH_PROCESSED_DIR, exp_name + '.npz')
                all_activity, unit_type = self.get_activity_unit_id(filename, config)
                all_activity = self.downsample(all_activity)
                all_activities.append(all_activity)
                all_unit_types.append(unit_type)

        return all_activities, all_unit_types
    
class StimZebrafish(Zebrafish):

    all_regions = [
        'in_l_LHb', 'in_l_MHb', 'in_l_cerebellum', 'in_l_di', 'in_l_dthal', 'in_l_hind', 'in_l_meso', 'in_l_preoptic', 'in_l_raphe', 'in_l_tectum', 'in_l_tel', 'in_l_vthal', 
        'in_r_LHb', 'in_r_MHb', 'in_r_cerebellum', 'in_r_di', 'in_r_dthal', 'in_r_hind', 'in_r_meso', 'in_r_preoptic', 'in_r_raphe', 'in_r_tectum', 'in_r_tel', 'in_r_vthal'
    ]

    def load_all_activities(self, config: DatasetConfig):
        exp_names = get_stim_exp_names()
        all_activities = []
        all_unit_types = []
        n_sessions = 0

        data_dir = ZEBRAFISH_STIM_PROCESSED_DIR
        if config.filter_type != 'none':
            data_dir = data_dir + f'_{config.filter_type}'

        for exp_type in config.exp_types:
            for i_exp, exp_name in enumerate(exp_names[exp_type]):
                n_sessions += 1
                if config.session_ids != None and n_sessions - 1 not in config.session_ids:
                    continue
                if config.animal_ids != None and i_exp not in config.animal_ids:
                    continue

                filename = os.path.join(data_dir, exp_name + '.npz')
                all_activity, unit_type = self.get_activity_unit_id(filename, config)
                all_activity = self.downsample(all_activity)
                all_activities.append(all_activity)
                all_unit_types.append(unit_type)

        return all_activities, all_unit_types
    
class ZebrafishAhrens(Zebrafish):

    def get_activity_unit_id(self, filename, config):
        """
        Load the neural activity from the file
        """
        with np.load(filename) as data:
            assert config.fc_dim is None, 'Only support using PC for Ahrens data'
            if config.pc_dim is None:
                activity: np.ndarray = data['M']
                unit_type = np.zeros((activity.shape[0], ))
            else:
                assert config.brain_regions == 'all', 'Only support using all brain regions when using PC'
                activity: np.ndarray = data['PC'][: config.pc_dim]
                unit_type = np.arange(activity.shape[0])
        return activity, unit_type.astype(int)

    def load_all_activities(self, config: DatasetConfig):
        all_activities = []
        all_unit_types = []
        n_sessions = 0

        data_dir = ZEBRAFISH_AHRENS_PROCESSED_DIR
        if config.filter_type != 'none':
            data_dir = data_dir + f'_{config.filter_type}'

        for i in range(N_ZEBRAFISH_AHNRENS_SESSIONS):
            n_sessions += 1
            if config.session_ids != None and n_sessions - 1 not in config.session_ids:
                continue
            if config.animal_ids != None and i not in config.animal_ids:
                continue

            filename = os.path.join(data_dir, f'{i}.npz')
            all_activity, unit_type = self.get_activity_unit_id(filename, config)
            all_activity = self.downsample(all_activity)
            all_activities.append(all_activity)
            all_unit_types.append(unit_type)

        return all_activities, all_unit_types
    
class ZebrafishJain(ZebrafishAhrens):

    def load_all_activities(self, config):
        all_activities = []
        all_unit_types = []
        n_sessions = 0

        data_dir = ZEBRAFISH_JAIN_PROCESSED_DIR
        if config.filter_type != 'none':
            data_dir = data_dir + f'_{config.filter_type}'

        for i in range(1):
            n_sessions += 1
            if config.session_ids != None and n_sessions - 1 not in config.session_ids:
                continue
            if config.animal_ids != None and i not in config.animal_ids:
                continue

            filename = os.path.join(data_dir, f'{i}.npz')
            all_activity, unit_type = self.get_activity_unit_id(filename, config)
            all_activity = self.downsample(all_activity)
            all_activities.append(all_activity)
            all_unit_types.append(unit_type)

        return all_activities, all_unit_types

class Celegans(NeuralDataset):

    data_dir = CELEGANS_PROCESSED_DIR
    n_sessions = N_CELEGANS_SESSIONS

    def get_activity_unit_id(self, filename, config: DatasetConfig):
        """
        Load the neural activity from the file
        """
        with np.load(filename) as data:
            assert config.fc_dim is None, 'Only support using PC for C. elegans data'
            if config.pc_dim is None:
                activity: np.ndarray = data['M']
                unit_type = np.zeros((activity.shape[0], ))
                for i, name in enumerate(data['neuron_names']):
                    if name in self.all_neuron_names:
                        unit_type[i] = self.all_neuron_names.index(name) + 1
            else:
                assert config.brain_regions == 'all', 'Only support using all brain regions when using PC'
                activity: np.ndarray = data['PC'][: config.pc_dim]
                unit_type = np.arange(activity.shape[0])
        return activity, unit_type.astype(int)

    def load_all_activities(self, config: DatasetConfig):
        all_activities = []
        all_unit_types = []

        data_dir = self.data_dir
        if config.filter_type != 'none':
            data_dir = data_dir + f'_{config.filter_type}'

        all_neuron_names = []
        for i in range(self.n_sessions):
            filename = os.path.join(data_dir, f'{i}.npz')
            with np.load(filename) as data:
                for name in data['neuron_names']:
                    if name[0] >= 'A' and name[0] <= 'Z' and name not in all_neuron_names:
                        all_neuron_names.append(name)
        self.all_neuron_names = all_neuron_names

        for idx in range(self.n_sessions):
            if config.session_ids != None and idx not in config.session_ids:
                continue
            if config.animal_ids != None and idx not in config.animal_ids:
                continue
            filename = os.path.join(data_dir, f'{idx}.npz')

            all_activity, unit_type = self.get_activity_unit_id(filename, config)
            all_activity = self.downsample(all_activity)
            all_activities.append(all_activity)
            all_unit_types.append(unit_type)

        return all_activities, all_unit_types
    
class CelegansFlavell(Celegans):

    data_dir = CELEGANS_FLAVELL_PROCESSED_DIR
    n_sessions = N_CELEGANS_FLAVELL_SESSIONS
    
class Mice(Celegans):

    def get_activity_unit_id(self, filename, config: DatasetConfig):
        """
        Load the neural activity from the file
        """
        with np.load(filename) as data:
            assert config.fc_dim is None, 'Only support using PC for mice data'
            if config.pc_dim is None:
                activity: np.ndarray = data['M']
                unit_type = data['area_ids']
            else:
                assert config.brain_regions == 'all', 'Only support using all brain regions when using PC'
                activity: np.ndarray = data['PC'][: config.pc_dim]
                unit_type = np.arange(activity.shape[0])
        return activity, unit_type.astype(int)

    def load_all_activities(self, config: DatasetConfig):
        all_activities = []
        all_unit_types = []
        all_sessions = get_mice_sessions()
        n_sessions = 0

        data_dir = MICE_PROCESSED_DIR
        if config.filter_type != 'none':
            data_dir = data_dir + f'_{config.filter_type}'

        for idx, (mouse, sessions) in enumerate(all_sessions.items()):
            for session in sessions:
                n_sessions += 1
                if config.session_ids != None and n_sessions - 1 not in config.session_ids:
                    continue
                if config.animal_ids != None and idx not in config.animal_ids:
                    continue

                filename = os.path.join(data_dir, f'{mouse}_{session}.npz')
                activity, unit_type = self.get_activity_unit_id(filename, config)
                activity = self.downsample(activity)
                all_activities.append(activity)
                all_unit_types.append(unit_type)
        
        return all_activities, all_unit_types

class Simulation(NeuralDataset):

    def load_all_activities(self, config: DatasetConfig):

        all_activities = []
        all_unit_types = []

        for seed in range(4):
            if config.session_ids != None and seed not in config.session_ids:
                continue
            if config.animal_ids != None and seed not in config.animal_ids:
                continue

            name = f'sim_{config.n_neurons}_{config.n_regions}_{config.ga}_{config.sim_noise_std}_s{seed}'
            if config.sparsity != 1:
                name += f'_sparsity_{config.sparsity}'
            filename = os.path.join(SIM_DIR, f'{name}.npz')

            data = np.load(filename)
            if config.pc_dim is None:
                activity: np.ndarray = data['M']
                n = activity.shape[0]
                if config.portion_observable_neurons < 1:
                    n_observable = int(n * config.portion_observable_neurons)
                    activity = activity[: n_observable]
            else:
                activity: np.ndarray = data['PC'][: config.pc_dim]
            
            activity = self.downsample(activity)
            all_activities.append(activity)

            if config.pc_dim is None:
                all_unit_types.append(np.zeros((activity.shape[0], ), dtype=int))
            else:
                all_unit_types.append(np.arange(activity.shape[0], dtype=int))
        
        return all_activities, all_unit_types

def get_baseline_performance(config: DatasetConfig, phase='train'):
    """
    Get the baseline performance for the dataset

    :param config: the config object
    :param phase: 'train', 'val' or 'test'
    :return: a dict containing the baseline performance, contains the following keys:
    'pred_num': 
        list of arrays, each has shape [d], number of trials for each channel of each session
    'copy_mse', 'copy_mae': 
        list of arrays, each has shape [d], sum of mse/mae of the copy baseline
    'chance_mse', 'chance_mae': 
        list of arrays, each has shape [d], sum of mse/mae of the chance baseline
    'session_chance_mse', 'session_chance_mae': 
        list of floats, mse/mae for each session
    'session_copy_mse', 'session_copy_mae': 
        list of floats, mse/mae for each session
    'avg_copy_mse', 'avg_copy_mae': 
        float, average mse/mae across all sessions
    'avg_chance_mse', 'avg_chance_mae': 
        float, average mse/mae across all sessions
    'mean_copy_mse', 'mean_copy_mae': 
        array of shape [d], mean mse/mae for each PC
    'mean_chance_mse', 'mean_chance_mae': 
        array of shape [d], mean mse/mae for each PC
    """

    if phase == 'train':
        phase = 'val'
        print("warning: using val set to calculate baseline performance for training set to save compute")

    config.show_chance = True
    if config.dataset == 'zebrafish':
        dataset = Zebrafish(config, phase=phase)
    elif config.dataset == 'simulation':
        dataset = Simulation(config, phase=phase)
    elif config.dataset == 'zebrafishstim':
        dataset = StimZebrafish(config, phase=phase)
    elif config.dataset == 'zebrafishahrens':
        dataset = ZebrafishAhrens(config, phase=phase)
    elif config.dataset == 'zebrafishjain':
        dataset = ZebrafishJain(config, phase=phase)
    elif config.dataset == 'celegans':
        dataset = Celegans(config, phase=phase)
    elif config.dataset == 'celegansflavell':
        dataset = CelegansFlavell(config, phase=phase)
    elif config.dataset == 'mice':
        dataset = Mice(config, phase=phase)
    else:
        raise NotImplementedError(config.dataset)

    return dataset.baseline