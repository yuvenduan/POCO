import torch
import numpy as np
import os.path as osp
import os
import datasets

from configs.configs import BaseConfig, NeuralPredictionConfig
from datasets.dataloader import DatasetIters, init_single_dataset
from models.model_utils import model_init
from utils.config_utils import load_config
from utils.data_utils import get_stim_exp_names
from configs.config_global import STIM_PROCESSED_DIR, DEVICE
from matplotlib import pyplot as plt

all_regions = [
    'in_l_LHb', 'in_l_MHb', 'in_l_cerebellum', 'in_l_di', 'in_l_dthal', 'in_l_hind', 'in_l_meso', 'in_l_preoptic', 'in_l_raphe', 'in_l_tectum', 'in_l_tel', 'in_l_vthal', 
    'in_r_LHb', 'in_r_MHb', 'in_r_cerebellum', 'in_r_di', 'in_r_dthal', 'in_r_hind', 'in_r_meso', 'in_r_preoptic', 'in_r_raphe', 'in_r_tectum', 'in_r_tel', 'in_r_vthal'
]

@torch.no_grad()
def get_diffs(config: NeuralPredictionConfig, net, neural_data, stim_magnitude, side, stim_time_step_dict, region='Hb', save_dir=None):
    """
    Get the difference in neural activity between control and stimulated conditions
    :param stim_magnitude: the magnitude of the stimulation
    :param side: 'l' or 'r'
    """

    stim_region = f'in_{side}_{region}'
                
    if stim_region not in all_regions:
        raise ValueError(f'{stim_region} not in all_regions')
    region_idx = all_regions.index(stim_region)
    
    exp_names = get_stim_exp_names()
    avg_diffs = []

    for i_exp, exp_name in enumerate(exp_names['control']):
        stim_time_steps = stim_time_step_dict[exp_name][side + 'Hb']
        data = neural_data[i_exp].transpose(1, 0) # T * N

        print(f'exp {i_exp}, indices: {stim_time_steps}')

        control_preds = []
        stim_preds = []

        for t in stim_time_steps:
            before = config.seq_length - config.pred_length
            if t - before < 0 or t + config.pred_length >= data.shape[0]:
                continue

            original = data[t - before: t]
            suppressed = data[t - before: t].copy()
            suppressed[before - 2: before, region_idx] += stim_magnitude

            original = torch.tensor(original).unsqueeze(1).float().to(DEVICE)
            suppressed = torch.tensor(suppressed).unsqueeze(1).float().to(DEVICE)

            # print(f'original shape: {original.shape}, suppressed shape: {suppressed.shape}')

            original_input = [torch.zeros(original.size(0), 0, original.size(2)).to(DEVICE)] * len(exp_names['control'])
            suppressed_input = [torch.zeros(suppressed.size(0), 0, suppressed.size(2)).to(DEVICE)] * len(exp_names['control'])
            original_input[i_exp] = original
            suppressed_input[i_exp] = suppressed

            original_pred = net(original_input, pred_step=config.pred_length)[i_exp]
            suppressed_pred = net(suppressed_input, pred_step=config.pred_length)[i_exp]

            control_preds.append(original_pred)
            stim_preds.append(suppressed_pred)

        control_preds = torch.cat(control_preds, dim=1).mean(dim=1) # T * B * N -> T * N
        stim_preds = torch.cat(stim_preds, dim=1).mean(dim=1)

        diff = stim_preds - control_preds
        avg_diffs.append(diff.cpu().numpy())

    avg_diffs = np.stack(avg_diffs, axis=-1) # T * n_regions(N) * n_animals

    for idx, region in enumerate(all_regions):
        region_diffs = avg_diffs[:, idx]
        # plot n_animal traces, but also show accented average trace
        plt.figure()
        for i in range(region_diffs.shape[1]):
            plt.plot(region_diffs[:, i], color='b', alpha=0.2, linewidth=1)
        plt.plot(region_diffs.mean(axis=1), color='b', linewidth=1)
        plt.title(f'{region} diff')
        plt.savefig(osp.join(save_dir, f'{region}_diff.png'))

    return avg_diffs

@torch.no_grad()
def virtual_stimulation(config: NeuralPredictionConfig):
    # conduct virtual optogenetic stimulation on the trained model, as in Fig 6L, Aeron et al., 2019 Cell

    config.patch_length = 100000
    config.test_split = 0
    config.val_split = 0
    data_set = datasets.StimZebrafish(config, phase='train')
    neural_data = data_set.neural_data
    datum_size = data_set.datum_size
    save_dir = osp.join(config.save_path, 'stim')

    print(f'neural_data shape: {[data.shape for data in neural_data]}')

    stim_regions = ['forebrain', 'lHb', 'rHb']
    net = model_init(config, datum_size)
    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth'), weights_only=True))

    assert config.downsample_ratio == 1
    assert config.brain_regions == 'average'

    stim_time_step_dict = {}
    exp_names = get_stim_exp_names()
    for i_exp, exp_name in enumerate(exp_names['control']):
        filename = osp.join(STIM_PROCESSED_DIR, exp_name + '.npz')
        f = np.load(filename)
        stim_time_step_dict[exp_name] = {}
        for region in stim_regions:
            stim_time_step_dict[exp_name][region] = f[f'stim_ndx_{region}']

    for stim_magnitude in [0.2, 1]:
        for region in ['LHb']:
            sub_save_dir = osp.join(save_dir, f'l_{region}_stim={str(stim_magnitude)}')
            os.makedirs(sub_save_dir, exist_ok=True)
            l_diffs = get_diffs(config, net, neural_data, stim_magnitude, 'l', stim_time_step_dict, region, sub_save_dir)

            sub_save_dir = osp.join(save_dir, f'r_{region}_stim={str(stim_magnitude)}')
            os.makedirs(sub_save_dir, exist_ok=True)
            r_diffs = get_diffs(config, net, neural_data, stim_magnitude, 'r', stim_time_step_dict, region, sub_save_dir)