import torch
import numpy as np
import os.path as osp
import os
import datasets

from tqdm import tqdm
from configs.configs import BaseConfig, NeuralPredictionConfig, DatasetConfig
from datasets.dataloader import DatasetIters, init_single_dataset
from models.model_utils import model_init
from utils.config_utils import load_config
from utils.data_utils import get_stim_exp_names
from configs.config_global import ZEBRAFISH_STIM_PROCESSED_DIR, DEVICE, FIG_DIR, EXP_TYPES
from matplotlib import pyplot as plt

@torch.no_grad()
def get_diffs(config: NeuralPredictionConfig, net, neural_data, stim_magnitude, side, stim_time_step_dict, region='Hb', save_dir=None):
    """
    Get the difference in neural activity between control and stimulated conditions
    :param stim_magnitude: the magnitude of the stimulation
    :param side: 'l' or 'r'
    """

    all_regions = [
        'in_l_LHb', 'in_l_MHb', 'in_l_cerebellum', 'in_l_di', 'in_l_dthal', 'in_l_hind', 'in_l_meso', 'in_l_preoptic', 'in_l_raphe', 'in_l_tectum', 'in_l_tel', 'in_l_vthal', 
        'in_r_LHb', 'in_r_MHb', 'in_r_cerebellum', 'in_r_di', 'in_r_dthal', 'in_r_hind', 'in_r_meso', 'in_r_preoptic', 'in_r_raphe', 'in_r_tectum', 'in_r_tel', 'in_r_vthal'
    ]

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
            suppressed[before - 1: before, region_idx] += stim_magnitude

            original = torch.tensor(original).unsqueeze(1).float().to(DEVICE)
            suppressed = torch.tensor(suppressed).unsqueeze(1).float().to(DEVICE)

            # print(f'original shape: {original.shape}, suppressed shape: {suppressed.shape}')

            original_input = [torch.zeros(original.size(0), 0, original.size(2)).to(DEVICE)] * len(exp_names['control'])
            suppressed_input = [torch.zeros(suppressed.size(0), 0, suppressed.size(2)).to(DEVICE)] * len(exp_names['control'])
            original_input[i_exp] = original
            suppressed_input[i_exp] = suppressed

            original_pred = net(original_input)[i_exp]
            suppressed_pred = net(suppressed_input)[i_exp]

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

    dataset_config: DatasetConfig = config.dataset_config[config.dataset_label[0]]
    dataset_config.patch_length = 100000
    dataset_config.test_split = 0
    dataset_config.val_split = 0
    data_set = datasets.StimZebrafish(dataset_config, phase='train')
    neural_data = data_set.neural_data
    datum_size = [data_set.input_size]
    unit_types = [data_set.unit_types]
    save_dir = osp.join(config.save_path, 'stim')

    print(f'neural_data shape: {[data.shape for data in neural_data]}')

    stim_regions = ['forebrain', 'lHb', 'rHb']
    net = model_init(config, datum_size, unit_types)
    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth'), weights_only=True))

    assert config.downsample_ratio == 1
    assert dataset_config.brain_regions == 'average'

    stim_time_step_dict = {}
    exp_names = get_stim_exp_names()
    for i_exp, exp_name in enumerate(exp_names['control']):
        filename = osp.join(ZEBRAFISH_STIM_PROCESSED_DIR, exp_name + '.npz')
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

def get_interaction_map(config: NeuralPredictionConfig, method='stim', interval=1, stim_magnitude=0.2, segments=5):
    """
    Get the interaction map for the zebrafish dataset
    :param config: NeuralPredictionConfig
    :param method: Method of calculating the interaction map, 'jac' for Jacobian, 'stim' for virtual stimulation
    :param stim_magnitude: Magnitude of stimulation for virtual stimulation method.
    """
    dataset_config: DatasetConfig = config.dataset_config[config.dataset_label[0]]
    dataset_config.patch_length = 100000
    dataset_config.test_split = 0
    dataset_config.val_split = 0
    dataset = datasets.Zebrafish(dataset_config, phase='train')
    regions = dataset.all_regions

    context_length = config.seq_length - config.pred_length

    neural_data = dataset.neural_data
    datum_size = [dataset.input_size]
    unit_types = [dataset.unit_types]

    net = model_init(config, datum_size, unit_types)
    net.load_state_dict(torch.load(osp.join(config.save_path, 'net_best.pth'), weights_only=True))

    all_maps = []
    all_segmented_maps = []

    for i_subject in tqdm(range(len(neural_data))):

        neural = neural_data[i_subject] # N * T
        neural_std = np.std(neural, axis=1)

        if neural_std.min() < 1e-6:
            print(f"channel with zero std, skip subject {i_subject}")
            continue 

        interaction_map = np.zeros((neural.shape[0], neural.shape[0], (neural.shape[1] - context_length - 1) // interval + 1)) # N * N * T
        
        for t in range(0, neural.shape[1] - context_length, interval):

            if method == 'stim':
                with torch.no_grad():
                    for source in range(neural.shape[0]):

                        original = neural[:, t: t + context_length].copy()

                        activated = original.copy()
                        suppressed = original.copy()

                        activated[source, :] += stim_magnitude * neural_std[source]
                        suppressed[source, :] -= stim_magnitude * neural_std[source]
                        all_trials = np.stack([original, activated, suppressed], axis=0) # 3 * N * T
                        all_trials = torch.tensor(all_trials).float().to(DEVICE).permute(2, 0, 1) # T * 3 * N

                        input = [torch.zeros(original.shape[1], 0, original.shape[0]).to(DEVICE)] * len(neural_data)
                        input[i_subject] = all_trials
                        
                        preds = net(input)[i_subject]
                        preds = preds.cpu().numpy() # T * 3 * N

                        diff = (preds[:, 1, :] - preds[:, 2, :]) / neural_std # T * N
                        diff = diff.mean(axis=0)

                        interaction_map[:, source, t // interval] = diff
            elif method == 'jac':
                raise NotImplementedError("Jacobian method is not implemented yet.")

        mean_interaction_map = interaction_map.mean(axis=2) # N * N
        all_maps.append(mean_interaction_map)

        # segment the interaction map into time segments
        segmented_maps = []
        segment_length = interaction_map.shape[2] // segments
        for i_segment in range(segments):
            start = i_segment * segment_length
            if i_segment == segments - 1:
                end = interaction_map.shape[2]
            else:
                end = (i_segment + 1) * segment_length
            segment_map = interaction_map[:, :, start:end].mean(axis=2)
            segmented_maps.append(segment_map)
        segmented_maps = np.stack(segmented_maps, axis=0) # segments * N * N
        all_segmented_maps.append(segmented_maps)

    return all_maps, all_segmented_maps, regions

def plot_interaction_map(interaction_map, names, save_dir, save_name, zero_out_diag=True):
    """
    Plot the interaction map(s).
    
    :param interaction_map: N x N or K x N x N array
    :param names: List of neuron names
    :param save_dir: Directory to save the plot
    :param save_name: Name of the plot file
    """
    os.makedirs(save_dir, exist_ok=True)
    interaction_map = np.asarray(interaction_map)

    # Handle input shape
    if interaction_map.ndim == 2:   # N x N
        interaction_map = interaction_map[None, ...]  # Make it 1 x N x N
    elif interaction_map.ndim != 3:
        raise ValueError(f"interaction_map must be N * N or K * N * N but got shape {interaction_map.shape}")

    k, n, m = interaction_map.shape
    if n != m:
        raise ValueError("Interaction map must be square (N * N)")
    
    interaction_map = interaction_map.copy()
    if zero_out_diag:
        for i in range(k):
            np.fill_diagonal(interaction_map[i], 0)

    # Get consistent color scaling across all maps
    vmax = np.max(np.abs(interaction_map))
    vmin = -vmax

    fig, axes = plt.subplots(1, k, figsize=(5 * k, 5), squeeze=False, constrained_layout=True)
    cmap = "viridis"

    for i in range(k):
        ax = axes[0, i]
        plot_map = interaction_map[i]
        im = ax.imshow(plot_map, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names, rotation=45)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Interaction Strength")

    plt.savefig(osp.join(save_dir, save_name))
    plt.close()

def analyze_interaction_maps(configs: dict, 
                             modes=[''],
                             models=None,
                             method='stim', 
                             interval=32, 
                             stim_magnitude=0.2,
                             save_dir='interaction_maps'
                            ):
    """
    Analyze interaction maps for multiple models

    :param configs: Dictionary of configurations to analyze. Keys are seeds and values are lists of NeuralPredictionConfig.
        Lists should be of length len(modes) * len(models).
    """

    save_dir = osp.join(FIG_DIR, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for i_mode in range(len(modes)):
        for i_model in range(len(models)):
            mode = modes[i_mode]
            model = models[i_model]

            sub_save_dir = osp.join(save_dir, mode, model)
            os.makedirs(sub_save_dir, exist_ok=True)
            print(f'Saving interaction maps to {sub_save_dir}', flush=True)

            result_list = []

            for seed in configs.keys():
                config: NeuralPredictionConfig = configs[seed][i_mode * len(models) + i_model]
                print(f'Analyzing interaction map for mode: {mode}, model: {model}, seed: {seed}', flush=True)

                all_maps, all_segmented_maps, regions = get_interaction_map(config, method=method, interval=interval, stim_magnitude=stim_magnitude)
                result_list.append((all_maps, all_segmented_maps))

            n_subjects = len(result_list[0][0])
            all_maps = [np.stack([r[0][j] for r in result_list]) for j in range(n_subjects)] # n_subjects * n_seeds * N * N
            all_segmented_maps = [np.stack([r[1][j] for r in result_list]) for j in range(n_subjects)] # n_subjects * n_seeds * segments * N * N

            if config.dataset_label[0] == 'zebrafish_avg':
                all_maps = [np.mean(maps, axis=0) for maps in all_maps] # n_subjects * N * N
                all_segmented_maps = [np.mean(maps, axis=0) for maps in all_segmented_maps] # n_subjects * segments * N * N

                # pool 'in_l_LHb' and 'in_r_LHb' into 'LHb', same for other regions
                pooled_maps = []
                pooled_segmented_maps = []
                pooled_regions = [region.split('_')[2] for region in regions if region.startswith('in_l_')]

                for maps in all_maps:
                    pooled_map = np.zeros((len(pooled_regions), len(pooled_regions)))
                    for i, region_i in enumerate(pooled_regions):
                        for j, region_j in enumerate(pooled_regions):
                            left_i = regions.index(f'in_l_{region_i}')
                            right_i = regions.index(f'in_r_{region_i}')
                            left_j = regions.index(f'in_l_{region_j}')
                            right_j = regions.index(f'in_r_{region_j}')
                            pooled_map[i, j] = (maps[left_i, left_j] + maps[left_i, right_j] + maps[right_i, left_j] + maps[right_i, right_j]) / 4
                    pooled_maps.append(pooled_map)

                for maps in all_segmented_maps:
                    pooled_segmented_map = np.zeros((maps.shape[0], len(pooled_regions), len(pooled_regions)))
                    for k in range(maps.shape[0]):
                        for i, region_i in enumerate(pooled_regions):
                            for j, region_j in enumerate(pooled_regions):
                                left_i = regions.index(f'in_l_{region_i}')
                                right_i = regions.index(f'in_r_{region_i}')
                                left_j = regions.index(f'in_l_{region_j}')
                                right_j = regions.index(f'in_r_{region_j}')
                                pooled_segmented_map[k, i, j] = (maps[k, left_i, left_j] + maps[k, left_i, right_j] + maps[k, right_i, left_j] + maps[k, right_i, right_j]) / 4
                    pooled_segmented_maps.append(pooled_segmented_map)

                all_maps = pooled_maps
                all_segmented_maps = pooled_segmented_maps

                # plot 0: map for every subject
                for i_subject in range(n_subjects):
                    plot_interaction_map(all_maps[i_subject], pooled_regions, sub_save_dir, f'subject_{i_subject}_map.png')

                # plot 1: compare cohorts
                assert n_subjects == 17
                cohort_indices_dict = {
                    'control': list(range(0, 5)),
                    'shocked': list(range(5, 10)),
                    'reshocked': list(range(10, 13)),
                    'ketamine': list(range(13, 17))
                }

                for i_exp, exp_type in enumerate(EXP_TYPES):
                    cohort_indices = cohort_indices_dict[exp_type]
                    cohort_avg = np.mean([all_maps[i] for i in cohort_indices], axis=0)
                    plot_interaction_map(cohort_avg, pooled_regions, sub_save_dir, f'cohort_{exp_type}_map.png')

                # plot 2: all maps over time
                for i_exp, exp_type in enumerate(EXP_TYPES):
                    cohort_indices = cohort_indices_dict[exp_type]
                    cohort_avg_segmented = np.mean([all_segmented_maps[i] for i in cohort_indices], axis=0)
                    plot_interaction_map(cohort_avg_segmented, pooled_regions, sub_save_dir, f'cohort_{exp_type}_segmented_map.png')
