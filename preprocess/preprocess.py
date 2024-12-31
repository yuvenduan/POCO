"""
Change data to numpy format, compute delta F / F
"""

import h5py
import os
import numpy as np
import scipy.io as sio
import analysis.plots as plots

from configs.config_global import \
    RAW_DIR, EXP_TYPES, PROCESSED_DIR, RAW_DATA_SUFFIX, \
    VISUAL_PROCESSED_DIR, VISUAL_RAW_DIR, STIM_RAW_DIR, STIM_PROCESSED_DIR, FIG_DIR
from utils.data_utils import get_exp_names, get_subject_ids, get_stim_exp_names
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run_pca(data, exp_name='', n_components=2048, fig_dir='preprocess'):
    """
    data: n_neurons * T
    return: n_components * T
    """
    n_components = min(n_components, data.shape[1], data.shape[0])
    pca = PCA(n_components=n_components, svd_solver='full')
    act = pca.fit_transform(data.T)
    print('EV 10:', pca.explained_variance_ratio_[:10])
    cumulated = np.cumsum(pca.explained_variance_ratio_)
    plots.error_plot(
        np.arange(1, len(cumulated) + 1),
        [cumulated],
        x_label='Number of PCs', y_label='Explained variance',
        save_dir=fig_dir,
        fig_name=f'{exp_name}_explained_variance',
        errormode='none',
        mode='errorshade',
        yticks=[0, 1]
    )

    print("Reduced Data Shape: ", act.shape)
    print(f"Cumulative Explained Variance for {n_components} PCs", cumulated[-1])
    print('')
    return act.T

def get_clustered_data(data, n_clusters=500, seed=0):
    """
    Do KMeans clustering on correlation matrix
    data: n_neurons * T
    return: n_clusters * T
    """
    corr_matrix = np.corrcoef(data)
    print("Correlation Matrix Shape: ", corr_matrix.shape)
    corr_matrix[np.isnan(corr_matrix)] = 0
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(corr_matrix)
    labels = kmeans.labels_
    
    # compute the mean of time traces in each cluster
    clustered_data = np.zeros((n_clusters, data.shape[1]))
    for i in range(n_clusters):
        clustered_data[i] = data[labels == i].mean(axis=0)
    return clustered_data

def plot_delta_F(data, exp_name='', fig_dir='preprocess', suffix=''):
    """
    Plot the difference between adjacent frames
    data: n_neurons * T
    """
    # plot distribution of delta F
    plots.distribution_plot(
        [[(data[:, 1: ] - data[:, : -1]).reshape(-1)]],
        ['All ROIs', ],
        left=-4.95, right=5, 
        interval=0.1,
        x_label='Delta F', y_label='Frequency',
        plot_dir=fig_dir,
        plot_name=f'{exp_name}_delta_F{suffix}',
        errormode='none',
        mode='errorshade',
        legend=False,
    )

def process_data_matrix(
    data, fig_dir, 
    roi_indices=None, 
    divide_baseline=False, 
    normalize_mode='zscore', 
    n_clusers=[], 
    plot_max_min=False, 
    exp_name='',
    pc_dim=2048
):
    """
    Process the data matrix, also plot the distribution of min F and max F, and sample traces of first 10 PCs / 10 random neurons

    :param data: n_neurons * T
    :param roi_indices: n_neurons or None, represent the valid ROIs to use
    :param divide_baseline: whether to calculate delta F / F
    :param normalize_mode: 'none', 'zscore', 'max'
    :param n_clusters: list of number of clusters to use
    return: dict with keys 'valid_indices', 'M', 'PC', 'FC_{n}', where n is in n_clusters
    """
    data_dict = {}
    n, T = data.shape
    sub_fig_dir = os.path.join(FIG_DIR, fig_dir, exp_name)

    if roi_indices is None:
        roi_indices = np.ones(n, dtype=bool)
    
    if divide_baseline:
        baseline = np.median(data, axis=1, keepdims=True)
        roi_indices &= baseline.reshape(-1) > 1
        data = (data - baseline) / (baseline + 1e-6)

    if plot_max_min:
        # min F
        plots.distribution_plot(
            [[data_dict['M'].min(axis=1)], [data_dict['M'].min(axis=1)[roi_indices]]],
            ['All ROIs', 'Filtered ROIs'],
            left=-100, right=100, 
            interval=10,
            x_label='Min F', y_label='Frequency',
            plot_dir=sub_fig_dir,
            plot_name=f'min_F',
            errormode='none'
        )

        # max F
        plots.distribution_plot(
            [[data_dict['M'].max(axis=1)], [data_dict['M'].max(axis=1)[roi_indices]]],
            ['All ROIs', 'Filtered ROIs'],
            left=-50, right=2000, 
            interval=10,
            x_label='Max F', y_label='Frequency',
            plot_dir=sub_fig_dir,
            plot_name=f'max_F',
            errormode='none'
        )

    if normalize_mode == 'zscore':
        mu, std = np.mean(data, axis=1, keepdims=True), np.std(data, axis=1, keepdims=True)
        normalized = (data - mu) / (std + 1e-6)
    elif normalize_mode == 'max':
        normalized = data / (np.max(np.abs(data), axis=1, keepdims=True) + 1e-6)
    else:
        normalized = data

    print(f"Data shape: {data.shape}, Valid indices: {roi_indices.sum()}")
    data_dict['valid_indices'] = roi_indices
    data_dict['M'] = normalized
    
    normalized = normalized[roi_indices]
    plot_delta_F(normalized, fig_dir=sub_fig_dir, exp_name=exp_name)
    data_dict['PC'] = run_pca(normalized, fig_dir=sub_fig_dir, exp_name=exp_name, n_components=pc_dim)
    plot_delta_F(data_dict['PC'], fig_dir=sub_fig_dir, suffix='_PC', exp_name=exp_name)

    # plot first 10 PCs
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 10))
    for i in range(10):
        ax = plt.subplot(10, 2, i * 2 + 1)
        ax.plot(data_dict['PC'][i])
        ax.title.set_text(f'PC {i}, full trace')
        # plot a random 64-step window
        ax = plt.subplot(10, 2, i * 2 + 2)
        t = np.random.randint(0, data_dict['PC'].shape[1] - 64)
        ax.plot(data_dict['PC'][i, t: t + 64])
        ax.title.set_text(f'PC {i}, 64-step window')
    plt.tight_layout()
    plt.savefig(os.path.join(sub_fig_dir, f'{exp_name}_first_10_PCs.pdf'))

    # plot 10 random neurons
    plt.figure(figsize=(7, 10))
    for i in range(10):
        idx = np.random.randint(0, normalized.shape[0])
        ax = plt.subplot(10, 2, i * 2 + 1)
        ax.plot(normalized[idx])
        ax.title.set_text(f'Neuron {idx}, full trace')
        # plot a random 64-step window
        ax = plt.subplot(10, 2, i * 2 + 2)
        t = np.random.randint(0, normalized.shape[1] - 64)
        ax.plot(normalized[idx, t: t + 64])
        ax.title.set_text(f'Neuron {idx}, 64-step window')
    plt.tight_layout()
    plt.savefig(os.path.join(sub_fig_dir, f'{exp_name}_10_random_neurons.pdf'))

    for n_cluster in n_clusers:
        data_dict[f'FC_{n_cluster}'] = get_clustered_data(normalized, n_cluster)
    return data_dict

def process_spontaneous_activity():
    exp_names = get_exp_names()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    normalize_mode = 'zscore'

    brain_areas = [ 
        'in_l_LHb', 'in_l_MHb', 'in_l_ctel', 'in_l_dthal', 'in_l_gc', 'in_l_raphe', 'in_l_tel', 'in_l_vent', 'in_l_vthal',
        'in_r_LHb', 'in_r_MHb', 'in_r_ctel', 'in_r_dthal', 'in_r_gc', 'in_r_raphe', 'in_r_tel', 'in_r_vent', 'in_r_vthal'
    ]

    for exp_type in EXP_TYPES:
        for exp_name in exp_names[exp_type]:

            filename = os.path.join(RAW_DIR, exp_name + RAW_DATA_SUFFIX)
            out_filename = os.path.join(PROCESSED_DIR, exp_name)

            with h5py.File(filename, "r") as f:
                print("Keys: %s" % f.keys())

                data_dict = {}
                for key in f.keys():
                    data_dict[key] = np.array(f[key])

                # print(data_dict['M'].shape, data_dict['frame_start_times_per_plane'].shape, data_dict['frame_end_times_per_plane'])

                # tail_movements = zip(*[data_dict[x] for x in ['tail_movement_start_times', 'tail_movement_end_times', 'tail_movement_is_escapeswim', 'tail_movement_is_forwardswim', 'tail_movement_is_turnswim', ]])
                # for i, (start, end, is_escape, is_forward, is_turn) in enumerate(tail_movements):
                #     print(f"Trial {i}: {start}, {end}, {is_escape}, {is_forward}, {is_turn}")

                roi_indices = np.zeros(data_dict['M'].shape[0], dtype=bool)
                for area in brain_areas:
                    assert area in data_dict, f"Area {area} not found in data"
                    roi_indices |= data_dict[area]

                return_dict = process_data_matrix(
                    data_dict['M'], 
                    fig_dir='preprocess/spontaneous_zebrafish',
                    roi_indices=None,
                    divide_baseline=True,
                    normalize_mode=normalize_mode,
                    n_clusers=[],
                    exp_name=exp_name
                )
                data_dict.update(return_dict)
                
                np.savez(out_filename, **data_dict)

def process_stim_activity():
    exp_names = get_stim_exp_names()
    os.makedirs(STIM_PROCESSED_DIR, exist_ok=True)

    brain_areas = [
        'in_l_LHb', 'in_l_MHb', 'in_l_cerebellum', 'in_l_di', 'in_l_dthal', 'in_l_hind', 'in_l_meso', 'in_l_raphe', 'in_l_tectum', 'in_l_tel', 'in_l_vthal', 
        'in_r_LHb', 'in_r_MHb', 'in_r_cerebellum', 'in_r_di', 'in_r_dthal', 'in_r_hind', 'in_r_meso', 'in_r_raphe', 'in_r_tectum', 'in_r_tel', 'in_r_vthal'
    ]

    stim_regions = ['forebrain', 'lHb', 'rHb']

    for exp_type in ['control', 'stim']:
        for exp_name in exp_names[exp_type]:

            filename = os.path.join(STIM_RAW_DIR, f'hbstim_{exp_name}_cnmf_.h5')
            out_filename = os.path.join(STIM_PROCESSED_DIR, exp_name)

            with h5py.File(filename, "r") as f:
                print("Keys: %s" % f.keys())

                data_dict = {}
                for key in f.keys():
                    data_dict[key] = np.array(f[key])

                print('Data shape', data_dict['M'].shape, 'Stimulation Time Shape', [data_dict[f'stim_ndx_{region}'].shape for region in stim_regions], flush=True)

                roi_indices = np.zeros(data_dict['M'].shape[0], dtype=bool)

                for area in brain_areas:
                    assert area in data_dict, f"Area {area} not found in data"
                    roi_indices |= data_dict[area]

                return_dict = process_data_matrix(
                    data_dict['M'], 
                    fig_dir='preprocess/stim_zebrafish',
                    roi_indices=None,
                    divide_baseline=True,
                    normalize_mode='zscore',
                    n_clusers=[],
                    exp_name=exp_name
                )
                data_dict.update(return_dict)
                np.savez(out_filename, **data_dict)

def process_visual_activity():
    os.makedirs(VISUAL_PROCESSED_DIR, exist_ok=True)

    for subject_id in range(1, 19):
        sub_dir = os.path.join(VISUAL_RAW_DIR, f'subject_{subject_id}')
        filename = os.path.join(sub_dir, 'TimeSeries.h5')
        mat_filename = os.path.join(sub_dir, 'data_full.mat')

        try:
            f = h5py.File(filename, "r")
        except:
            print(f"Subject {subject_id}: Cannot open TimeSeries.h5")
            continue

        data_dict = {}
        for key in f.keys():
            data_dict[key] = np.array(f[key])

        # print(data_dict['CellResp'].shape, data_dict['CellRespAvr'].shape, data_dict['CellRespZ'].std(axis=0))
        
        data = sio.loadmat(mat_filename, squeeze_me=True, struct_as_record=False)['data']

        p = data.periods
        A = data.Behavior_full_motorseed
        B = data.BehaviorAvr_motorseed
        print(p, A.shape, B.shape)

        activity = data_dict['CellResp'].T
        try:
            behavior = data.Behavior_full
            behavior_motorseed = data.Behavior_full_motorseed
            eye = data.Eye_full
            eye_motorseed = data.Eye_full_motorseed
        except:
            print(f"Subject {subject_id}: Behavior / Eye not found")
            continue
        stim = data.stim_full

        plot_delta_F(activity, exp_name=f'subject_{subject_id}', fig_dir='visual_preprocess')
        PC = run_pca(activity, exp_name=f'subject_{subject_id}', fig_dir='visual_preprocess')
        plot_delta_F(PC, exp_name=f'subject_{subject_id}', fig_dir='visual_preprocess', suffix='_PC')

        out_filename = os.path.join(VISUAL_PROCESSED_DIR, f'subject_{subject_id}')
        data_dict = {
            'M': activity,
            'PC': PC,
            'behavior': behavior,
            'behavior_motorseed': behavior_motorseed,
            'eye': eye,
            'eye_motorseed': eye_motorseed,
            'stim': stim,
        }
        np.savez(out_filename, **data_dict)