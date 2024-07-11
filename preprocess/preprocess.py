"""
Change data to numpy format, compute delta F / F
"""

import h5py
import os
import numpy as np
import scipy.io as sio
import analysis.plots as plots

from configs.config_global import RAW_DIR, EXP_TYPES, PROCESSED_DIR, RAW_DATA_SUFFIX, VISUAL_PROCESSED_DIR, VISUAL_RAW_DIR
from utils.data_utils import get_exp_names, get_subject_ids
from sklearn.decomposition import PCA

def run_pca(data, exp_name='', n_components=2048, fig_dir='preprocess'):
    """
    data: n_neurons * T
    return: n_components * T
    """
    n_components = min(n_components, data.shape[1])
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
    print(f"512-dim Explained variance: {cumulated[511]}")
    print('')
    return act.T

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

def process_spontaneous_activity():
    exp_names = get_exp_names()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

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

                plots.distribution_plot(
                    [[data_dict['M'].min(axis=1)], [data_dict['M'].min(axis=1)[roi_indices]]],
                    ['All ROIs', 'Filtered ROIs'],
                    left=-100, right=100, 
                    interval=10,
                    x_label='Min F', y_label='Frequency',
                    plot_dir='preprocess',
                    plot_name=f'{exp_name}_min_F',
                    errormode='none'
                )

                # max F
                plots.distribution_plot(
                    [[data_dict['M'].max(axis=1)], [data_dict['M'].max(axis=1)[roi_indices]]],
                    ['All ROIs', 'Filtered ROIs'],
                    left=-50, right=2000, 
                    interval=10,
                    x_label='Max F', y_label='Frequency',
                    plot_dir='preprocess',
                    plot_name=f'{exp_name}_max_F',
                    errormode='none'
                )

                valid_indices = np.ones(data_dict['M'].shape[0], dtype=bool)
                valid_indices &= data_dict['M'].min(axis=1) > 10
                print(f"Indices: {data_dict['M'].shape}, Valid indices: {valid_indices.sum()}")

                # delta F / F, then normalize by max abs value, the resuling value is in [-1, 1]
                data_dict['valid_indices'] = valid_indices
                baseline = np.median(data_dict['M'], axis=1, keepdims=True)
                data_dict['M'] = (data_dict['M'] - baseline) / baseline
                # data_dict['M'] = data_dict['M'] / (np.max(np.abs(data_dict['M']), axis=1, keepdims=True) + 1e-4)
                # assert np.all(np.abs(data_dict['M']) <= 1)

                normalized = data_dict['M'] / (np.max(np.abs(data_dict['M']), axis=1, keepdims=True) + 1e-4)
                data_dict['M'] = normalized

                normalized = normalized[valid_indices]
                plot_delta_F(normalized, exp_name=exp_name)
                data_dict['PC'] = run_pca(normalized, exp_name=exp_name)
                plot_delta_F(data_dict['PC'], exp_name=exp_name, suffix='_PC')
                
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

        