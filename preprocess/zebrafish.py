"""
Change data to numpy format, compute delta F / F
"""

import h5py
import os
import numpy as np
import scipy.io as sio
import analysis.plots as plots

from configs.config_global import \
    ZEBRAFISH_RAW_DIR, EXP_TYPES, ZEBRAFISH_PROCESSED_DIR, RAW_DATA_SUFFIX, \
    ZEBRAFISH_STIM_RAW_DIR, ZEBRAFISH_STIM_PROCESSED_DIR, ZEBRAFISH_AHRENS_RAW_DIR, ZEBRAFISH_AHRENS_PROCESSED_DIR, \
    ZEBRAFISH_JAIN_RAW_DIR, ZEBRAFISH_JAIN_PROCESSED_DIR
from utils.data_utils import get_exp_names, get_stim_exp_names
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from .utils import bandpass, run_pca, process_data_matrix

def process_zebrafish_activity():
    exp_names = get_exp_names()
    os.makedirs(ZEBRAFISH_PROCESSED_DIR, exist_ok=True)
    normalize_mode = 'zscore'

    brain_areas = [ 
        'in_l_LHb', 'in_l_MHb', 'in_l_ctel', 'in_l_dthal', 'in_l_gc', 'in_l_raphe', 'in_l_tel', 'in_l_vent', 'in_l_vthal',
        'in_r_LHb', 'in_r_MHb', 'in_r_ctel', 'in_r_dthal', 'in_r_gc', 'in_r_raphe', 'in_r_tel', 'in_r_vent', 'in_r_vthal'
    ]

    curves = []

    for exp_type in EXP_TYPES:
        for exp_name in exp_names[exp_type]:

            filename = os.path.join(ZEBRAFISH_RAW_DIR, exp_name + RAW_DATA_SUFFIX)
            out_filename = os.path.join(ZEBRAFISH_PROCESSED_DIR, exp_name)

            with h5py.File(filename, "r") as f:

                data_dict = {}
                for key in f.keys():
                    data_dict[key] = np.array(f[key])

                # print(data_dict['M'].shape, data_dict['frame_start_times_per_plane'].shape, data_dict['frame_end_times_per_plane'])

                # visualize baseline activity, 5th percentile
                baseline = np.percentile(data_dict['M'], 5, axis=0)
                curves.append(baseline)
                continue

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

    # plot mean +- std of baseline activity, take the min of length
    min_len = min([len(curve) for curve in curves])
    curves = [curve[:min_len] for curve in curves]
    curves = np.stack(curves, axis=1)
    
    plots.error_plot(
        np.arange(min_len),
        [curves],
        errormode='std',
        mode='errorshade',
        legend=False,
        x_label='Time step',
        y_label='5th Percentile F',
        fig_name='baseline_activity',
        save_dir='preprocess',
    )

def process_zebrafish_stim_activity():
    exp_names = get_stim_exp_names()

    os.makedirs(ZEBRAFISH_STIM_PROCESSED_DIR, exist_ok=True)

    brain_areas = [
        'in_l_LHb', 'in_l_MHb', 'in_l_cerebellum', 'in_l_di', 'in_l_dthal', 'in_l_hind', 'in_l_meso', 'in_l_raphe', 'in_l_tectum', 'in_l_tel', 'in_l_vthal', 
        'in_r_LHb', 'in_r_MHb', 'in_r_cerebellum', 'in_r_di', 'in_r_dthal', 'in_r_hind', 'in_r_meso', 'in_r_raphe', 'in_r_tectum', 'in_r_tel', 'in_r_vthal'
    ]

    stim_regions = ['forebrain', 'lHb', 'rHb']

    for exp_type in ['control', 'stim']:
        for exp_name in exp_names[exp_type]:

            filename = os.path.join(ZEBRAFISH_STIM_RAW_DIR, f'hbstim_{exp_name}_cnmf_.h5')
            out_filename = os.path.join(ZEBRAFISH_STIM_PROCESSED_DIR, exp_name)

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

def process_zebrafish_ahrens_activity(filter_mode='none'):

    processed_dir = ZEBRAFISH_AHRENS_PROCESSED_DIR
    if filter_mode != 'none':
        processed_dir = os.path.join(processed_dir + '_' + filter_mode)

    os.makedirs(processed_dir, exist_ok=True)

    count = 0
    for subject_id in range(1, 19):
        sub_dir = os.path.join(ZEBRAFISH_AHRENS_RAW_DIR, f'subject_{subject_id}')
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
        print(data_dict.keys())
        
        data = sio.loadmat(mat_filename, squeeze_me=True, struct_as_record=False)['data']

        p = data.periods
        A = data.Behavior_full_motorseed
        B = data.BehaviorAvr_motorseed

        activity = data_dict['CellResp'].T
        return_dict = process_data_matrix(
            activity,
            f'preprocess/{filter_mode}/zebrafish_ahrens',
            pc_dim=2048,
            exp_name=f'subject_{subject_id}',
            filter_mode=filter_mode
        )
        data_dict = {}
        data_dict.update(return_dict)

        out_filename = os.path.join(processed_dir, f'{count}.npz')
        count += 1
        np.savez(out_filename, **data_dict)

def preprocess_zebrafish_jain(filter_mode='none'):
    import tensorstore as ts

    ds_traces = ts.open({
        'open': True,
        'driver': 'zarr3',
        'kvstore': 'gs://zapbench-release/volumes/20240930/traces'
    }).result()

    traces = np.array(ds_traces)
    traces = traces.T

    return_dict = process_data_matrix(
        traces,
        f'preprocess/{filter_mode}/zebrafish_jain',
        pc_dim=2048,
        exp_name='jain',
        filter_mode=filter_mode
    )
    data_dict = {}
    data_dict.update(return_dict)

    os.makedirs(ZEBRAFISH_JAIN_PROCESSED_DIR, exist_ok=True)
    out_filename = os.path.join(ZEBRAFISH_JAIN_PROCESSED_DIR, f'0.npz')
    np.savez(out_filename, **data_dict)