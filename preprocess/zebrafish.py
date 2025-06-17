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
    ZEBRAFISH_AHRENS_RAW_DIR, ZEBRAFISH_AHRENS_PROCESSED_DIR
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