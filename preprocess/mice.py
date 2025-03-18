import numpy as np
import os.path as osp
import h5py
import os

from scipy import signal
from scipy.io import loadmat
from configs.config_global import MICE_RAW_DIR, MICE_BRAIN_AREAS, MICE_PROCESSED_DIR
from .utils import process_data_matrix
from utils.data_utils import get_mice_sessions

def mice_preprocess(filter_mode='none'):
    processed_dir = MICE_PROCESSED_DIR
    if filter_mode != 'none':
        processed_dir = osp.join(processed_dir + '_' + filter_mode)
    os.makedirs(processed_dir, exist_ok=True)

    all_sessions = get_mice_sessions()
    for mouse, sessions in all_sessions.items():
        for session in sessions:
            data_dir = osp.join(MICE_RAW_DIR, mouse, session)

            all_activity = []
            all_activity_df = []
            area_ids = []
            
            for area_id, area in enumerate(MICE_BRAIN_AREAS):
                file_name = osp.join(data_dir, f'{mouse}_{area}_{session}.mat')
                file_name_df = os.path.join(data_dir, f'{mouse}_{area}_df_{session}.mat')

                if not osp.exists(file_name) or not osp.exists(file_name_df):
                    continue

                data = loadmat(file_name)[area]
                data_df = loadmat(file_name_df)[area + '_df']

                area_ids.append(np.ones(data.shape[0], dtype=int) * area_id)
                all_activity.append(data)
                all_activity_df.append(data_df)

            all_activity = np.concatenate(all_activity, axis=0)
            all_activity_df = np.concatenate(all_activity_df, axis=0)
            area_ids = np.concatenate(area_ids, axis=0)

            data_dict = process_data_matrix(
                all_activity_df, f'preprocess/{filter_mode}/mice', 
                pc_dim=512, exp_name=f'{mouse}_{session}', 
                normalize_mode='zscore',
                filter_mode=filter_mode
            )

            data_dict['area_ids'] = area_ids
            np.savez(osp.join(processed_dir, f'{mouse}_{session}.npz'), **data_dict)