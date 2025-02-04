import numpy as np
import os.path as osp
import h5py
import os

from scipy import signal
from scipy.io import loadmat
from configs.config_global import MICE_RAW_DIR, MICE_BRAIN_AREAS, MICE_PROCESSED_DIR
from preprocess.preprocess import process_data_matrix
from preprocess.celegans import preprocess_data
from utils.data_utils import get_mice_sessions

def mice_preprocess():
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
            all_activity_df = preprocess_data(all_activity_df.T, 5.36)[1].T
            area_ids = np.concatenate(area_ids, axis=0)

            data_dict = process_data_matrix(all_activity_df, 'preprocess/mice', pc_dim=512, exp_name=f'{mouse}_{session}', normalize_mode='none')
            os.makedirs(MICE_PROCESSED_DIR, exist_ok=True)
            data_dict['area_ids'] = area_ids
            np.savez(osp.join(MICE_PROCESSED_DIR, f'{mouse}_{session}.npz'), **data_dict)