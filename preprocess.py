"""
Change data to numpy format, compute delta F / F
"""

import h5py
import os
import numpy as np
import analysis.plots as plots

from configs.config_global import RAW_DIR, EXP_TYPES, PROCESSED_DIR, RAW_DATA_SUFFIX
from utils.data_utils import get_exp_names
from sklearn.decomposition import PCA

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

            tail_movements = zip(*[data_dict[x] for x in ['tail_movement_start_times', 'tail_movement_end_times', 'tail_movement_is_escapeswim', 'tail_movement_is_forwardswim', 'tail_movement_is_turnswim', ]])
            #for i, (start, end, is_escape, is_forward, is_turn) in enumerate(tail_movements):
            #    print(f"Trial {i}: {start}, {end}, {is_escape}, {is_forward}, {is_turn}")

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
            valid_indices &= data_dict['M'].min(axis=1) > 1
            print(f"Indices: {data_dict['M'].shape}, Valid indices: {valid_indices.sum()}")

            # delta F / F, then normalize by max abs value, the resuling value is in [-1, 1]
            data_dict['M'] = data_dict['M'][valid_indices]
            baseline = np.median(data_dict['M'], axis=1, keepdims=True)
            data_dict['M'] = (data_dict['M'] - baseline) / baseline
            # data_dict['M'] = data_dict['M'] / (np.max(np.abs(data_dict['M']), axis=1, keepdims=True) + 1e-4)
            # assert np.all(np.abs(data_dict['M']) <= 1)

            # plot distribution of delta F
            plots.distribution_plot(
                [[(data_dict['M'][:, 1: ] - data_dict['M'][:, : -1]).reshape(-1)]],
                ['All ROIs', ],
                left=-4.95, right=5, 
                interval=0.1,
                x_label='Delta F', y_label='Frequency',
                plot_dir='preprocess',
                plot_name=f'{exp_name}_delta_F',
                errormode='none',
                mode='errorshade',
                legend=False,
            )

            """
            The other way of testing some neural signal is like given some DDM and some neural trajactory in each trial, test how well does the neural trajectory follow the decision variable in DDM.
            Or, postulate that some task condition changes the parameter
            Or, show that DIC could be lower when neural data is used (how do we use neural data?)
            """

            # plot cumulative explained variance for first 2048 PCs
            pca = PCA(n_components=2048, svd_solver='full')
            act = pca.fit_transform(data_dict['M'].T)
            print('EV 10:', pca.explained_variance_ratio_[:10])
            cumulated = np.cumsum(pca.explained_variance_ratio_)
            plots.error_plot(
                np.arange(1, len(cumulated) + 1),
                [cumulated],
                x_label='Number of PCs', y_label='Explained variance',
                save_dir='preprocess',
                fig_name=f'{exp_name}_explained_variance',
                errormode='none',
                mode='errorshade',
                yticks=[0, 1]
            )
            print("Reduced Data Shape: ", act.shape)
            data_dict['PC'] = act.T

            print(f"512-dim Explained variance: {cumulated[511]}")
            print(f"max: {data_dict['M'].max()}, min: {data_dict['M'].min()}, std: {data_dict['M'].std(axis=1).mean()}")
            print('')

            # plot distribution of delta F
            plots.distribution_plot(
                [[(data_dict['PC'][:, 1: ] - data_dict['PC'][:, : -1]).reshape(-1)]],
                ['All ROIs', ],
                left=-4.95, right=5, 
                interval=0.1,
                x_label='Delta F', y_label='Frequency',
                plot_dir='preprocess',
                plot_name=f'{exp_name}_delta_F_PC',
                errormode='none',
                mode='errorshade',
                legend=False,
            )
            
            np.savez(out_filename, **data_dict)