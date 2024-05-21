import numpy as np
import os

from utils.curbd import threeRegionSim
from configs.config_global import SIM_DIR
from sklearn.decomposition import PCA
from analysis import plots

for mode in [1, 3]:
    for n in [200, 500, 1500, 3000]:

        out = threeRegionSim(number_units=n, dtData=0.1, tau=1, T=300, fig_save_name=f'sim_{n}_{mode}.png', leadTime=5)
        exit(0)
        out = threeRegionSim(number_units=n, dtData=0.1, tau=1, T=3000, fig_save_name=f'sim_long_{n}_{mode}.png', leadTime=1500)

        os.makedirs(SIM_DIR, exist_ok=True)
        R = np.concatenate([out['Ra'][:, ::10], out['Rb'][:, ::10], out['Rc'][:, ::10]], axis=0)
        data_dict = {'M': R, }

        # plot cumulative explained variance for first 2048 PCs
        pca = PCA(n_components=min(2048, n * mode), svd_solver='full')
        act = pca.fit_transform(data_dict['M'].T)
        print('EV 10:', pca.explained_variance_ratio_[:10])
        cumulated = np.cumsum(pca.explained_variance_ratio_)

        plots.error_plot(
            np.arange(1, len(cumulated) + 1),
            [cumulated],
            x_label='Number of PCs', y_label='Explained variance',
            save_dir='preprocess',
            fig_name=f'sim_explained_variance_{n}',
            errormode='none',
            mode='errorshade',
            yticks=[0, 1]
        )

        print("Reduced Data Shape: ", act.shape)
        data_dict['PC'] = act.T
        print(f"512-dim Explained variance: {cumulated[511]}")

        np.savez(os.path.join(SIM_DIR, f'sim_{n}.npz'), **data_dict)