import numpy as np
import os

from utils.curbd import threeRegionSim
from configs.config_global import SIM_DIR
from sklearn.decomposition import PCA
from analysis import plots

for mode in [1, 3]:
    for n in [200, 500, 1500, 3000, ]:

        frac_inter = 0 if mode == 1 else 0.05
        ga = 2.0 if (mode == 1 and n == 200) else 1.8
        out = threeRegionSim(number_units=n, dtData=0.1, tau=1, T=300, fig_save_name=f'sim_{n}_{mode}.png', leadTime=20, fracInterReg=frac_inter, ga=ga)
        out = threeRegionSim(number_units=n, dtData=0.1, tau=1, T=30000, fig_save_name=f'sim_long_{n}_{mode}.png', leadTime=500, fracInterReg=frac_inter, ga=ga)

        os.makedirs(SIM_DIR, exist_ok=True)
        if mode == 1:
            R = out['Ra'][:, ::10]
        elif mode == 3:
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
            save_dir='sim',
            fig_name=f'sim_explained_variance_{n}_{mode}',
            errormode='none',
            mode='errorshade',
            yticks=[0, 1]
        )

        print("Reduced Data Shape: ", act.shape)
        data_dict['PC'] = act.T
        print(f"512-dim Explained variance: {cumulated[min(512, n * mode) - 1]}")

        np.savez(os.path.join(SIM_DIR, f'sim_{n}_{mode}.npz'), **data_dict)

        data_dict = None
        out = None
        R = None