import numpy as np
import os

from utils.curbd import threeRegionSim
from configs.config_global import SIM_DIR
from preprocess.preprocess import run_pca
from sklearn.decomposition import PCA
from analysis import plots

def run(
    mode = 1, # number of regions
    n = 500, # number of neurons in each region
    ga = 2.0, # chaos factor
    noise_std = 0, # noise standard deviation,
    T = 3276.8,
    sparsity = 1
):

    name = f'sim_{n}_{mode}_{ga}_{noise_std}'
    if sparsity != 1:
        name += f'_sparsity_{sparsity}'

    frac_inter = 0 if mode == 1 else 0.05
    out = threeRegionSim(
        number_units=n, dtData=0.01, tau=0.1, T=10, 
        fig_save_name=name + '.png', leadTime=5, fracInterReg=frac_inter, ga=ga, noise_std=noise_std, sparsity=sparsity, one_region=(mode == 1)
    )
    out = threeRegionSim(
        number_units=n, dtData=0.01, tau=0.1, T=T, 
        fig_save_name=name + f'_long.png', leadTime=500, fracInterReg=frac_inter, ga=ga, noise_std=noise_std, sparsity=sparsity, one_region=(mode == 1)
    )

    if mode == 1:
        R = out['Ra']
    elif mode == 3:
        R = np.concatenate([out['Ra'], out['Rb'], out['Rc']], axis=0)
    data_dict = {'M': R, }

    # plot cumulative explained variance for first 2048 PCs
    pca = PCA(n_components=min(2048, n * mode), svd_solver='full')
    # randomly select 10000 samples to fit PCA
    fit_data = data_dict['M'].T[np.random.choice(data_dict['M'].shape[1], 5000, replace=False)]
    pca.fit(fit_data)
    act = pca.transform(data_dict['M'].T)
    print('EV 10:', pca.explained_variance_ratio_[:10])
    cumulated = np.cumsum(pca.explained_variance_ratio_)

    plots.error_plot(
        np.arange(1, len(cumulated) + 1),
        [cumulated],
        x_label='Number of PCs', y_label='Explained variance',
        save_dir='sim',
        fig_name=f'{name}_explained_variance',
        errormode='none',
        mode='errorshade',
        yticks=[0, 1]
    )

    print("Reduced Data Shape: ", act.shape)
    data_dict['PC'] = act.T
    print(f"512-dim Explained variance: {cumulated[min(512, n * mode) - 1]}")

    np.savez(os.path.join(SIM_DIR, f'{name}.npz'), **data_dict)

def save_sim_activity():

    os.makedirs(SIM_DIR, exist_ok=True)
    n = 1500
    for sparsity in [1]:
        run(mode=1, n=n, ga=1.6, sparsity=sparsity)

    return

    for n in [1536, ]:
        run(mode=1, n=n, ga=2.0, noise_std=0)

    for mode in [1, 3]:
        for n in [200, 500, 1500, 3000, ]:

            ga = 2.0 if (mode == 1 and n == 200) else 1.8
            run(mode=mode, n=n, ga=ga, noise_std=0)

    for noise_std in [0.01, 0.03, 0.05, 0.1, 0.2, ]:
        run(mode=1, n=500, ga=1.8, noise_std=noise_std)

    for ga in [1.4, 1.6, 1.8, 2.0, 2.2, ]:
        run(mode=1, n=500, ga=ga, noise_std=0)