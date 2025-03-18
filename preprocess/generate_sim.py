import numpy as np
import os
import random
import numpy as np

from utils.curbd import threeRegionSim
from configs.config_global import SIM_DIR
from .utils import process_data_matrix
from sklearn.decomposition import PCA
from analysis import plots

def run(
    mode = 1, # number of regions
    n = 500, # number of neurons in each region
    ga = 2.0, # chaos factor
    noise_std = 0, # noise standard deviation,
    T = 3276.8,
    sparsity = 1,
    seed = 0
):

    random.seed(seed)
    np.random.seed(seed)

    name = f'sim_{n}_{mode}_{ga}_{noise_std}_s{seed}'
    if sparsity != 1:
        name += f'_sparsity_{sparsity}'

    frac_inter = 0 if mode == 1 else 0.05
    out = threeRegionSim(
        number_units=n, dtData=0.01, tau=0.1, T=T, 
        fig_save_name=name + f'_.pdf', leadTime=500, fracInterReg=frac_inter, ga=ga, noise_std=noise_std, sparsity=sparsity, one_region=(mode == 1)
    )

    if mode == 1:
        R = out['Ra']
    elif mode == 3:
        R = np.concatenate([out['Ra'], out['Rb'], out['Rc']], axis=0)
    data_dict = process_data_matrix(R, 'preprocess/sim', pc_dim=128, exp_name=name, normalize_mode='zscore', plot_window=64 * 5)

    os.makedirs(SIM_DIR, exist_ok=True)
    np.savez(os.path.join(SIM_DIR, f'{name}.npz'), **data_dict)

def save_sim_activity():

    # for n in [64, 128, 256, 384, 512, 1024, 1536, ]:
    for n in [128, 256, 512, ]:
        for seed in range(4):
            run(mode=1, n=n, ga=2.0, noise_std=0, seed=seed)
    return

    os.makedirs(SIM_DIR, exist_ok=True)
    n = 1500
    for sparsity in [1]:
        run(mode=1, n=n, ga=1.6, sparsity=sparsity)

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