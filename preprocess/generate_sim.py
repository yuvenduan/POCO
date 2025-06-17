import numpy as np
import os
import random
import numpy as np

from utils.curbd import threeRegionSim
from configs.config_global import SIM_DIR
from .utils import process_data_matrix

def run(
    mode = 1, # number of regions
    n = 500, # number of neurons in each region
    ga = 2.0, # chaos factor
    noise_std = 0, # noise standard deviation,
    T = 4096 * 0.01 * 5, # 4096 steps
    sparsity = 1,
    seed = 0,
    template_connectivity = None,
    connectivity_noise = 0,
    tseed = 0,
    pc_dim=128,
):

    random.seed(seed)
    np.random.seed(seed)

    if template_connectivity is not None:
        assert sparsity == 1, "sparsity should be 1 when using template connectivity"
        assert mode == 1, "only one region when using template connectivity"
        name = f'tsim_{n}_{ga}_{noise_std}_{connectivity_noise}_s{seed}_ts{tseed}'
    else:
        name = f'sim_{n}_{mode}_{ga}_{noise_std}_s{seed}'
        if sparsity != 1:
            name += f'_sparsity_{sparsity}'

    frac_inter = 0 if mode == 1 else 0.05
    out = threeRegionSim(
        number_units=n, dtData=0.01, tau=0.1, T=T, 
        fig_save_name=name + f'_.pdf', leadTime=500, fracInterReg=frac_inter, ga=ga, noise_std=noise_std, sparsity=sparsity, one_region=(mode == 1),
        template_connectivity=template_connectivity, connectivity_noise=connectivity_noise
    )

    if mode == 1:
        R = out['Ra']
    elif mode == 3:
        R = np.concatenate([out['Ra'], out['Rb'], out['Rc']], axis=0)
    data_dict = process_data_matrix(R, 'preprocess/sim', pc_dim=pc_dim, exp_name=name, normalize_mode='zscore', plot_window=64 * 5)

    os.makedirs(SIM_DIR, exist_ok=True)
    np.savez(os.path.join(SIM_DIR, f'{name}.npz'), **data_dict)

def save_sim_activity():

    os.makedirs(SIM_DIR, exist_ok=True)

    """
    # Uncomment this block to run the simulation with varying individual differences
    total_sims = 16 * 4 * 16
    cur = 0

    for n in [300]:
        for tseed in range(16):
            np.random.seed(n + tseed)
            template = np.random.randn(n, n)
            for noise_std in [0, 0.05, 0.5, 1]:
                for seed in range(16):
                    run(mode=1, n=n, ga=2.0, seed=seed, template_connectivity=template, connectivity_noise=noise_std, pc_dim=0, tseed=tseed, noise_std=0.1)
                    cur += 1
                    print(f'Finished {cur}/{total_sims} sims')
    """

    
    for n in [150, 300]:
        for tseed in range(16):
            np.random.seed(n + tseed)
            template = np.random.randn(n, n)
            for noise_std in [0]:
                for seed in range(1):
                    run(mode=1, n=n, ga=2.0, seed=seed, template_connectivity=template, connectivity_noise=noise_std, pc_dim=0, tseed=tseed, noise_std=0.1)
