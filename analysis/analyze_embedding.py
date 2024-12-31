import numpy as np
import os
import torch
import os.path as osp
import matplotlib.pyplot as plt
import umap

from analysis.plots import error_plot, grouped_plot, get_model_colors
from configs.configs import NeuralPredictionConfig
from datasets.datasets import get_baseline_performance
from analysis.plots import grouped_plot, errorbar_plot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.data_utils import get_exp_names, get_subject_ids
from configs.config_global import PROCESSED_DIR, SIM_DIR, VISUAL_PROCESSED_DIR

def pca_reduce(data: np.ndarray) -> np.ndarray:
    return PCA(n_components=2).fit_transform(data)

def tsne_reduce(data: np.ndarray) -> np.ndarray:
    return TSNE(n_components=2, random_state=42, perplexity=max(5, min(data.shape[0] // 10, 30))).fit_transform(data)

def umap_reduce(data: np.ndarray) -> np.ndarray:
    return umap.UMAP(n_components=2).fit_transform(data)

def reduce(data: np.ndarray, method: str) -> np.ndarray:
    if method == 'PCA':
        return pca_reduce(data)
    elif method == 'TSNE':
        return tsne_reduce(data)
    elif method == 'UMAP':
        return umap_reduce(data)
    else:
        raise ValueError(f"Unknown method {method}")

def visualize_embedding(cfg: NeuralPredictionConfig, methods: list = ['PCA', 'TSNE', 'UMAP']):
    # load model weights
    if cfg.decoder_type != "POYO" or cfg.model_type != 'Decoder':
        raise ValueError("This function is only for POYO model")

    params = torch.load(osp.join(cfg.save_path, 'net_best.pth'), weights_only=True)

    # warning: this is a hacky way to get the embedding weights, might not work if code changes
    unit_embed = params['decoder.unit_emb.weight'].cpu().numpy()
    sessiom_embed = params['decoder.session_emb.weight'].cpu().numpy()
    figure_dir = osp.join(cfg.save_path, 'embedding')
    os.makedirs(figure_dir, exist_ok=True)

    print("Unit embedding shape:", unit_embed.shape)
    print("Session/Animal embedding shape:", sessiom_embed.shape)

    # plot 1: show unit embedding
    if cfg.pc_dim is not None:
        cmap = plt.get_cmap('coolwarm')
        colors = [cmap(np.linspace(0, 1, cfg.pc_dim))] * sessiom_embed.shape[0]
        assert unit_embed.shape[0] == sessiom_embed.shape[0] * cfg.pc_dim
        session_start = np.arange(0, unit_embed.shape[0] + 1, cfg.pc_dim)
        s = None
    elif cfg.dataset == 'zebrafish':
        exp_names = get_exp_names()
        session_start = [0, ]
        colors = []

        for exp_type in cfg.exp_types:
            for i_exp, exp_name in enumerate(exp_names[exp_type]):
                if cfg.animal_ids != 'all' and i_exp not in cfg.animal_ids:
                    continue

                filename = os.path.join(PROCESSED_DIR, exp_name + '.npz')
                fishdata = np.load(filename)

                n_neurons = fishdata['M'].shape[0]
                color_index = np.full(n_neurons, 10)
                brain_regions = ['LHb', 'MHb', 'ctel', 'dthal', 'gc', 'raphe', 'tel', 'vent', 'vthal']

                for i, region in enumerate(brain_regions):
                    indices = fishdata['in_l_' + region] | fishdata['in_r_' + region]
                    color_index[indices] = i + 1
                
                colors.append([f'C{x}' for x in color_index])
                session_start.append(session_start[-1] + n_neurons)
        s = 1

        assert session_start[-1] == unit_embed.shape[0]
        assert len(colors) == sessiom_embed.shape[0]
        assert sum(len(x) for x in colors) == unit_embed.shape[0]
        
    for i in range(max(1, sessiom_embed.shape[0])):
        for method in methods:
            reduced_unit_embed = reduce(unit_embed[session_start[i]: session_start[i + 1]], method)
            plt.figure(figsize=(7, 5))
            plt.scatter(reduced_unit_embed[:, 0], reduced_unit_embed[:, 1], c=colors[i], s=s)
            plt.title(f"Unit Embedding ({method})")
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")

            if cfg.pc_dim is not None:
                # add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=cfg.pc_dim))
                sm._A = []
                plt.colorbar(sm, label='PC index', ax=plt.gca())
            elif cfg.dataset == 'zebrafish':
                # add legend that explains the 10 colors
                regions = ['LHb', 'MHb', 'ctel', 'dthal', 'gc', 'raphe', 'tel', 'vent', 'vthal']
                for j, region in enumerate(regions):
                    plt.scatter([], [], c=f'C{j}', label=region)
                plt.legend()

            plt.tight_layout()
            os.makedirs(osp.join(figure_dir, f'animal_{i}'), exist_ok=True)
            plt.savefig(osp.join(figure_dir, f'animal_{i}', f'unit_embedding_{method}.png'))
            plt.close()

            print(f"Animal {i} unit embedding visualization saved for {method}", flush=True)

    # plot 2: show session embedding
    if cfg.dataset == 'zebrafish':
        assert sessiom_embed.shape[0] == 19
        colors = [f'C{x // 5}' for x in range(19)]
    elif cfg.dataset == 'simulation':
        return
    else:
        raise NotImplementedError(f"Unknown dataset {cfg.dataset}")
    
    for method in methods:
        reduced_session_embed = reduce(sessiom_embed, method)
        plt.figure(figsize=(5, 5))
        plt.scatter(reduced_session_embed[:, 0], reduced_session_embed[:, 1], c=colors)
        plt.title(f"Session Embedding ({method})")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")

        if cfg.dataset == 'zebrafish':
            # add legend that explains the 4 colors
            cohorts = ['control', 'shocked', 'reshocked', 'katamine']
            for i, cohort in enumerate(cohorts):
                plt.scatter([], [], c=f'C{i}', label=cohort)
            plt.legend()

        plt.tight_layout()
        plt.savefig(osp.join(figure_dir, f'session_embedding_{method}.png'))
        plt.close()