import numpy as np
import os
import torch
import os.path as osp
import matplotlib.pyplot as plt
import umap

from analysis.plots import error_plot, grouped_plot, get_model_colors
from configs.configs import NeuralPredictionConfig, DatasetConfig
from datasets.datasets import get_baseline_performance
from analysis.plots import grouped_plot, errorbar_plot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.data_utils import get_exp_names
from datasets.datasets import Zebrafish, Mice

from configs.config_global import ZEBRAFISH_PROCESSED_DIR, SIM_DIR, FIG_DIR, MICE_BRAIN_AREAS, ZEBRAFISH_BRAIN_AREAS

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

def compute_distance_matrix(all_data: np.ndarray, session_start, all_indices: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Compute the distance matrix between different classes

    :param all_data: (n_units, n_features)
    :param session_start: list of indices that separate different sessions, should start with 0, n_sessions = len(session_start) - 1 
        distance is computed within each session, then averaged
        if session_start is None, distance is computed across all data
    :param indices: (n_units, ) class labels
    :param n_classes: number of classes
    :return: two distance matrices, one for euclidean distance and one for cosine similarity
    """

    if session_start is None:
        session_start = [0, len(all_data)]
    n_sessions = len(session_start) - 1

    weights = np.zeros((n_classes, n_classes))
    D_E = np.zeros((n_classes, n_classes)) # euclidean distance
    D_C = np.zeros((n_classes, n_classes)) # cosine similarity

    for i_session in range(n_sessions):
        start = session_start[i_session]
        end = session_start[i_session + 1]

        data = all_data[start: end]
        indices = all_indices[start: end]

        for i in range(n_classes):
            for j in range(n_classes):
                A = data[indices == i + 1]
                B = data[indices == j + 1]
                dist_e = np.linalg.norm(A[:, None] - B[None, :], axis=2) # n * m
                dist_c = np.dot(A, B.T) / (np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None, :]) # n * m
                # exclude self distance
                if i == j:
                    D_E[i, i] += dist_e[dist_e > 1e-6].sum()
                    D_C[i, i] += dist_c[dist_c < 1 - 1e-5].sum()
                    weights[i, i] += len(A) * (len(A) - 1)
                    assert (dist_e <= 1e-6).sum() == A.shape[0], "are there identical embeddings?"
                    assert (dist_c >= 1 - 1e-5).sum() == A.shape[0], "are there identical embeddings?"
                else:
                    D_E[i, j] += dist_e.sum()
                    D_C[i, j] += dist_c.sum()
                    D_E[j, i] = D_E[i, j]
                    D_C[j, i] = D_C[i, j]
                    weights[i, j] += len(A) * len(B)
                    weights[j, i] = weights[i, j]
    
    # normalize by number of samples
    D_E /= weights
    D_C /= weights

    return D_E, D_C

def plot_distance_matrices(D_E: np.ndarray, D_C: np.ndarray, names: list, figure_dir: str, figure_name: str):
    # plot distance matrix
    n_classes = len(names)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    im = ax[0].imshow(D_E, cmap='hot', interpolation='nearest')
    ax[0].set_title("Euclidean Distance")
    ax[0].set_xticks(np.arange(n_classes))
    ax[0].set_yticks(np.arange(n_classes))
    ax[0].set_xticklabels(names, rotation=45)
    ax[0].set_yticklabels(names, rotation=45)
    fig.colorbar(im, ax=ax[0])

    im = ax[1].imshow(D_C, cmap='coolwarm', interpolation='nearest')
    ax[1].set_title("Cosine Similarity")
    ax[1].set_xticks(np.arange(n_classes))
    ax[1].set_yticks(np.arange(n_classes))
    ax[1].set_xticklabels(names, rotation=45)
    ax[1].set_yticklabels(names, rotation=45)
    fig.colorbar(im, ax=ax[1])

    plt.tight_layout()
    plt.savefig(osp.join(figure_dir, figure_name))
    plt.close()

def visualize_embedding(cfg: NeuralPredictionConfig, methods: list = ['PCA', 'TSNE', 'UMAP'], sub_dir_name=None):
    # load model weights
    if cfg.decoder_type not in ["POCO", "POYO"] or cfg.model_type != 'Decoder':
        raise ValueError("This function is only for POYO model")

    params = torch.load(osp.join(cfg.save_path, 'net_best.pth'), weights_only=True)

    # warning: this is a hacky way to get the embedding weights, might not work if code changes
    unit_embed = params['decoder.unit_emb.weight'].cpu().numpy()
    sessiom_embed = params['decoder.session_emb.weight'].cpu().numpy()

    if sub_dir_name is None:
        sub_dir_name = cfg.dataset_label[0]
    figure_dir = osp.join(FIG_DIR, 'embedding', sub_dir_name, "seed" + str(cfg.seed))
    os.makedirs(figure_dir, exist_ok=True)

    print("Unit embedding shape:", unit_embed.shape)
    print("Session embedding shape:", sessiom_embed.shape)

    dataset_label = cfg.dataset_label[0]
    dataset_config: DatasetConfig = cfg.dataset_config[dataset_label]
    s = None
    masks = None
    assert len(cfg.dataset_label) == 1, "Only one dataset is supported for now"
    
    region_ids = None

    if dataset_config.pc_dim is not None:
        cmap = plt.get_cmap('coolwarm')
        colors = [cmap(np.linspace(0, 1, dataset_config.pc_dim))] * sessiom_embed.shape[0]
        assert unit_embed.shape[0] == sessiom_embed.shape[0] * dataset_config.pc_dim
        session_start = np.arange(0, unit_embed.shape[0] + 1, dataset_config.pc_dim)

    elif dataset_label == 'zebrafish':
        session_start = [0, ]
        colors = []
        masks = []
        dataset = Zebrafish(cfg.dataset_config[dataset_label])
        s = 3
        region_names = ZEBRAFISH_BRAIN_AREAS

        for unit_type in dataset.unit_types:
            session_start.append(session_start[-1] + len(unit_type))
            colors.append(['C' + str(i if i <= 9 else i - 9) for i in unit_type])
            masks.append(unit_type > 0)
        region_ids = np.concatenate(dataset.unit_types)
        region_ids[region_ids > 9] = region_ids[region_ids > 9] - 9

    elif dataset_label == 'mice':
        session_start = [0, ]
        colors = []
        s = 6
        dataset = Mice(cfg.dataset_config[dataset_label])
        region_names = MICE_BRAIN_AREAS

        for unit_type in dataset.unit_types:
            session_start.append(session_start[-1] + len(unit_type))
            colors.append(['C' + str(i) for i in unit_type])

        region_ids = np.concatenate(dataset.unit_types) + 1
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    return_dict = {}

    # plot 1: average distance between brain regions
    if region_ids is not None and dataset_label in ['mice', 'zebrafish']:
        n_classes = len(region_names)
        D_E, D_C = compute_distance_matrix(unit_embed, session_start, region_ids, n_classes)
        plot_distance_matrices(D_E, D_C, region_names, figure_dir, "distance_matrix.pdf")

        return_dict['unit_D_E'] = D_E
        return_dict['unit_D_C'] = D_C
        return_dict['region_names'] = region_names

    for i in range(sessiom_embed.shape[0]):
        # plot 2: visualize unit embedding
        for method in methods:

            reduced_unit_embed = reduce(unit_embed[session_start[i]: session_start[i + 1]], method)
            if masks is not None:
                reduced_unit_embed = reduced_unit_embed[masks[i]]
                color = [colors[i][j] for j in range(len(colors[i])) if masks[i][j]]
            else:
                color = colors[i]

            plt.figure(figsize=(4.5, 3.5))
            plt.scatter(reduced_unit_embed[:, 0], reduced_unit_embed[:, 1], c=color, s=s)
            plt.title(f"Unit Embedding ({method})")
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")

            if dataset_config.pc_dim is not None:
                # add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=dataset_config.pc_dim))
                sm._A = []
                plt.colorbar(sm, label='PC index', ax=plt.gca())
            else:
                # add legend that explains the 10 colors
                for j, region in enumerate(region_names):
                    plt.scatter([], [], c=f'C{j}', label=region)
                plt.legend()

            plt.tight_layout()
            os.makedirs(osp.join(figure_dir, f'session_{i}'), exist_ok=True)
            plt.savefig(osp.join(figure_dir, f'session_{i}', f'unit_embedding_{method}.pdf'))
            plt.close()

            print(f"Animal {i} unit embedding visualization saved for {method}", flush=True)

    # plot 3: show session embedding
    session_classes = np.zeros(sessiom_embed.shape[0])
    class_names = None
    if dataset_label in ['zebrafish', 'zebrafish_pc']:
        assert sessiom_embed.shape[0] == 19
        session_classes = np.array([x // 5 + 1 for x in range(19)])
        class_names = ['control', 'shocked', 'reshocked', 'katamine']
    elif dataset_label in ['mice', 'mice_pc']:
        assert sessiom_embed.shape[0] == 12
        session_classes = np.array([1] + [2] * 5 + [3] * 4 + [4] * 2)
        class_names = ['M1', 'M2', 'M3', 'M4']
    else:
        return return_dict

    colors = [f'C{i}' for i in session_classes]

    # correct the session embedding by adding the mean of the unit embedding, in each session
    for i in range(sessiom_embed.shape[0]):
        sessiom_embed[i] += np.mean(unit_embed[session_start[i]: session_start[i + 1]], axis=0)

    if class_names is not None:
        n_classes = len(class_names)
        D_E, D_C = compute_distance_matrix(sessiom_embed, None, session_classes, n_classes)
        plot_distance_matrices(D_E, D_C, class_names, figure_dir, "session_distance_matrix.pdf")

        return_dict['session_D_E'] = D_E
        return_dict['session_D_C'] = D_C
        return_dict['session_names'] = class_names
    
    for method in methods:
        reduced_session_embed = reduce(sessiom_embed, method)
        plt.figure(figsize=(4, 3.5))
        plt.scatter(reduced_session_embed[:, 0], reduced_session_embed[:, 1], c=colors, s=48, marker='x')
        plt.title(f"Session Embedding ({method})")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")

        if class_names is not None:
            for i, region in enumerate(class_names):
                plt.scatter([], [], c=f'C{i + 1}', label=region)
            plt.legend()

        plt.tight_layout()
        plt.savefig(osp.join(figure_dir, f'session_embedding_{method}.pdf'))
        plt.close()

    return return_dict

def analyze_embedding_all_seeds(cfgs, idx, methods=['PCA', 'TSNE', 'UMAP'], sub_dir_name=None):
    """
    Analyze the embedding of all seeds
    """

    return_dict_list = []

    for seed in cfgs.keys():
        cfg: NeuralPredictionConfig = cfgs[seed][idx]

        if sub_dir_name is None:
            sub_dir_name = cfg.dataset_label[0]

        return_dict = visualize_embedding(cfg, methods, sub_dir_name)
        return_dict_list.append(return_dict)

    figure_dir = osp.join(FIG_DIR, 'embedding', sub_dir_name)

    if 'unit_D_E' in return_dict_list[0]:
        unit_D_E = np.mean([return_dict['unit_D_E'] for return_dict in return_dict_list], axis=0)
        unit_D_C = np.mean([return_dict['unit_D_C'] for return_dict in return_dict_list], axis=0)
        region_names = return_dict_list[0]['region_names']
        plot_distance_matrices(unit_D_E, unit_D_C, region_names, figure_dir, "average_distance_matrix.pdf")

    if 'session_D_E' in return_dict_list[0]:
        session_D_E = np.mean([return_dict['session_D_E'] for return_dict in return_dict_list], axis=0)
        session_D_C = np.mean([return_dict['session_D_C'] for return_dict in return_dict_list], axis=0)
        class_names = return_dict_list[0]['session_names']
        plot_distance_matrices(session_D_E, session_D_C, class_names, figure_dir, "average_session_distance_matrix.pdf")