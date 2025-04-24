import numpy as np
import os

from analysis import plots
from sklearn.cluster import KMeans
from scipy import signal
from sklearn.decomposition import PCA
from configs.config_global import FIG_DIR

def bandpass(traces, f_l, f_h, sampling_freq=1):
    """
    Apply a bandpass filter to the input traces.

    Parameters:
        traces (np.ndarray): Input traces to be filtered.
        f_l (float): Lower cutoff frequency in Hz.
        f_h (float): Upper cutoff frequency in Hz.
        sampling_freq (float): Sampling frequency in Hz.

    Returns:
        filtered (np.ndarray): Filtered traces.

    """
    cut_off_h = f_h * sampling_freq / 2 ## in units of sampling_freq/2
    cut_off_l = f_l * sampling_freq / 2 ## in units of sampling_freq/2
    #### Note: the input f_l and f_h are angular frequencies. Hence the argument sampling_freq in the function is redundant: since the signal.butter function takes angular frequencies if fs is None.
    
    sos = signal.butter(4, [cut_off_l, cut_off_h], 'bandpass', fs=sampling_freq, output='sos')
    ### filtering the traces forward and backwards
    filtered = signal.sosfilt(sos, traces)
    filtered = np.flip(filtered, axis=1)
    filtered = signal.sosfilt(sos, filtered)
    filtered = np.flip(filtered, axis=1)
    return filtered

def run_pca(data, exp_name='', n_components=2048, fig_dir='preprocess'):
    """
    data: n_neurons * T
    return: n_components * T
    """
    n_components = min(n_components, data.shape[1], data.shape[0])
    pca = PCA(n_components=n_components, svd_solver='full')
    act = pca.fit_transform(data.T)
    print('EV 10:', pca.explained_variance_ratio_[:10])
    cumulated = np.cumsum(pca.explained_variance_ratio_)
    plots.error_plot(
        np.arange(1, len(cumulated) + 1),
        [cumulated],
        x_label='Number of PCs', y_label='Explained variance',
        save_dir=fig_dir,
        fig_name=f'{exp_name}_explained_variance',
        errormode='none',
        mode='errorshade',
        yticks=[0, 1]
    )

    print("Reduced Data Shape: ", act.shape)
    print(f"Cumulative Explained Variance for {n_components} PCs", cumulated[-1])
    print('')
    return act.T

def get_clustered_data(data, n_clusters=500, seed=0):
    """
    Do KMeans clustering on correlation matrix
    data: n_neurons * T
    return: n_clusters * T
    """
    corr_matrix = np.corrcoef(data)
    print("Correlation Matrix Shape: ", corr_matrix.shape)
    corr_matrix[np.isnan(corr_matrix)] = 0
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(corr_matrix)
    labels = kmeans.labels_
    
    # compute the mean of time traces in each cluster
    clustered_data = np.zeros((n_clusters, data.shape[1]))
    for i in range(n_clusters):
        clustered_data[i] = data[labels == i].mean(axis=0)
    return clustered_data

def plot_delta_F(data, exp_name='', fig_dir='preprocess', suffix=''):
    """
    Plot the difference between adjacent frames
    data: n_neurons * T
    """
    # plot distribution of delta F
    plots.distribution_plot(
        [[(data[:, 1: ] - data[:, : -1]).reshape(-1)]],
        ['All ROIs', ],
        left=-4.95, right=5, 
        interval=0.1,
        x_label='Delta F', y_label='Frequency',
        plot_dir=fig_dir,
        plot_name=f'{exp_name}_delta_F{suffix}',
        errormode='none',
        mode='errorshade',
        legend=False,
    )

def process_data_matrix(
    data, fig_dir, 
    roi_indices=None, 
    divide_baseline=False, 
    normalize_mode='zscore', 
    n_clusers=[], 
    plot_max_min=False, 
    exp_name='',
    pc_dim=2048,
    plot_window=64,
    filter_mode='none'
):
    """
    Process the data matrix, also plot the distribution of min F and max F, and sample traces of first 10 PCs / 10 random neurons

    :param data: n_neurons * T
    :param roi_indices: n_neurons or None, represent the valid ROIs to use
    :param divide_baseline: whether to calculate delta F / F
    :param normalize_mode: 'none', 'zscore', 'max'
    :param n_clusters: list of number of clusters to use
    :param plot_max_min: whether to plot the distribution of min F and max F
    :param exp_name: experiment name
    :param pc_dim: number of PCs to use
    :param plot_window: window size to plot
    :param filter_mode: 'none', 'lowpass', 'highpass', 'bandpass'

    return: dict with keys 'valid_indices', 'M', 'PC', 'FC_{n}', where n is in n_clusters
    """
    data_dict = {}
    n, T = data.shape
    sub_fig_dir = os.path.join(FIG_DIR, fig_dir, exp_name)

    if roi_indices is None:
        roi_indices = np.ones(n, dtype=bool)
    
    if divide_baseline:
        baseline = np.median(data, axis=1, keepdims=True)
        roi_indices &= baseline.reshape(-1) > 1
        data = (data - baseline) / (baseline + 1e-6)

    if plot_max_min:
        # min F
        plots.distribution_plot(
            [[data_dict['M'].min(axis=1)], [data_dict['M'].min(axis=1)[roi_indices]]],
            ['All ROIs', 'Filtered ROIs'],
            left=-100, right=100, 
            interval=10,
            x_label='Min F', y_label='Frequency',
            plot_dir=sub_fig_dir,
            plot_name=f'min_F',
            errormode='none'
        )

        # max F
        plots.distribution_plot(
            [[data_dict['M'].max(axis=1)], [data_dict['M'].max(axis=1)[roi_indices]]],
            ['All ROIs', 'Filtered ROIs'],
            left=-50, right=2000, 
            interval=10,
            x_label='Max F', y_label='Frequency',
            plot_dir=sub_fig_dir,
            plot_name=f'max_F',
            errormode='none'
        )

    if normalize_mode == 'zscore':
        mu, std = np.mean(data, axis=1, keepdims=True), np.std(data, axis=1, keepdims=True)
        normalized = (data - mu) / (std + 1e-6)
    elif normalize_mode == 'max':
        normalized = data / (np.max(np.abs(data), axis=1, keepdims=True) + 1e-6)
    else:
        normalized = data

    if filter_mode == 'lowpass':
        normalized = bandpass(normalized, f_l=1e-6, f_h=0.2)
    elif filter_mode == 'highpass':
        normalized = bandpass(normalized, f_l=0.001, f_h=0.99)
    elif filter_mode == 'bandpass':
        normalized = bandpass(normalized, f_l=0.001, f_h=0.2)
    else:
        assert filter_mode == 'none', f"Unknown filter mode {filter_mode}"

    print(f"Data shape: {data.shape}, Valid indices: {roi_indices.sum()}")
    data_dict['valid_indices'] = roi_indices
    data_dict['M'] = normalized

    if pc_dim == 0:
        return data_dict
    
    normalized = normalized[roi_indices]
    plot_delta_F(normalized, fig_dir=sub_fig_dir, exp_name=exp_name)
    data_dict['PC'] = run_pca(normalized, fig_dir=sub_fig_dir, exp_name=exp_name, n_components=pc_dim)
    plot_delta_F(data_dict['PC'], fig_dir=sub_fig_dir, suffix='_PC', exp_name=exp_name)

    # plot first 10 PCs
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 10))
    for i in range(10):
        ax = plt.subplot(10, 2, i * 2 + 1)
        ax.plot(data_dict['PC'][i])
        ax.title.set_text(f'PC {i}, full trace')
        # plot a random plot_window-step window
        ax = plt.subplot(10, 2, i * 2 + 2)
        t = np.random.randint(0, data_dict['PC'].shape[1] - plot_window)
        ax.plot(data_dict['PC'][i, t: t + plot_window])
        ax.title.set_text(f'PC {i}, {plot_window}-step window')
    plt.tight_layout()
    plt.savefig(os.path.join(sub_fig_dir, f'{exp_name}_first_10_PCs.pdf'))
    plt.close()

    # plot 10 random neurons
    plt.figure(figsize=(7, 10))
    for i in range(10):
        idx = np.random.randint(0, normalized.shape[0])
        ax = plt.subplot(10, 2, i * 2 + 1)
        ax.plot(normalized[idx])
        ax.title.set_text(f'Neuron {idx}, full trace')
        # plot a random plot_window-step window
        ax = plt.subplot(10, 2, i * 2 + 2)
        t = np.random.randint(0, normalized.shape[1] - plot_window)
        ax.plot(normalized[idx, t: t + plot_window])
        ax.title.set_text(f'Neuron {idx}, {plot_window}-step window')
    plt.tight_layout()
    plt.savefig(os.path.join(sub_fig_dir, f'{exp_name}_10_random_neurons.pdf'))
    plt.close()

    for n_cluster in n_clusers:
        data_dict[f'FC_{n_cluster}'] = get_clustered_data(normalized, n_cluster)
    return data_dict
