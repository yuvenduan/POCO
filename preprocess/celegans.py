# Parts of code adapated from "BunDLe-Net embedding of C.elegans neuronal data in 3-D latent space"
# https://github.com/akshey-kumar/BunDLe-Net

import numpy as np
import os.path as osp
import h5py
import mat73
import os

from scipy import signal
from configs.config_global import CELEGANS_RAW_DIR, CELEGANS_PROCESSED_DIR
from preprocess.preprocess import process_data_matrix

def readmat(filename):
    """
    Read a matlab file and return the data dictionary.
    """
    with h5py.File(filename, 'r') as f:
        data = {}
        for k, v in f.items():
            data[k] = v
    return data

class Database:
    """
    Loading neuronal and behavioural data from matlab files 

    Attributes:
        data_set_no (int): The number of the data set.
        states (numpy.ndarray): A single array of states, where each number corresponds to a behaviour.
        state_names (list): List of state names.
        neuron_traces (numpy.ndarray): Array of neuron traces.
        neuron_names (numpy.ndarray): Array of neuron names.
        fps (float): Frames per second.

    Methods:
        exclude_neurons: Excludes specified neurons from the database.
        categorise_neurons: Categorises neurons based on whether it is sensory,
                            inter or motor neuron. 

    """
    def __init__(self, data_set_no):
        self.data_set_no = data_set_no
        data_dict = mat73.loadmat(osp.join(CELEGANS_RAW_DIR, 'NoStim_Data.mat'))
        data = data_dict['NoStim_Data']

        deltaFOverF_bc = data['deltaFOverF_bc'][self.data_set_no]
        derivatives = data['derivs'][self.data_set_no]
        NeuronNames = data['NeuronNames'][self.data_set_no]
        fps = data['fps'][self.data_set_no]
        States = data['States'][self.data_set_no]

        self.states = np.sum([n*States[s] for n, s in enumerate(States)], axis = 0).astype(int) # making a single states array in which each number corresponds to a behaviour
        self.state_names = [*States.keys()]
        self.neuron_traces = np.array(deltaFOverF_bc).T
        #self.derivative_traces = derivatives['traces'].T
        self.neuron_names = np.array(NeuronNames, dtype=object)
        self.fps = fps

        ### To handle bug in dataset 3 where in neuron_names the last entry is a list. we replace the list with the contents of the list
        self.neuron_names = np.array([x if not isinstance(x, list) else x[0] for x in self.neuron_names])


    def exclude_neurons(self, exclude_neurons):
        """
        Excludes specified neurons from the database.

        Args:
            exclude_neurons (list): List of neuron names to exclude.

        Returns:
            None

        """
        neuron_names = self.neuron_names
        mask = np.zeros_like(self.neuron_names, dtype='bool')
        for exclude_neuron in exclude_neurons:
            mask = np.logical_or(mask, neuron_names==exclude_neuron)
        mask = ~mask
        self.neuron_traces = self.neuron_traces[mask] 
        #self.derivative_traces = self.derivative_traces[mask] 
        self.neuron_names = self.neuron_names[mask]

    def _only_identified_neurons(self):
        mask = np.logical_not([x.isnumeric() for x in self.neuron_names])
        self.neuron_traces = self.neuron_traces[mask] 
        #self.derivative_traces = self.derivative_traces[mask] 
        self.neuron_names = self.neuron_names[mask]

    def categorise_neurons(self):
        self._only_identified_neurons()
        #neuron_list = mat73.loadmat('data/raw/Order279.mat')['Order279']
        #neuron_category = mat73.loadmat('data/raw/ClassIDs_279.mat')['ClassIDs_279']
        neuron_list = mat73.loadmat(osp.join(CELEGANS_RAW_DIR, 'Order279.mat'))['Order279']
        neuron_category = mat73.loadmat(osp.join(CELEGANS_RAW_DIR, 'ClassIDs_279.mat'))['ClassIDs_279']
        category_dict = {neuron: int(category) for neuron, category in zip(neuron_list, neuron_category)}

        mask = np.array([category_dict[neuron] for neuron in self.neuron_names])
        mask_s = mask == 1
        mask_i = mask == 2
        mask_m = mask == 3

        self.neuron_names_s = self.neuron_names[mask_s]
        self.neuron_names_i = self.neuron_names[mask_i]
        self.neuron_names_m = self.neuron_names[mask_m]

        self.neuron_traces_s = self.neuron_traces[mask_s]
        self.neuron_traces_i = self.neuron_traces[mask_i]
        self.neuron_traces_m = self.neuron_traces[mask_m]

        return mask

flat_partial = lambda x: x.reshape(x.shape[0],-1)

############################################
####### Data preprocessing functions #######
############################################

def bandpass(traces, f_l, f_h, sampling_freq):
    """
    Apply a bandpass filter to the input traces. Note: the input f_l and f_h are angular frequencies. 

    Parameters:
        traces (np.ndarray): Input traces to be filtered.
        f_l (float): Lower cutoff frequency in Hz.
        f_h (float): Upper cutoff frequency in Hz.
        sampling_freq (float): Sampling frequency in Hz.

    Returns:
        filtered (np.ndarray): Filtered traces.

    """    
    cut_off_h = f_h*sampling_freq/2 ## in units of sampling_freq/2
    cut_off_l= f_l*sampling_freq/2 ## in units of sampling_freq/2
    #### Note: the input f_l and f_h are angular frequencies. Hence the argument sampling_freq in the function is redundant: since the signal.butter function takes angular frequencies if fs is None.
    
    sos = signal.butter(4, [cut_off_l, cut_off_h], 'bandpass', fs=sampling_freq, output='sos')
    ### filtering the traces forward and backwards
    filtered = signal.sosfilt(sos, traces)
    filtered = np.flip(filtered, axis=1)
    filtered = signal.sosfilt(sos, filtered)
    filtered = np.flip(filtered, axis=1)
    return filtered

def preprocess_data(X, fps):
    """Preprocess the input data by applying bandpass filtering.
    
    Args:
        X: Input data of shape T * n_neurons.
        fps (float): Frames per second.
    
    Returns:
        numpy.ndarray: Preprocessed data after bandpass filtering.
    """
    time = 1 / fps * np.arange(0, X.shape[0])
    filtered = bandpass(X.T, f_l=1e-10, f_h=0.2, sampling_freq=fps).T

    return time, filtered

def celegans_preprocess():
    
    for data_set_no in range(5):
        
        db = Database(data_set_no)
        traces = db.neuron_traces
        time, filtered = preprocess_data(traces.T, float(db.fps))
        filtered = filtered.T

        save_dict = {
            'time': time,
            'neuron_names': db.neuron_names
        }
        
        ret = process_data_matrix(
            filtered, 
            "preprocess/celegans", 
            divide_baseline=False, # already dF / F
            normalize_mode='zscore',
            exp_name=f'celegans_{data_set_no}',
            pc_dim=128
        )
        save_dict.update(ret)
        os.makedirs(CELEGANS_PROCESSED_DIR, exist_ok=True)
        out_filename = osp.join(CELEGANS_PROCESSED_DIR, f'{data_set_no}.npz')
        np.savez(out_filename, **save_dict)

