import os
import os.path as osp
import torch
import logging

from torch.utils.data import DataLoader, Subset

from configs.configs import BaseConfig, NeuralPredictionConfig, DatasetConfig
import datasets
from configs.config_global import ROOT_DIR

# TODO: could implement this entire structure as an iterator,
# so that a batch is a list of batches for each dataloader
# to consider reset all iterator at the end of training and testing
# an alternative could be a data iterator that keep iterating though all datasets
# note that the notion of epoch doesn't applies anymore
class DatasetIters(object):

    def __init__(self, config: NeuralPredictionConfig, phase: str):
        """
        Initialize a list of data loaders
        if only one dataset is specified, return a list containing one data loader
        """
        if type(config.dataset) is list or type(config.dataset) is tuple:
            dataset_list = config.dataset
        elif type(config.dataset) is str:
            dataset_list = [config.dataset]
        else:
            raise NotImplementedError('Dataset config not recognized')

        self.num_datasets = len(dataset_list)
        self.phase = phase
        assert len(config.mod_w) == self.num_datasets, 'mod_w and dataset len must match'

        self.data_iters = []
        self.iter_lens = []
        self.min_iter_len = None

        self.data_loaders = []
        self.input_sizes = []
        self.unit_types = []

        for dataset, label in zip(dataset_list, config.dataset_label):
            data_loader, input_size, unit_type = init_single_dataset(dataset, phase, config.dataset_config[label])
            self.data_loaders.append(data_loader)
            self.input_sizes.append(input_size)
            self.unit_types.append(unit_type)

        self.reset()

    def reset(self):
        # recreate iterator for each of the dataset
        self.data_iters = []
        self.iter_lens = []
        for data_l in self.data_loaders:
            data_iter = iter(data_l)
            self.data_iters.append(data_iter)
            self.iter_lens.append(len(data_iter))
        self.min_iter_len = min(self.iter_lens)

    def get_baselines(self, key='session_copy_mse'):
        assert self.phase != 'train', 'Baselines are only computed for test and val'
        all_baselines = []
        for loader in self.data_loaders:
            all_baselines.extend(loader.dataset.baseline[key])
        return all_baselines

def init_single_dataset(dataset_name: str, phase: str, config: DatasetConfig):
    collate_f = None
    train_flag = phase == 'train'
    input_size = None

    if dataset_name == 'zebrafish':
        dataset = datasets.Zebrafish(config, phase=phase)
    elif dataset_name == 'simulation':
        dataset = datasets.Simulation(config, phase=phase)
    elif dataset_name == 'zebrafish_stim':
        dataset = datasets.StimZebrafish(config, phase=phase)
    elif dataset_name == 'celegans':
        dataset = datasets.Celegans(config, phase=phase)
    elif dataset_name == 'celegansflavell':
        dataset = datasets.CelegansFlavell(config, phase=phase)
    elif dataset_name == 'mice':
        dataset = datasets.Mice(config, phase=phase)
    else:
        raise NotImplementedError('Dataset not implemented')

    collate_f = dataset.collate_fn
    input_size = dataset.input_size
    unit_types = dataset.unit_types
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=train_flag,
                             num_workers=config.num_workers, collate_fn=collate_f)

    return data_loader, input_size, unit_types
