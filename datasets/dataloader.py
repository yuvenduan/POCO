import os
import os.path as osp
import torch
import logging

from torch.utils.data import DataLoader, Subset

from configs.configs import BaseConfig, SupervisedLearningBaseConfig
import datasets
from configs.config_global import ROOT_DIR

# TODO: could implement this entire structure as an iterator,
# so that a batch is a list of batches for each dataloader
# to consider reset all iterator at the end of training and testing
# an alternative could be a data iterator that keep iterating though all datasets
# note that the notion of epoch doesn't applies anymore
class DatasetIters(object):

    def __init__(self, config: SupervisedLearningBaseConfig, phase: str):
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
        assert len(config.mod_w) == self.num_datasets, 'mod_w and dataset len must match'

        self.data_iters = []
        self.iter_lens = []
        self.min_iter_len = None

        self.data_loaders = []
        self.datum_sizes = []
        for d_set in dataset_list:
            data_loader, datum_size = init_single_dataset(d_set, phase, config)
            self.data_loaders.append(data_loader)
            self.datum_sizes.append(datum_size)
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

def init_single_dataset(dataset_name: str, phase: str, config: SupervisedLearningBaseConfig):
    collate_f = None
    train_flag = phase == 'train'
    datum_size = None

    if dataset_name == 'zebrafish':
        data_set = datasets.Zebrafish(config, phase=phase)
        collate_f = data_set.collate_fn
        datum_size = data_set.datum_size
    elif dataset_name == 'simulation':
        data_set = datasets.Simulation(config, phase=phase)
        collate_f = data_set.collate_fn
        datum_size = data_set.datum_size
    elif dataset_name == 'zebrafish_visual':
        data_set = datasets.VisualZebrafish(config, phase=phase)
        collate_f = data_set.collate_fn
        datum_size = data_set.datum_size
    elif dataset_name == 'zebrafish_stim':
        data_set = datasets.StimZebrafish(config, phase=phase)
        collate_f = data_set.collate_fn
        datum_size = data_set.datum_size
    elif dataset_name == 'celegans':
        data_set = datasets.Celegans(config, phase=phase)
        collate_f = data_set.collate_fn
        datum_size = data_set.datum_size
    elif dataset_name == 'mice':
        data_set = datasets.Mice(config, phase=phase)
        collate_f = data_set.collate_fn
        datum_size = data_set.datum_size
    else:
        raise NotImplementedError('Dataset not implemented')

    data_loader = DataLoader(data_set, batch_size=config.batch_size, shuffle=train_flag,
                             num_workers=config.num_workers, collate_fn=collate_f) # drop last is False, not sure if this causes problems
    return data_loader, datum_size
