import logging
import os
import json
from copy import deepcopy

import numpy as np
import pandas as pd

from configs.config_global import ROOT_DIR

def configs_transpose(config_dict, config_shape):
    """For each list of configs in config_dict, transpose the list to a new list, config_shape should be a 2d tuple"""
    new_config_dict = dict()
    for seed, config_list in config_dict.items():
        new_config_list = []
        for y in range(config_shape[1]):
            for x in range(config_shape[0]):
                new_config_list.append(config_list[x * config_shape[1] + y])
        new_config_dict[seed] = new_config_list
    return new_config_dict

def configs_dict_unpack(configs_dict):
    """unpack configs_dict to config_list"""
    new_config_list = []
    for configs_list in configs_dict.values():
        new_config_list += configs_list
    return new_config_list


def add_config_seed(config_list, number_of_seed):
    """
    take a list of configs.
    return a dictionary of list of configs,
    the key is the random seed, the value is the list
    """
    ret_configs = dict()
    for seed in range(number_of_seed):
        new_config_list = list()
        for cfg in config_list:
            new_cfg = deepcopy(cfg)
            new_cfg.seed = seed
            new_config_list.append(new_cfg)
        ret_configs[seed] = new_config_list
    return ret_configs


def auto_name(configs_dict, config_diffs):
    """Helper function for automatically naming models based on configs."""
    new_config_dict = {}
    for seed, config_list in configs_dict.items():
        new_configs_list = list()
        assert len(config_list) == len(config_diffs)
        for config, config_diff in zip(config_list, config_diffs):
            name = 'model'
            for key, val in config_diff.items():
                name += '_' + str(key) + str(val)

            # replace char that are not suitable for path
            name = name.replace(",", "").replace(" ", "_")
            name = name.replace("[", "_").replace("]", "_").replace(".", "_")
            name = name.replace("'", "").replace("/", '_')
            config.model_name = name + '_s' + str(seed)
            config.save_path = os.path.join(ROOT_DIR, 'experiments',
                                            config.experiment_name, config.model_name)
            new_configs_list.append(config)
        new_config_dict[seed] = new_configs_list
    return new_config_dict


def save_config(config, save_path, also_save_as_text=True, show_message=True):
    """
    Save config.
    adapted from https://github.com/gyyang/olfaction_evolution
    return True if the save path is new, False if training is already done
    """
    if os.path.exists(save_path):
        file_path = os.path.join(save_path, 'progress.txt')
        if not config.overwrite and os.path.exists(file_path):
            try:
                exp_data = pd.read_table(file_path)
                # TODO: check the last line of the file
                if len(exp_data.index) == config.max_batch // config.log_every:
                    if show_message:
                        logging.warning('Save dir {} already exists and training is already done. '.format(save_path)
                                + 'Skipped. (to overwrite use -o)')
                    return False
            except:
                pass
        
        if show_message:
            logging.warning('Save dir {} already exists!'.format(save_path)
                            + 'Storing info there anyway. ')
    else:
        os.makedirs(save_path)

    config_dict = config.to_dict()
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)

    if also_save_as_text:
        with open(os.path.join(save_path, 'config.txt'), "w") as f:
            for k, v in config_dict.items():
                f.write(str(k) + ' >>> ' + str(v) + '\n\n')
    return True

import configs.configs

def load_config(save_path):
    with open(os.path.join(save_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    # Create a new instance of your config class
    config = configs.configs.NeuralPredictionConfig()

    # Set attributes, handling nested structures
    for key, val in config_dict.items():
        if key == "dataset_config" and isinstance(val, dict):
            # If `dataset_config` is a dictionary, reinstantiate its values as DatasetConfig objects
            dataset_config_dict = {}
            for dataset_key, dataset_val in val.items():
                if isinstance(dataset_val, dict):  # Ensure it's a valid dictionary
                    dataset_config = configs.configs.DatasetConfig()
                    for k, v in dataset_val.items():
                        setattr(dataset_config, k, v)
                    dataset_config_dict[dataset_key] = dataset_config
                else:
                    raise ValueError(f"Unexpected value in dataset_config for key {dataset_key}: {dataset_val}")
            setattr(config, key, dataset_config_dict)
        else:
            setattr(config, key, val)

    return config

def _vary_config_combinatorial(base_config, config_ranges):
    """Return combinatorial configurations.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        base_config: BaseConfig object, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
            config_ranges should not have repetitive values

    Return:
        configs: a list of config objects [config1, config2, ...]
            Loops over all possible combinations of hp1, hp2, ...
        config_diffs: a list of config diff from base_config
    """
    # Unravel the input index
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = int(np.prod(dims))

    configs, config_diffs = list(), list()
    for i in range(n_max):
        new_config = deepcopy(base_config)

        config_diff = dict()
        indices = np.unravel_index(i, shape=dims)
        # Set up new config
        for key, index in zip(keys, indices):
            val = config_ranges[key][index]
            setattr(new_config, key, val)
            config_diff[key] = val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def _vary_config_sequential(base_config, config_ranges):
    """Return sequential configurations.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        base_config: BaseConfig object, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
            config_ranges values should have equal length,
            otherwise this will only loop through the shortest one

    Return:
        configs: a list of config objects [config1, config2, ...]
            Loops over all hyperparameters hp1, hp2 together sequentially
        config_diffs: a list of config diff from base_config
    """
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    if len(dims) == 0:
        n_max = 1
    else:
        n_max = np.min(dims)

    configs, config_diffs = list(), list()
    for i in range(n_max):
        new_config = deepcopy(base_config)
        config_diff = dict()
        for key in keys:
            val = config_ranges[key][i]
            setattr(new_config, key, val)
            config_diff[key] = val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def _vary_config_control(base_config, config_ranges):
    """Return control configurations.
    adapted from https://github.com/gyyang/olfaction_evolution

    Each config_range is gone through sequentially. The base_config is
    trained only once.

    Args:
        base_config: BaseConfig object, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
            base_config must contain keys in config_ranges

    Return:
        configs: a list of config objects [config1, config2, ...]
            Loops over all hyperparameters hp1, hp2 independently
        config_diffs: a list of config diff from base_config
    """
    keys = list(config_ranges.keys())
    # Remove the base_config value from the config_ranges
    new_config_ranges = {}
    for key, val in config_ranges.items():
        base_config_val = getattr(base_config, key)
        new_config_ranges[key] = [v for v in val if v != base_config_val]

    # Unravel the input index
    dims = [len(new_config_ranges[k]) for k in keys]
    n_max = int(np.sum(dims))

    configs, config_diffs = list(), list()
    configs.append(deepcopy(base_config))
    config_diffs.append({})

    for i in range(n_max):
        new_config = deepcopy(base_config)

        index = i
        for j, dim in enumerate(dims):
            if index >= dim:
                index -= dim
            else:
                break

        config_diff = dict()
        key = keys[j]

        val = new_config_ranges[key][index]
        setattr(new_config, key, val)
        config_diff[key] = val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def vary_config(base_config, config_ranges, mode,
                num_seed=1, default_name=False):
    """Return configurations.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        base_config: BaseConfig object, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
        mode: str, can take 'combinatorial', 'sequential', and 'control'
        num_seed: int, number of random seeds
        default_name: bool, whether to use auto_name function

    Return:
        configs_dict: a dictionary of list of config objects [config1, config2, ...]
            the keys in the dictionary is the random seed
    """
    if mode == 'combinatorial':
        _vary_config = _vary_config_combinatorial
    elif mode == 'sequential':
        _vary_config = _vary_config_sequential
    elif mode == 'control':
        _vary_config = _vary_config_control
    else:
        raise ValueError('Unknown mode {}'.format(str(mode)))

    assert 'seed' not in config_ranges.keys(), 'seed cannot be specified in config range'
    configs, config_diffs = _vary_config(base_config, config_ranges)
    configs_dict = add_config_seed(configs, num_seed)

    if default_name:
        for seed, config_list in configs_dict.items():
            for i, config in enumerate(config_list):
                config.model_name = str(i).zfill(6) + '_s' + str(seed)  # default name
                config.save_path = os.path.join(ROOT_DIR, 'experiments',
                                                config.experiment_name,
                                                config.model_name)
    else:
        # Automatic set names for configs_dict
        configs_dict = auto_name(configs_dict, config_diffs)
    return configs_dict
