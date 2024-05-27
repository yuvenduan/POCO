"""Experiments and corresponding analysis.
format adapted from https://github.com/gyyang/olfaction_evolution

Each experiment is described by a function that returns a list of configurations
function name is the experiment name
"""

from collections import OrderedDict
from configs.configs import BaseConfig, NeuralPredictionConfig
from utils.config_utils import vary_config
from copy import deepcopy

import os.path as osp

def adjust_batch_size(configs):
    for seed, config_list in configs.items():
        for config in config_list:
            if config.shared_backbone == False:
                config.batch_size *= 19
    return configs

def linear_rnn_test():
    config = NeuralPredictionConfig()
    config.experiment_name = 'linear_rnn_test'

    config.train_fish_ids = [0, 1, 2, 3, 4]
    config.wdecay = 1e-6
    config.perform_val = False
    config.rnn_type = 'Linear'
    config.target_mode = 'raw'

    config_ranges = OrderedDict()
    config_ranges['hidden_size'] = [8, 64, 512]
    config_ranges['exp_types'] = [['control'], ['shocked'], ['reshocked']]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def autoregressive_rnns():
    config = NeuralPredictionConfig()
    config.experiment_name = 'autoregressive_rnns'
    config.loss_mode = 'autoregressive'

    config_ranges = OrderedDict()
    config_ranges['target_mode'] = ['raw', ]
    config_ranges['normalize_mode'] = ['none', ]
    config_ranges['shared_backbone'] = [True,  ]
    config_ranges['rnn_type'] = ['Transformer', 'S4', 'LSTM', 'RNN', ]
    
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def sim_compare_n_neurons():
    config = NeuralPredictionConfig()
    config.experiment_name = 'sim_compare_n_neurons'
    config.loss_mode = 'autoregressive'
    config.dataset = 'simulation'

    config_ranges = OrderedDict()
    config_ranges['n_regions'] = [1, 3, ]
    config_ranges['n_neurons'] = [200, 500, 1500, 3000, ]
    config_ranges['rnn_type'] = ['Transformer', 'S4', 'LSTM', 'RNN', ]
    
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def sim_compare_train_length():
    config = NeuralPredictionConfig()
    config.experiment_name = 'sim_compare_train_length'
    config.loss_mode = 'autoregressive'
    config.dataset = 'simulation'

    config_ranges = OrderedDict()
    config_ranges['n_regions'] = [1, ]
    config_ranges['n_neurons'] = [500, 3000, ]
    config_ranges['rnn_type'] = ['Transformer', 'RNN', ]
    config_ranges['train_data_length'] = [500, 3000, 10000, 30000, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def sim_compare_pca_dim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'sim_compare_pca_dim'
    config.loss_mode = 'autoregressive'
    config.dataset = 'simulation'

    config_ranges = OrderedDict()
    config_ranges['n_regions'] = [1, ]
    config_ranges['n_neurons'] = [500, 3000, ]
    config_ranges['rnn_type'] = ['Transformer', 'RNN', ]
    config_ranges['pc_dim'] = [None, 64, 512, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def seqvae_test():
    config = NeuralPredictionConfig()
    config.experiment_name = 'seqvae_test'
    config.model_type = 'SeqVAE'

    config.encoder_hidden_size = 128
    config.decoder_hidden_size = 32
    config.kl_loss_coef = 0.1

    config_ranges = OrderedDict()
    config_ranges['target_mode'] = ['raw', ]
    config_ranges['normalize_mode'] = ['zscore', ]
    config_ranges['shared_backbone'] = [True, False, ]
    config_ranges['kl_loss_coef'] = [0, 0.002, 0.01, 0.1, 0.3]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    adjust_batch_size(configs)
    return configs

def seqvae_sim_test():
    config = NeuralPredictionConfig()
    config.experiment_name = 'seqvae_sim_test'
    config.model_type = 'SeqVAE'
    config.dataset = 'simulation'

    config.encoder_hidden_size = 128
    config.decoder_hidden_size = 200
    config.kl_loss_coef = 0.1
    config.n_regions = 1

    config_ranges = OrderedDict()
    config_ranges['n_neurons'] = [200, 500, 1500, 3000, ]
    config_ranges['kl_loss_coef'] = [0.1, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
    adjust_batch_size(configs)
    return configs

def meta_rnn():
    config = NeuralPredictionConfig()
    config.experiment_name = 'meta_rnn'

    config.model_type = 'MetaRNN'
    config.inner_train_step = 1

    config_ranges = OrderedDict()
    config_ranges['algorithm'] = ['maml', 'none', ]
    config_ranges['teacher_forcing'] = [False, True, ]
    config_ranges['inner_lr'] = [3, 10, 100, ]
    config_ranges['inner_test_time_train_step'] = [2, 10, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            if config.algorithm == 'none':
                config.max_batch = config.log_every = 1
            
            if config.teacher_forcing == True:
                config.loss_mode = 'prediction'
            else:
                config.loss_mode = 'autoregressive'

    return configs

def meta_rnn_sim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'meta_rnn_sim'
    config.dataset = 'simulation'
    config.model_type = 'MetaRNN'

    config_ranges = OrderedDict()
    config_ranges['n_regions'] = [1, 3, ]
    config_ranges['n_neurons'] = [200, 500, 1500, 3000, ]
    config_ranges['algorithm'] = ['maml', 'none', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            if config.algorithm == 'none':
                config.max_batch = config.log_every = 1
            
            if config.teacher_forcing == True:
                config.loss_mode = 'prediction'
            else:
                config.loss_mode = 'autoregressive'

    return configs

def meta_rnn_compare_alpha():
    config = NeuralPredictionConfig()
    config.experiment_name = 'meta_rnn_compare_alpha'

    config.model_type = 'MetaRNN'
    config.inner_train_step = 1
    config.inner_test_time_train_step = 5
    config.learnable_alpha = False

    config_ranges = OrderedDict()
    config_ranges['algorithm'] = ['maml', 'none', ]
    config_ranges['teacher_forcing'] = [True, ]
    config_ranges['inner_lr'] = [3, 10, ]
    config_ranges['alpha'] = [0.3, 1, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            if config.algorithm == 'none':
                config.max_batch = config.log_every = 1
            
            if config.teacher_forcing == True:
                config.loss_mode = 'prediction'
            else:
                config.loss_mode = 'autoregressive'

    return configs

def conv_baselines():
    config = NeuralPredictionConfig()
    config.experiment_name = 'conv_baselines'

    config.wdecay = 1e-6
    config.lr = 1e-2
    config.max_batch = 2000
    config.model_type = 'Conv'
    config.loss_mode = 'prediction'

    config_ranges = OrderedDict()
    config_ranges['normalize_mode'] = ['none', ]
    config_ranges['kernel_size'] = [1, 2, 3, 5, 10, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    return configs