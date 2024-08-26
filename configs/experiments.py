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

def test():
    config = NeuralPredictionConfig()
    config.experiment_name = 'test'
    config.max_batch = 1000

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_pc', ]
    config_ranges['model_label'] = ['POYO', 'Transformer', ]
    
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def sim_compare_train_length():
    config = NeuralPredictionConfig()
    config.experiment_name = 'sim_compare_train_length'
    config.loss_mode = 'autoregressive'
    config.dataset = 'simulation'
    config.pc_dim = None
    config.num_workers = 0

    config_ranges = OrderedDict()
    config_ranges['n_regions'] = [1, ]
    config_ranges['n_neurons'] = [64, 128, 256, 512, 768, 1024, 1536, ]
    config_ranges['rnn_type'] = ['Transformer', 'RNN', 'S4']
    config_ranges['train_data_length'] = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def tcn_sim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'tcn_sim'
    config.dataset = 'simulation'
    config.pc_dim = None
    config.model_type = 'TCN'
    config.loss_mode = 'prediction'

    config_ranges = OrderedDict()
    config_ranges['n_neurons'] = [512, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def sim_compare_noise():
    config = NeuralPredictionConfig()
    config.experiment_name = 'sim_compare_noise'
    config.experiment_name = 'sim_compare_train_length'
    config.loss_mode = 'autoregressive'
    config.dataset = 'simulation'
    config.train_data_length = 3000
    config.pc_dim = None

    config_ranges = OrderedDict()
    config_ranges['n_regions'] = [1, ]
    config_ranges['n_neurons'] = [500, ]
    config_ranges['rnn_type'] = ['Transformer', 'RNN', ]
    config_ranges['sim_noise_std'] = [0, 0.01, 0.03, 0.05, 0.1, 0.2, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
    return configs

def sim_compare_sparsity():
    config = NeuralPredictionConfig()
    config.experiment_name = 'sim_compare_sparsity'
    config.loss_mode = 'autoregressive'
    config.dataset = 'simulation'
    config.ga = 1.6
    config.pc_dim = None
    config.mem = 64

    config_ranges = OrderedDict()
    config_ranges['n_regions'] = [1, ]
    config_ranges['n_neurons'] = [1500, ]
    config_ranges['rnn_type'] = ['Transformer', 'RNN', ]
    config_ranges['sparsity'] = [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def sim_partial_obervable():
    config = NeuralPredictionConfig()
    config.experiment_name = 'sim_partial_obervable'
    config.loss_mode = 'autoregressive'
    config.dataset = 'simulation'
    config.train_data_length = 3000
    config.pc_dim = None

    config_ranges = OrderedDict()
    config_ranges['n_regions'] = [1, ]
    config_ranges['n_neurons'] = [500, ]
    config_ranges['rnn_type'] = ['Transformer', 'RNN', ]
    config_ranges['portion_observable_neurons'] = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
    return configs

def sim_compare_ga():
    config = NeuralPredictionConfig()
    config.experiment_name = 'sim_compare_ga'
    config.loss_mode = 'autoregressive'
    config.dataset = 'simulation'
    config.train_data_length = 3000
    config.pc_dim = None

    config_ranges = OrderedDict()
    config_ranges['n_regions'] = [1, ]
    config_ranges['n_neurons'] = [500, ]
    config_ranges['rnn_type'] = ['Transformer', 'RNN', ]
    config_ranges['ga'] = [1.4, 1.6, 1.8, 2.0, 2.2, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
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

def configure_dataset(configs: dict, large=False):
    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            label = config.dataset_label

            if label == 'spontaneous_pc':
                config.dataset = 'zebrafish'
                config.pc_dim = 512
            elif label == 'spontaneous':
                config.dataset = 'zebrafish'
                config.pc_dim = None
                config.batch_size = 8 * (1 + large)
                config.mem = 128
                config.test_set_window_stride = 8
            elif label == 'visual_pc':
                config.dataset = 'zebrafish_visual'
                config.pc_dim = 512
                config.use_stimuli = False
                config.use_eye_movements = False
            elif label == 'visual':
                config.dataset = 'zebrafish_visual'
                config.pc_dim = None
                config.use_stimuli = False
                config.use_eye_movements = False
                config.batch_size = 1 * (1 + large)
                config.mem = 128
                config.test_set_window_stride = 32
            elif label[:3] == 'sim':
                config.dataset = 'simulation'
                config.pc_dim = None
                config.sampling_rate = 5
                config.n_neurons = int(label[3:])
                config.mem = 128
                if config.n_neurons > 512:
                    config.batch_size = 16
            else:
                raise ValueError(f'Unknown dataset label: {label}')

    return configs

def configure_models(configs: dict):
    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            config.shared_backbone = True
            label = config.model_label
            
            if label[:2] == 'AR':
                config.loss_mode = 'autoregressive'
                config.model_type = 'Autoregressive'
                config.rnn_type = label[3:]
                config.num_layers = 4 if config.rnn_type in ['S4', 'Transformer'] else 1
            elif label[:6] == 'Latent':
                config.loss_mode = 'autoregressive'
                config.model_type = 'LatentModel'
                config.rnn_type = label[7:]
                if config.rnn_type == 'CTRNN':
                    config.rnn_type = 'CTRNNCell'
                elif config.rnn_type[: 5] == 'LRRNN':
                    config.rnn_rank = int(config.rnn_type[6:])
                    config.rnn_type = 'LRRNN'
                config.tf_interval = 4
            elif label == 'TCN':
                config.loss_mode = 'prediction'
                config.model_type = 'TCN'
                config.shared_backbone = config.pc_dim is not None
            elif label.split('_')[0] in ['Linear', 'MLP', 'Transformer', 'POYO']:
                config.model_type = 'Decoder'
                config.loss_mode = 'prediction'
                config.decoder_type = label.split('_')[0]
                config.encoder_dir = None
                config.separate_projs = config.decoder_type != 'POYO'

                if label[-5: ] == 'TOTEM':
                    data_label = config.dataset_label
                    config.encoder_dir = \
                        f'experiments/vqvae/model_dataset_label{data_label}_compression_factor{config.compression_factor}_s{seed}'
                    config.encoder_state_dict_file = 'net_20000.pth'
            else:
                raise ValueError(f'Unknown model label: {label}')

    return configs

def vqvae_pretrain(config: NeuralPredictionConfig, config_ranges, seeds=2):
    config.model_type = 'VQVAE'
    config.loss_mode = 'reconstruction'
    config.seq_length = 48
    config.pred_length = 0
    config.save_every = config.max_batch

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=seeds)
    return configs

def vqvae():
    config = NeuralPredictionConfig()
    config.experiment_name = 'vqvae'
    config.show_chance = False

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_pc', 'visual_pc', ]
    config_ranges['compression_factor'] = [4, 16, ]
    configs = vqvae_pretrain(config, config_ranges)
    configs = configure_dataset(configs)
    return configs

def vqvae_large(): # requires h100, for smaller gpu use smaller batch size
    config = NeuralPredictionConfig()
    config.experiment_name = 'vqvae'
    config.show_chance = False

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['visual', 'spontaneous', ]
    config_ranges['compression_factor'] = [4, 16, ]
    configs = vqvae_pretrain(config, config_ranges)
    configs = configure_dataset(configs, large=True)
    return configs

def compare_compression_factor():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_compression_factor'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous', 'spontaneous_pc', ]
    config_ranges['model_label'] = ['POYO', 'POYO_TOTEM', ]
    config_ranges['compression_factor'] = [4, 16, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def poyo_compare_num_latents():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_num_latents'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous', 'spontaneous_pc', ]
    config_ranges['model_label'] = ['POYO', 'POYO_TOTEM', ]
    config_ranges['poyo_num_latents'] = [4, 16, 64, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_lr():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_lr'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous', 'spontaneous_pc', ]
    config_ranges['model_label'] = ['POYO', 'POYO_TOTEM', 'Linear', ]
    config_ranges['use_lr_scheduler'] = [True, False, ]
    config_ranges['lr'] = [5e-5, 3e-4, 1e-3, 5e-3, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_wd():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_wd'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous', 'spontaneous_pc', ]
    config_ranges['model_label'] = ['POYO', 'POYO_TOTEM', 'Linear', ]
    config_ranges['wdecay'] = [1e-3, 1e-2, 0.1, 0.5, 1, 5, 10, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_hidden_size():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_hidden_size'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous', 'spontaneous_pc', ]
    config_ranges['model_label'] = ['POYO', 'POYO_TOTEM', ]
    config_ranges['decoder_hidden_size'] = [64, 256, 1024, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2, )
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_num_layers():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_num_layers'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous', 'spontaneous_pc', ]
    config_ranges['model_label'] = ['POYO', 'POYO_TOTEM', ]
    config_ranges['decoder_num_layers'] = [1, 2, 4, 8, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2, )
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_unit_dropout():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_unit_dropout'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous', 'spontaneous_pc', ]
    config_ranges['model_label'] = ['POYO', 'POYO_TOTEM']
    config_ranges['poyo_unit_dropout'] = [0, 0.1, 0.3, 0.5, 0.7, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2, )
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

model_list = [
    'POYO_TOTEM', 'Linear_TOTEM', 'MLP_TOTEM', 'Transformer_TOTEM',
    'POYO', 'Linear', 'MLP', 'Transformer', 
    'AR_Transformer', 'AR_S4', 'AR_RNN', 'AR_LSTM',
    'Latent_PLRNN', 'Latent_LRRNN_4', 'Latent_CTRNN',
    'TCN'
]

def compare_models_pc():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_pc', 'visual_pc', ]
    config_ranges['model_label'] = model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_sim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['sim256', 'sim1024', ]
    config_ranges['model_label'] = model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

single_neuron_model_list = [
    'POYO', 'Linear', 'POYO_TOTEM', 
]

def compare_models_single_neuron():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['visual', 'spontaneous', ]
    config_ranges['model_label'] = single_neuron_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_individual_region():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_individual_region'
    config.dataset = 'zebrafish'
    config.pc_dim = None
    config.normalize_mode = 'zscore'
    config.datalabel = 'spontaneous'

    config_ranges = OrderedDict()
    config_ranges['brain_regions'] = ['l_LHb', 'l_MHb', 'l_dthal', 'l_gc', 'l_raphe', 'l_vent', 'l_vthal', ]
    config_ranges['model_label'] = model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    return configs

def sim_compare_models_train_length():
    config = NeuralPredictionConfig()
    config.experiment_name = 'sim_compare_models'
    config.dataset = 'simulation'
    config.pc_dim = None
    config.sampling_rate = 5

    config_ranges = OrderedDict()
    config_ranges['n_neurons'] = [512, ]
    config_ranges['model_label'] = model_list
    config_ranges['train_data_length'] = [512, 2048, 8192, 65536, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    return configs