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
    config.loss_mode = 'autoregressive'
    config.model_type = 'Autoregressive'
    config.max_batch = 1000

    config_ranges = OrderedDict()
    config_ranges['rnn_type'] = ['S4', 'LSTM', ]
    
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
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

def autoregressive_rnns_average():
    config = NeuralPredictionConfig()
    config.experiment_name = 'autoregressive_rnns_average'
    config.loss_mode = 'autoregressive'
    config.brain_regions = 'average'
    config.pc_dim = None

    config_ranges = OrderedDict()
    config_ranges['target_mode'] = ['raw', ]
    config_ranges['normalize_mode'] = ['zscore', ]
    config_ranges['shared_backbone'] = [True,  ]
    config_ranges['rnn_type'] = ['Transformer', 'S4', 'LSTM', 'RNN', ]
    
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def autoregressive_rnns_individual_region():
    config = NeuralPredictionConfig()
    config.experiment_name = 'autoregressive_rnns_individual_region'
    config.loss_mode = 'autoregressive'
    config.sampling_rate = 1
    config.pc_dim = None
    config.mem = 64

    config_ranges = OrderedDict()
    config_ranges['target_mode'] = ['raw', ]
    config_ranges['normalize_mode'] = ['zscore', ]
    config_ranges['shared_backbone'] = [True,  ]
    config_ranges['brain_regions'] = ['l_LHb', 'l_MHb', 'l_ctel', 'l_dthal', 'l_gc', 'l_raphe', 'l_tel', 'l_vent', 'l_vthal', ]
    config_ranges['rnn_type'] = ['Transformer', 'RNN', ]
    
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def latent_models():
    config = NeuralPredictionConfig()
    config.experiment_name = 'latent_models'
    config.loss_mode = 'autoregressive'
    config.model_type = 'LatentModel'
    config.sampling_rate = 1

    config_ranges = OrderedDict()
    config_ranges['target_mode'] = ['raw', ]
    config_ranges['normalize_mode'] = ['none', ]
    config_ranges['shared_backbone'] = [True,  ]
    config_ranges['tf_interval'] = [1, 3, 5, 10, 25, ]
    config_ranges['rnn_type'] = ['PLRNN', 'CTRNNCell', ]
    
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def latent_models_average():
    config = NeuralPredictionConfig()
    config.experiment_name = 'latent_models_average'
    config.loss_mode = 'autoregressive'
    config.model_type = 'LatentModel'
    config.brain_regions = 'average'
    config.sampling_rate = 1
    config.pc_dim = None
    config.tf_interval = 5

    config_ranges = OrderedDict()
    config_ranges['target_mode'] = ['raw', ]
    config_ranges['normalize_mode'] = ['zscore', ]
    config_ranges['shared_backbone'] = [True,  ]
    config_ranges['hidden_size'] = [64, 128, 512, ]
    config_ranges['rnn_type'] = ['PLRNN', 'CTRNNCell', ]
    
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def latent_models_individual_region():
    config = NeuralPredictionConfig()
    config.experiment_name = 'latent_models_individual_region'
    config.loss_mode = 'autoregressive'
    config.model_type = 'LatentModel'
    config.sampling_rate = 1

    config_ranges = OrderedDict()
    config_ranges['target_mode'] = ['raw', ]
    config_ranges['normalize_mode'] = ['zscore', ]
    config_ranges['shared_backbone'] = [True,  ]
    config_ranges['brain_regions'] = ['l_LHb', 'l_MHb', 'l_ctel', 'l_dthal', 'l_gc', 'l_raphe', 'l_tel', 'l_vent', 'l_vthal', ]
    config_ranges['rnn_type'] = ['Transformer', 'RNN', ]
    
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def autoregressive_rnns_visual_with_stimuli():
    config = NeuralPredictionConfig()
    config.experiment_name = 'autoregressive_rnns_visual_with_stimuli'
    config.loss_mode = 'autoregressive'
    config.dataset = 'zebrafish_visual'

    config_ranges = OrderedDict()
    config_ranges['use_stimuli'] = [True, False, ]
    config_ranges['use_eye_movements'] = [True, False, ]
    config_ranges['target_mode'] = ['raw', ]
    config_ranges['shared_backbone'] = [True, ]
    config_ranges['rnn_type'] = ['Transformer', 'S4', 'LSTM', 'RNN', ]
    
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            if config.use_eye_movements:
                config.use_motor = True
            config.stimuli_dim = 24 * config.use_stimuli

    return configs

def autoregressive_rnns_visual_with_stimuli_individual():
    config = NeuralPredictionConfig()
    config.experiment_name = 'autoregressive_rnns_visual_with_stimuli_individual'
    config.loss_mode = 'autoregressive'
    config.dataset = 'zebrafish_visual'
    config.animal_ids = [1]
    config.patch_length = 2880
    config.train_split = 5 / 6
    config.val_split = 1 / 6

    config_ranges = OrderedDict()
    config_ranges['use_stimuli'] = [True, False, ]
    config_ranges['use_eye_movements'] = [True, False, ]
    config_ranges['target_mode'] = ['raw', ]
    config_ranges['shared_backbone'] = [True, ]
    config_ranges['rnn_type'] = ['Transformer', 'S4', 'LSTM', 'RNN', ]
    
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            if config.use_eye_movements:
                config.use_motor = True
            config.stimuli_dim = 24 * config.use_stimuli

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

def autoregressive_rnns_sim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'sim_compare_train_length'
    config.loss_mode = 'autoregressive'
    config.dataset = 'simulation'
    config.pc_dim = None
    config.num_workers = 0
    config.sampling_rate = 5

    config_ranges = OrderedDict()
    config_ranges['n_regions'] = [1, ]
    config_ranges['n_neurons'] = [512, ]
    config_ranges['rnn_type'] = ['Transformer', 'S4', 'LSTM', 'RNN', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            if config.rnn_type in ['S4', 'Transformer']:
                config.num_layers = 4
    return configs

def latent_model_sim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'latent_model_sim'
    config.loss_mode = 'autoregressive'
    config.dataset = 'simulation'
    config.pc_dim = None
    config.model_type = 'LatentModel'

    config_ranges = OrderedDict()
    config_ranges['n_neurons'] = [512, ]
    config_ranges['tf_interval'] = [1, 3, 5, 10, 25]
    config_ranges['rnn_type'] = ['PLRNN', 'CTRNNCell']

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

def seqvae_test():
    config = NeuralPredictionConfig()
    config.experiment_name = 'seqvae_test'
    config.model_type = 'SeqVAE'

    config.encoder_hidden_size = 128
    config.decoder_hidden_size = 32
    config.kl_loss_coef = 0.1
    config.animal_ids = [0]
    config.exp_types = ['control']
    config.loss_mode = 'recontruction_prediction'

    config_ranges = OrderedDict()
    config_ranges['target_mode'] = ['raw', ]
    config_ranges['shared_backbone'] = [True, False, ]
    config_ranges['kl_loss_coef'] = [0.1, 0.3]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def seqvae_sim_test():
    config = NeuralPredictionConfig()
    config.experiment_name = 'seqvae_sim_test'
    config.model_type = 'SeqVAE'
    config.dataset = 'simulation'

    config.encoder_hidden_size = 128
    config.decoder_hidden_size = 200
    config.kl_loss_coef = 0.1
    config.loss_mode = 'recontruction_prediction'

    config_ranges = OrderedDict()
    config_ranges['n_neurons'] = [200, 500, 1500, 3000, ]
    config_ranges['kl_loss_coef'] = [0.1, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
    return configs

def linear_baselines():
    config = NeuralPredictionConfig()
    config.experiment_name = 'linear_baselines'

    config.wdecay = 1e-6
    config.lr = 1e-3
    config.model_type = 'Linear'
    config.loss_mode = 'prediction'

    config_ranges = OrderedDict()
    config_ranges['normalize_mode'] = ['none', ]
    config_ranges['per_channel'] = [False, True, ]
    config_ranges['linear_input_length'] = [1, 2, 3, 5, 10, 20, 40]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            config.shared_backbone = not config.per_channel

    return configs

def low_rank_rnn():
    config = NeuralPredictionConfig()
    config.experiment_name = 'low_rank_rnn'
    config.model_type = 'LatentModel'
    config.rnn_type = 'LRRNN'
    config.loss_mode = 'autoregressive'
    config.latent_to_ob = 'identity'
    config.exp_types = ['control']
    
    config_ranges = OrderedDict()
    # config_ranges['animal_ids'] = [[x] for x in range(5)]
    config_ranges['rnn_rank'] = [None, ]
    config_ranges['rnn_alpha'] = [0.05, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
    return configs

def vqvae():
    config = NeuralPredictionConfig()
    config.experiment_name = 'vqvae'
    config.model_type = 'VQVAE'
    config.pred_length = 0
    config.loss_mode = 'reconstruction'
    config.seq_length = 48
    config.save_every = config.max_batch

    config_ranges = OrderedDict()
    config_ranges['compression_factor'] = [4, 8, 16, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def vqvae_average():
    config = NeuralPredictionConfig()
    config.experiment_name = 'vqvae_average'
    config.model_type = 'VQVAE'
    config.pred_length = 0
    config.loss_mode = 'reconstruction'
    config.seq_length = 48
    config.save_every = config.max_batch
    config.brain_regions = 'average'
    config.pc_dim = None
    config.normalize_mode = 'zscore'

    config_ranges = OrderedDict()
    config_ranges['compression_factor'] = [4, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def vqvae_sim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'vqvae_sim'
    config.model_type = 'VQVAE'
    config.pred_length = 0
    config.loss_mode = 'reconstruction'
    config.seq_length = 48
    config.save_every = config.max_batch
    config.dataset = 'simulation'
    config.pc_dim = None
    config.sampling_rate = 5
    config.mem = 64

    config_ranges = OrderedDict()
    config_ranges['n_neurons'] = [512, 1536, ]
    config_ranges['train_data_length'] = [2048, 8192, ]
    config_ranges['compression_factor'] = [4, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    return configs

def vqvae_decode():
    config = NeuralPredictionConfig()
    config.experiment_name = 'vqvae_decode'
    config.model_type = 'Decoder'
    config.encoder_state_dict_file = 'net_5000.pth'
    config.loss_mode = 'prediction'

    config_ranges = OrderedDict()
    config_ranges['decoder_type'] = ['Transformer', 'MLP', ]
    config_ranges['compression_factor'] = [4, 8, 16, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            config.encoder_dir = f'experiments/vqvae/model_compression_factor{config.compression_factor}_s{seed}'

    return configs

def configure_models(configs: dict):
    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            config.shared_backbone = True
            label = config.model_label

            if label[:5] == 'TOTEM':
                config.model_type = 'Decoder'
                config.encoder_state_dict_file = 'net_5000.pth'
                config.loss_mode = 'prediction'
                if config.dataset == 'zebrafish' and config.brain_regions == 'average':
                    config.encoder_dir = \
                        f'experiments/vqvae_average/model_compression_factor{config.compression_factor}_s{seed}'
                elif config.dataset == 'zebrafish':
                    config.encoder_dir = \
                        f'experiments/vqvae/model_compression_factor{config.compression_factor}_s{seed}'
                elif config.dataset == 'simulation':
                    if config.train_data_length > 327680 // config.sampling_rate:
                        config.train_data_length = 327680 // config.sampling_rate
                    config.encoder_dir = \
                        f'experiments/vqvae_sim/model_n_neurons{config.n_neurons}_train_data_length{config.train_data_length}_compression_factor{config.compression_factor}_s{seed}'
                else:
                    raise ValueError(f'Unknown dataset: {config.dataset}')
                config.decoder_type = label[6:]
            elif label[:2] == 'AR':
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
            elif label == 'Linear':
                config.loss_mode = 'prediction'
                config.model_type = 'Linear'
                config.per_channel = True
                config.shared_backbone = False
                config.linear_input_length = config.seq_length - config.pred_length
            elif label == 'TCN':
                config.loss_mode = 'prediction'
                config.model_type = 'TCN'
                config.shared_backbone = config.pc_dim is not None
            else:
                raise ValueError(f'Unknown model label: {label}')

    return configs  

model_list = [
    'TOTEM_Transformer', 'TOTEM_MLP', 
    'AR_Transformer', 'AR_S4', 'AR_RNN', 'AR_LSTM',
    'Latent_PLRNN', 'Latent_LRRNN_4', 'Latent_CTRNN',
    'Linear', 'TCN'
]

def pc_compare_models():
    config = NeuralPredictionConfig()
    config.experiment_name = 'pc_compare_models'
    config.dataset = 'zebrafish'

    config_ranges = OrderedDict()
    config_ranges['model_label'] = model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    return configs

def pc_compare_models_individual():
    config = NeuralPredictionConfig()
    config.experiment_name = 'pc_compare_models_individual'
    config.dataset = 'zebrafish'
    config.exp_types = ['control']
    config.max_batch = 2500

    config_ranges = OrderedDict()
    config_ranges['animals_ids'] = [[x] for x in range(5)]
    config_ranges['model_label'] = model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    return configs

def compare_models_average():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_average'
    config.dataset = 'zebrafish'
    config.brain_regions = 'average'
    config.pc_dim = None
    config.normalize_mode = 'zscore'

    config_ranges = OrderedDict()
    config_ranges['model_label'] = model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    return configs

def compare_models_individual_region():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_individual_region'
    config.dataset = 'zebrafish'
    config.pc_dim = None
    config.normalize_mode = 'zscore'

    config_ranges = OrderedDict()
    config_ranges['brain_regions'] = ['l_LHb', 'l_MHb', 'l_dthal', 'l_gc', 'l_raphe', 'l_vent', 'l_vthal', ]
    config_ranges['model_label'] = model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    return configs

def sim_compare_models():
    config = NeuralPredictionConfig()
    config.experiment_name = 'sim_compare_models'
    config.dataset = 'simulation'
    config.pc_dim = None
    config.sampling_rate = 5
    config.mem = 128

    config_ranges = OrderedDict()
    config_ranges['n_neurons'] = [256, 512, ]
    config_ranges['train_data_length'] = [65536, ]
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