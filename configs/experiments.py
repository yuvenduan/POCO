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
    config.experiment_name = 'linear_baselines_test'

    config.wdecay = 1e-6
    config.lr = 1e-3
    config.model_type = 'Linear'
    config.loss_mode = 'prediction'

    config_ranges = OrderedDict()
    config_ranges['normalize_mode'] = ['none', ]
    config_ranges['per_channel'] = [True, ]
    config_ranges['linear_input_length'] = [48, ]
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

def direct_decode():
    config = NeuralPredictionConfig()
    config.experiment_name = 'direct_decode'
    config.model_type = 'Decoder'
    config.encoder_dir = None
    config.loss_mode = 'prediction'
    config.separate_projs = True

    config_ranges = OrderedDict()
    config_ranges['mu_std_loss_coef'] = [None, 0, 1, ]
    config_ranges['decoder_type'] = ['Transformer', 'MLP', 'Linear', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1, )
    return configs

def poyo_test():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_test'
    config.model_type = 'Decoder'
    config.encoder_dir = None
    config.loss_mode = 'prediction'
    config.separate_projs = False
    config.decoder_type = 'POYO'

    config_ranges = OrderedDict()
    config_ranges['compression_factor'] = [1, 8, 48, ]
    config_ranges['mu_std_loss_coef'] = [None, 0, 1, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1, )
    return configs

def poyo_test_multi_query():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_test_multi_query'
    config.model_type = 'Decoder'
    config.encoder_dir = None
    config.loss_mode = 'prediction'
    config.decoder_type = 'POYO'
    config.dataset_label = 'spontaneous_pc'

    config_ranges = OrderedDict()
    config_ranges['poyo_query_mode'] = ['single', 'multi']
    config_ranges['compression_factor'] = [4, 16, ]
    config_ranges['model_label'] = ['POYO', 'POYO_TOTEM', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
    configs = configure_models(configs)
    return configs

def poyo_compare_params():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_params'
    config.model_type = 'Decoder'
    config.encoder_dir = None
    config.loss_mode = 'prediction'
    config.decoder_type = 'POYO'
    config.dataset_label = 'spontaneous_pc'

    config_ranges = OrderedDict()
    config_ranges['compression_factor'] = [4, 16, ]
    config_ranges['poyo_num_latents'] = [4, 16, 64, ]
    config_ranges['separate_projs'] = [True, False, ]
    config_ranges['model_label'] = ['POYO_TOTEM', 'POYO', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    return configs

def compare_hidden_size():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_hidden_size'
    config.model_type = 'Decoder'
    config.encoder_dir = None
    config.loss_mode = 'prediction'
    config.mu_std_loss_coef = 0
    config.show_chance = False

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous', 'spontaneous_pc', ]
    config_ranges['model_label'] = ['POYO', 'Transformer', 'Linear',  ]
    config_ranges['decoder_hidden_size'] = [64, 256, 1024, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1, )
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_num_layers():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_num_layers'
    config.model_type = 'Decoder'
    config.encoder_dir = None
    config.loss_mode = 'prediction'
    config.mu_std_loss_coef = 0
    config.show_chance = False

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous', 'spontaneous_pc', ]
    config_ranges['model_label'] = ['Transformer', 'POYO',  ]
    config_ranges['decoder_num_layers'] = [1, 2, 4, 8, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1, )
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
    'POYO', 'Linear', 'MLP', 'POYO_TOTEM', 
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

def pc_compare_models():
    config = NeuralPredictionConfig()
    config.experiment_name = 'pc_compare_models'
    config.dataset = 'zebrafish'

    config_ranges = OrderedDict()
    config_ranges['model_label'] = model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    return configs

def pc_compare_models_visual():
    config = NeuralPredictionConfig()
    config.experiment_name = 'pc_compare_models_visual'
    config.dataset = 'zebrafish_visual'
    config.use_stimuli = False
    config.use_eye_movements = False

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