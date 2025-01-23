"""Experiments and corresponding analysis.
format adapted from https://github.com/gyyang/olfaction_evolution

Each experiment is described by a function that returns a list of configurations
function name is the experiment name
"""

from collections import OrderedDict
from configs.configs import BaseConfig, NeuralPredictionConfig
from configs.config_global import EXP_TYPES
from utils.config_utils import vary_config
from configs.configure_model_datasets import configure_models, configure_dataset

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
    config.ga = 1.8
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
    config_ranges['poyo_num_latents'] = [4, 16, ]

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

def poyo_ablations():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_ablations'
    config.dataset_label = 'spontaneous_pc'

    config_ranges = OrderedDict()
    config_ranges['model_label'] = ['POYO', 'POYO', 'POYO', 'POYO', 'Transformer', 'Transformer_pop', ]
    config_ranges['poyo_output_mode'] = ['latent', 'latent']  + ['query'] * 4
    config_ranges['separate_projs'] = [True, False, True, False, True, True, ]

    configs = vary_config(config, config_ranges, mode='sequential', num_seed=2, )
    configs = configure_models(configs, customize=True)
    configs = configure_dataset(configs)
    return configs

def poyo_compare_mu_std_module():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_mu_std_module'
    config.model_label = 'POYO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_pc', ]
    config_ranges['mu_std_module_mode'] = ['combined_mu_only', 'combined', 'original', 'learned', 'none', ]
    config_ranges['normalize_input'] = [True, False]
    config_ranges['mu_std_separate_projs'] = [True, False]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2, )
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def poyo_compare_tmax(): # 'multi_m' and 'multi' 
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_tmax'
    config.model_label = 'POYO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_pc', ]
    config_ranges['poyo_query_mode'] = ['multi_m', 'multi', 'single', ]
    config_ranges['rotary_attention_tmax'] = [4, 20, 100, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2, )
    configs = configure_models(configs)
    configs = configure_dataset(configs)

    for seed, config_list in configs.items():
        for cfg in config_list:
            if cfg.poyo_query_mode == 'multi_m':
                cfg.poyo_query_mode = 'multi'
                cfg.separate_projs = True

    return configs

def poyo_compare_mu_std_module_neuron():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_mu_std_module'
    config.model_label = 'POYO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_pc', ]
    config_ranges['mu_std_module_mode'] = ['combined', 'original', 'learned', 'none', ]
    config_ranges['normalize_input'] = [True, False]
    config_ranges['mu_std_separate_projs'] = [True, False]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2, )
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def linear_test():
    config = NeuralPredictionConfig()
    config.experiment_name = 'linear_test'
    config.model_label = 'Linear'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_pc', 'spontaneous', ]
    config_ranges['mu_std_module_mode'] = ['none', 'combined', ]
    config_ranges['normalize_input'] = [True, False, ]
    config_ranges['mu_std_separate_projs'] = [True, False, ]

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

"""
model_list = [
    # 'POYO_TOTEM', 'Linear_TOTEM', 'MLP_TOTEM', 'Transformer_TOTEM',
    'POYO', 'Linear', 'MLP', 'Transformer', 'TCN',
    'AR_Transformer', 'AR_S4', 'AR_RNN', 'AR_LSTM',
    'Latent_PLRNN', 'Latent_LRRNN_4', 'Latent_CTRNN', 
]
"""
model_list = ['Linear', 'Latent_PLRNN', 'AR_Transformer', 'TCN', 'POYO']

def compare_models_pc():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_pc', ]
    config_ranges['model_label'] = model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_fc():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_fc', ]
    config_ranges['model_label'] = model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_long_time_scale():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_long_time_scale'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_pc', 'visual_pc', ]
    config_ranges['sampling_freq'] = [2, 5, ]
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
    config_ranges['normalize_input'] = [True, ]
    config_ranges['model_label'] = model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def conv_encoder_test():
    config = NeuralPredictionConfig()
    config.experiment_name = 'conv_encoder_test'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous', 'spontaneous_pc',  ]
    config_ranges['model_label'] = ['POYO_cnn', 'Linear_cnn', ]
    config_ranges['conv_channels'] = [64, 512, ]
    config_ranges['conv_stride'] = [4, 16, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_population_token_dim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_population_token_dim'
    config.population_token = True

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_pc', ]
    config_ranges['compression_factor'] = [4, 16]
    config_ranges['population_token_dim'] = [128, 512, ]
    config_ranges['model_label'] = ['Transformer', 'Linear', 'MLP']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

single_neuron_model_list = [
    'POYO', 'Linear',
]

def compare_models_single_neuron():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous', 'stim', ]
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
    config.dataset_label = 'spontaneous'

    config_ranges = OrderedDict()
    config_ranges['brain_regions'] = ['l_LHb', 'l_MHb', 'l_dthal', 'l_gc', 'l_raphe', 'l_vent', 'l_vthal', ]
    config_ranges['model_label'] = selected_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

selected_model_list = ['Linear', 'Latent_PLRNN', 'AR_Transformer', 'TCN', 'POYO']

def compare_models_sim_pc():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_sim_pc'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['sim_pc512', 'sim_pc1536', ]
    config_ranges['model_label'] = selected_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_train_length_pc():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_train_length_pc'
    
    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['spontaneous_pc', ]
    config_ranges['model_label'] = selected_model_list
    config_ranges['train_data_length'] = [128, 256, 512, 1024, 2048, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_train_length_single_neuron():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_train_length_single_neuron'
    
    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['celegans', ]
    config_ranges['model_label'] = selected_model_list
    config_ranges['train_data_length'] = [128, 256, 512, 1024, 2048, 3000, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_train_length_sim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_train_length_sim'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = [f'sim{n}' for n in [64, 128, 256, 384, 512, 1024, 1536, ]]
    config_ranges['model_label'] = ['Linear', 'Latent_PLRNN', 'POYO']
    config_ranges['train_data_length'] = [256, 512, 1024, 2048, 4096, 16384, 32768, 65536, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def per_animal_pc():
    config = NeuralPredictionConfig()
    config.experiment_name = 'per_animal_pc'
    config.dataset_label = 'spontaneous_pc'

    config_ranges = OrderedDict()
    config_ranges['model_label'] = ['Linear', 'Latent_PLRNN', 'POYO']
    config_ranges['id'] = range(19)

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            config.animal_ids = [config.id % 5]
            config.exp_types = [EXP_TYPES[config.id // 5]]

    return configs

def multi_animal_pc():
    config = NeuralPredictionConfig()
    config.experiment_name = 'multi_animal_pc'
    config.dataset_label = 'spontaneous_pc'

    config_ranges = OrderedDict()
    config_ranges['model_label'] = ['Linear', 'Latent_PLRNN', 'POYO']
    config_ranges['animal_ids'] = [[0], [1], [2], [3], [4], [0, 1], [2, 3, 4], 'all']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def multi_animal_single_neuron():
    config = NeuralPredictionConfig()
    config.experiment_name = 'multi_animal_single_neuron'
    config.dataset_label = 'celegans'

    config_ranges = OrderedDict()
    config_ranges['model_label'] = ['Linear', 'Latent_PLRNN', 'POYO']
    config_ranges['animal_ids'] = [[0], [1], [2], [3], [4], [0, 1], [2, 3, 4], 'all']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def multi_cohorts_pc():
    config = NeuralPredictionConfig()
    config.experiment_name = 'multi_cohorts_pc'
    config.dataset_label = 'spontaneous_pc'

    config_ranges = OrderedDict()
    config_ranges['model_label'] = ['Linear', 'Latent_PLRNN', 'POYO', ]
    config_ranges['exp_type_id'] = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            config.exp_types = [EXP_TYPES[i] for i in config.exp_type_id]
    return configs

def compare_pc_dim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_pc_dim'
    config.dataset_label = 'spontaneous_pc'
    config.dataset = 'zebrafish'

    config_ranges = OrderedDict()
    config_ranges['model_label'] = ['Linear', 'Latent_PLRNN', 'POYO']
    config_ranges['pc_dim'] = [1, 4, 32, 128, 512, 2048, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    return configs

def compare_models_stim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_stim'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['stim_avg', 'stim_pc', ]
    config_ranges['model_label'] = selected_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_celegans():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_c_elegans'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['celegans', 'celegans_pc', ]
    config_ranges['model_label'] = selected_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_mice():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_mice'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['mice', 'mice_pc', ]
    config_ranges['model_label'] = selected_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs