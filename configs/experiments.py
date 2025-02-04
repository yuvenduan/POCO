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

single_session_model_list = ['POYO', 'PaiFilter', 'Linear', 'DLinear', 'TCN', 'AR_Transformer', 'Latent_PLRNN', ]
multi_session_model_list = ['POYO', 'Linear', 'Latent_PLRNN', 'MultiAR_Transformer', ]
single_neuron_model_list = ['POYO', 'Linear', 'DLinear']

def poyo_compare_embedding_mode():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_embedding_mode'
    config.model_label = 'POYO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ]
    config_ranges['unit_embedding_components'] = [[], ['session'], ['session', 'unit_type'], ]
    config_ranges['latent_session_embedding'] = [True, False, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multi_species():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multi_species'
    config.dataset_label = [
        'zebrafish', 'celegans', 'mice', 
        'zebrafish_pc', 'celegans_pc', 'mice_pc',
    ]
    config.max_batch = 20000
    config_ranges = OrderedDict()
    config_ranges['model_label'] = ['POYO', 'Linear', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_zebrafish_pc_single_session():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_zebrafish_single_session'
    config.max_batch = 5000

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = [f'zebrafish_pc-{session}' for session in range(19)]
    config_ranges['model_label'] = single_session_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_celegans_single_session():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_celegans_single_session'
    config.max_batch = 5000

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = [f'celegans-{session}' for session in range(5)] + [f'celegans_pc-{session}' for session in range(5)]
    config_ranges['model_label'] = single_session_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_mice_single_session():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_mice_single_session'
    config.max_batch = 5000

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = [f'mice-{session}' for session in range(12)] + [f'mice_pc-{session}' for session in range(12)]
    config_ranges['model_label'] = single_session_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multi_session():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multi_session'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['zebrafish_pc', 'celegans', 'celegans_pc', 'mice', 'mice_pc', ]
    config_ranges['model_label'] = multi_session_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_zebrafish_single_neuron():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_zebrafish_single_neuron'
    config.max_batch = 20000

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['zebrafish', ]
    config_ranges['model_label'] = single_neuron_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_sim_multi_session():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_sim_multi_session'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = [f'sim_{n}' for n in [128, 512]]
    config_ranges['model_label'] = multi_session_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_sim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_sim'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = [f'sim_{n}-{seed}' for n in [128, 512] for seed in range(4)]
    config_ranges['model_label'] = single_session_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

selected_model_list = ['Linear', 'Latent_PLRNN', 'AR_Transformer', 'TCN', 'POYO']

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
    config_ranges['dataset_label'] = [f'sim_{n}' for n in [64, 128, 256, 384, 512, 1024, 1536, ]]
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