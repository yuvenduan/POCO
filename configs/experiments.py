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

def poyo_test():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multi_session'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['zebrafish_pc', 'zebrafishahrens_pc', 'celegans', 'celegansflavell', 'mice', ]
    config_ranges['model_label'] = ['POYO', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
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
    config.mu_std_separate_projs = False

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['lowpass', 'none']
    config_ranges['dataset_label'] = dataset_list
    config_ranges['mu_module_mode'] = ['last_combined', 'last', 'combined', 'original', ]
    config_ranges['std_module_mode'] = ['combined', 'combined_softplus', 'combined_exp', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2, )
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def poyo_compare_conditioning():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_conditioning'
    config.model_label = 'POYO'
    config.conditioning = 'mlp'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', ]
    config_ranges['dataset_label'] = dataset_list + large_dataset_list
    config_ranges['conditioning_dim'] = [0, 128, 1024]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2, )
    configs = configure_models(configs)
    configs = configure_dataset(configs)

    for seed in configs.keys():
        for config in configs[seed]:
            config: NeuralPredictionConfig
            if config.conditioning_dim == 0:
                config.conditioning = 'none'
    return configs

single_session_model_list = ['POYO', 'Linear', 'Latent_PLRNN', 'PaiFilter', 'TexFilter', 'AR_Transformer', 'DLinear', 'TCN', 'TSMixer', ]
multi_session_model_list = ['POYO', 'Linear', 'Latent_PLRNN', 'TexFilter', 'PaiFilter', 'MultiAR_Transformer', 'MLP', ]
single_neuron_model_list = ['POYO', 'Linear', 'MLP', 'PaiFilter', 'TexFilter',  ]

# POYO experiments: poyo_compare_embedding_mode poyo_compare_compression_factor poyo_compare_num_latents poyo_compare_hidden_size poyo_compare_num_layers poyo_compare_num_heads
# core experiments (single session): compare_models_zebrafish_pc_single_session compare_models_celegans_single_session compare_models_mice_single_session compare_models_zebrafish_ahrens_single_session compare_models_celegans_flavell_single_session
# core experiments (others): compare_models_multi_session compare_models_multi_species compare_models_sim_multi_session compare_models_sim compare_models_zebrafish_single_neuron

def poyo_compare_embedding_mode():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_embedding_mode'
    config.model_label = 'POYO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ]
    config_ranges['unit_embedding_components'] = [[], ['session'], ['session', 'unit_type'], ]
    config_ranges['latent_session_embedding'] = [False, True, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def poyo_compare_compression_factor():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_compression_factor'
    config.model_label = 'POYO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ]
    config_ranges['compression_factor'] = [4, 8, 16, 24, 48, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def poyo_compare_num_latents():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_num_latents'
    config.model_label = 'POYO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ]
    config_ranges['poyo_num_latents'] = [1, 2, 4, 8, 16, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_hidden_size():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_hidden_size'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = dataset_list
    config_ranges['model_label'] = ['MLP', 'POYO', ]
    config_ranges['decoder_hidden_size'] = [32, 128, 256, 512, 1024, 1536, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_context_window_length():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_context_window_length'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = dataset_list
    config_ranges['model_label'] = ['POYO', 'Linear', 'MLP']
    config_ranges['seq_length'] = [config.pred_length + x for x in [2, 4, 16, 48]]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)

    for seed in configs.keys():
        for config in configs[seed]:
            config: NeuralPredictionConfig
            config.compression_factor = config.seq_length - config.pred_length

    return configs

def poyo_compare_num_layers():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_num_layers'
    config.model_label = 'POYO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ]
    config_ranges['decoder_num_layers'] = [1, 2, 4, 8, 12, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def poyo_compare_num_heads():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_num_heads'
    config.model_label = 'POYO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['celegans', 'zebrafish', 'mice', 'zebrafish_pc', 'celegans_pc', 'mice_pc', ]
    config_ranges['decoder_num_heads'] = [1, 2, 4, 8, 16, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multi_species():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multi_species'
    config.dataset_label = [
        'zebrafish', 'celegans', 'celegansflavell', 'mice', 
        'zebrafish_pc', 'celegans_pc', 'celegansflavell_pc', 'mice_pc',
    ]
    config.max_batch = 40000
    config_ranges = OrderedDict()
    config_ranges['model_label'] = ['POYO', 'Linear', ]
    config_ranges['log_loss'] = [False, ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def zebrafish_multi_datasets():
    config = NeuralPredictionConfig()
    config.experiment_name = 'zebrafish_multi_datasets'
    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = [
        ['zebrafish_pc', 'zebrafishahrens_pc', 'zebrafishjain_pc'], 
        ['zebrafish_pc', 'zebrafishahrens_pc', ],
    ]
    config_ranges['model_label'] = ['POYO', ]
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

def compare_models_celegans_flavell_single_session():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_celegans_flavell_single_session'
    config.max_batch = 5000

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = [f'celegansflavell-{session}' for session in range(40)]
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

def compare_models_zebrafish_ahrens_single_session():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_zebrafish_ahrens_single_session'
    config.max_batch = 5000

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = [f'zebrafishahrens_pc-{session}' for session in range(15)]
    config_ranges['model_label'] = single_session_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

dataset_list = ['zebrafish_pc', 'zebrafishahrens_pc', 'celegansflavell', 'celegans', 'mice', 'mice_pc', ]
large_dataset_list = ['zebrafishahrens', 'zebrafish', ]

def compare_dataset_filter():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_dataset_filter'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['lowpass', 'highpass', 'bandpass', 'none']
    config_ranges['dataset_label'] = dataset_list
    config_ranges['model_label'] = multi_session_model_list

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multi_session():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multi_session'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = dataset_list
    config_ranges['model_label'] = multi_session_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def create_splits(n_sessions, n_split_list, dataset_name):
    splits = []
    for n_split in n_split_list:
        if n_split == 1:
            splits.append(f'{dataset_name}-0-{n_sessions}')
            continue
        elif n_split == n_sessions:
            for i in range(n_sessions):
                splits.append(f'{dataset_name}-{i}')
            continue
        split_size = n_sessions // n_split
        residual = n_sessions - split_size * n_split
        start = 0
        for i in range(n_split):
            end = start + split_size + (1 if i < residual else 0)
            splits.append(f'{dataset_name}-{start}-{end}')
            start = end
    return splits

def compare_models_multiple_splits_celegans_flavell():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multiple_splits'
    config.max_batch = 10000

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = \
        create_splits(40, [1, 2, 3, 5, 8, 12, 20, 40], 'celegansflavell') + \
        create_splits(40, [1, 2, 3, 5, 8, 12, 20, 40], 'celegansflavell_pc')
    config_ranges['model_label'] = ['POYO', 'Linear', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multiple_splits_mice():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multiple_splits'
    config.max_batch = 10000

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = \
        create_splits(12, [1, 2, 3, 4, 6, 12], 'mice') + \
        create_splits(12, [1, 2, 3, 4, 6, 12], 'mice_pc')
    config_ranges['model_label'] = ['POYO', 'Linear', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multiple_splits_zebrafish():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multiple_splits'
    config.max_batch = 10000

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = \
        create_splits(19, [1, 2, 3, 5, 10, 19], 'zebrafish') + \
        create_splits(19, [1, 2, 3, 5, 10, 19], 'zebrafish_pc')
    config_ranges['model_label'] = ['POYO', 'Linear', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_train_length_celegans_flavell():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_train_length'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['celegansflavell', 'celegansflavell_pc', ]
    config_ranges['model_label'] = ['POYO', 'Linear', ]
    config_ranges['train_data_length'] = [128, 256, 512, 768, 1024, 1536, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_train_length_mice():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_train_length'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['mice', 'mice_pc', ]
    config_ranges['model_label'] = ['POYO', 'Linear', ]
    config_ranges['train_data_length'] = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_train_length_zebrafish():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_train_length'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['zebrafish', 'zebrafish_pc', ]
    config_ranges['model_label'] = ['POYO', 'Linear', ]
    config_ranges['train_data_length'] = [128, 256, 512, 1024, 1536, 2048, 3072]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_zebrafish_single_neuron(): # need h100
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_zebrafish_single_neuron'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = large_dataset_list
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

def basemodels(): # train the base model on some sessions/datasets, used for finetuning
    config = NeuralPredictionConfig()
    config.experiment_name = 'basemodels'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = [
        ['zebrafish_pc-*', ],
        ['zebrafishahrens_pc-0-12', ],
        ['zebrafish_pc', 'zebrafishahrens_pc-0-12', ],
        ['zebrafish_pc-*', 'zebrafishahrens_pc', ],
    ]
    config_ranges['model_label'] = ['POYO', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def finetuning(): # compare finetuning (all), only train embedding, train from scratch, linear train from scratch
    config = NeuralPredictionConfig()
    config.experiment_name = 'finetuning'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['celegansflavell', 'celegansflavell_pc', 'zebrafish_pc', 'zebrafishahrens_pc', 'celegans', 'celegans_pc', 'mice', 'mice_pc', ]
    config_ranges['model_label'] = ['POYO', 'Linear', ]
    config_ranges['finetuning'] = [True, False]

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