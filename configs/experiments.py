"""Experiments and corresponding analysis.
format adapted from https://github.com/gyyang/olfaction_evolution

Each experiment is described by a function that returns a list of configurations
function name is the experiment name
"""

import os
from collections import OrderedDict
from configs.configs import BaseConfig, NeuralPredictionConfig
from configs.config_global import EXP_TYPES
from utils.config_utils import vary_config
from configs.configure_model_datasets import configure_models, configure_dataset

def all_experiments():
    # only a sanity check that all functions return configs correctly
    # single session models
    compare_models_zebrafish_pc_single_session()
    compare_models_celegans_single_session()
    compare_models_mice_single_session()
    compare_models_celegans_flavell_single_session()
    compare_models_zebrafish_ahrens_single_session()

    # multi session models
    compare_models_zebrafish_single_neuron()
    compare_models_multi_session()
    compare_models_multi_species()
    zebrafish_multi_datasets()

    # simulations
    compare_models_sim_multi_session()
    compare_models_sim()
    compare_models_sim_all()

    # compare number of sessions / training length
    compare_models_multiple_splits_celegans_flavell()
    compare_models_multiple_splits_mice()
    compare_models_multiple_splits_zebrafish()
    compare_train_length_celegans_flavell()
    compare_train_length_mice()
    compare_train_length_zebrafish()

    # compare context window / dataset filter
    compare_context_window_length()
    compare_dataset_filter()
    compare_large_dataset_filter()

    # finetuning
    basemodels()
    finetuning()
    compare_pretraining_dataset()

    # try different model params
    poyo_compare_compression_factor()
    poyo_compare_num_latents()
    compare_hidden_size()
    poyo_compare_num_layers()
    ablations()

def poco_test(): # sanity check: different versions of poco should give the same result
    config = NeuralPredictionConfig()
    config.experiment_name = 'test'
    config.max_batch = 1000

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['zebrafishahrens']
    config_ranges['model_label'] = ['POCOtest', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def dataset_test(): # just test that dataset loads ...
    config = NeuralPredictionConfig()
    config.experiment_name = 'dataset_test'
    config.max_batch = 0

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = [['zebrafish', 'zebrafishahrens', 'celegans', 'celegansflavell', 'mice', ]]
    config_ranges['model_label'] = ['NLinear']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def poyo_compare_conditioning():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_conditioning'
    config.model_label = 'POCO'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', ]
    config_ranges['dataset_label'] = dataset_list + large_dataset_list
    config_ranges['conditioning_dim'] = [0, 4, 16, 128, 1024]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2, )

    for seed in configs.keys():
        for config in configs[seed]:
            config: NeuralPredictionConfig
            if config.conditioning_dim == 0:
                config.model_label = 'POYO'

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

single_session_model_list = ['POCO', 'MLP', 'NLinear', 'Latent_PLRNN', 'TexFilter', 'NetFormer', 'AR_Transformer', 'DLinear', 'TCN', 'TSMixer', ]
multi_session_model_list = ['POCO', 'MLP', 'NLinear', 'Latent_PLRNN', 'TexFilter', 'NetFormer', 'MultiAR_Transformer', ]
single_neuron_model_list = ['POCO', 'MLP', 'NLinear', 'TexFilter',  ]

dataset_list = ['zebrafish_pc', 'zebrafishahrens_pc', 'celegansflavell', 'celegans', 'mice', 'mice_pc', ]
large_dataset_list = ['zebrafishahrens', 'zebrafish', ]

# POYO experiments: poyo_compare_compression_factor poyo_compare_num_latents compare_hidden_size poyo_compare_num_layers
# core experiments (single session): compare_models_zebrafish_pc_single_session compare_models_celegans_single_session compare_models_mice_single_session compare_models_zebrafish_ahrens_single_session compare_models_celegans_flavell_single_session
# core experiments (others): compare_models_multi_session compare_models_multi_species zebrafish_multi_datasets compare_models_zebrafish_single_neuron compare_models_sim_multi_session compare_models_sim
# compare dataset size: compare_models_multiple_splits_celegans_flavell compare_models_multiple_splits_mice compare_models_multiple_splits_zebrafish compare_train_length_celegans_flavell compare_train_length_mice compare_train_length_zebrafish

def zebrafish_stim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'zebrafish_stim'
    config.max_batch = 1000
    config.seq_length = 11
    config.pred_length = 10
    config.compression_factor = 1

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['zebrafishstim_avg']
    config_ranges['model_label'] = ['Latent_PLRNN', 'POCO', 'MLP', 'AR_Transformer']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def zebrafish_region_avg():
    config = NeuralPredictionConfig()
    config.experiment_name = 'zebrafish_region_avg'
    config.dataset_label = 'zebrafish_avg'
    
    config_ranges = OrderedDict()
    config_ranges['context_length'] = [1, 48]
    config_ranges['pred_length'] = [1, 16]
    config_ranges['model_label'] = ['ARMLP', 'Latent_PLRNN', 'POCO', 'MLP', 'AR_Transformer']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    for seed in configs.keys():
        for config in configs[seed]:
            config: NeuralPredictionConfig
            config.seq_length = config.context_length + config.pred_length
            config.compression_factor = max(1, (config.seq_length - config.pred_length) // 3)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_dataset_filter():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_dataset_filter'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['lowpass', 'bandpass', 'none']
    config_ranges['dataset_label'] = dataset_list
    config_ranges['model_label'] = multi_session_model_list

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_large_dataset_filter():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_dataset_filter'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['lowpass', 'bandpass', 'none']
    config_ranges['dataset_label'] = ['zebrafishahrens']
    config_ranges['model_label'] = single_neuron_model_list

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multi_session():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_dataset_filter'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', ]
    config_ranges['dataset_label'] = dataset_list
    config_ranges['model_label'] = multi_session_model_list
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def ablations():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_dataset_filter'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', ]
    config_ranges['dataset_label'] = dataset_list
    config_ranges['model_label'] = ['POCO', 'POYO', 'MLP', 'TACO', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    for seed in configs.keys():
        for config in configs[seed]:
            config: NeuralPredictionConfig
            if config.model_label == 'TACO' and config.dataset_label == 'mice':
                config.batch_size = 16
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def poyo_compare_embedding_mode():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_embedding_mode'
    config.model_label = 'POCO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['zebrafish_pc', 'mice']
    config_ranges['unit_embedding_components'] = [[], ['session'], ['session', 'unit_type'], ]
    config_ranges['latent_session_embedding'] = [False, True, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def poyo_compare_compression_factor():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_compression_factor'
    config.model_label = 'POCO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = dataset_list
    config_ranges['compression_factor'] = [1, 4, 16, 48, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def poyo_compare_num_latents():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_num_latents'
    config.model_label = 'POCO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = dataset_list
    config_ranges['poyo_num_latents'] = [1, 2, 4, 8, 16, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_wd():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_wd'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = dataset_list
    config_ranges['model_label'] = ['POCO', 'NLinear', 'TexFilter']
    config_ranges['wdecay'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 10]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_lr():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_lr'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = dataset_list
    config_ranges['model_label'] = ['POCO', 'NLinear', 'TexFilter']
    config_ranges['lr'] = [1e-4, 3e-4, 1e-3, 5e-3, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_hidden_size():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_hidden_size'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = dataset_list
    config_ranges['model_label'] = ['POCO', ]
    config_ranges['decoder_hidden_size'] = [8, 32, 128, 512, 1024, 1536, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_context_window_length():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_context_window_length'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['zebrafish_pc', 'mice', ]
    config_ranges['model_label'] = ['POCO', 'NLinear', 'TexFilter']
    config_ranges['seq_length'] = [config.pred_length + x for x in [1, 3, 12, 24, 48]]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)
    configs = configure_models(configs)
    configs = configure_dataset(configs)

    for seed in configs.keys():
        for config in configs[seed]:
            config: NeuralPredictionConfig
            config.compression_factor = max(1, (config.seq_length - config.pred_length) // 3)
    return configs

def poyo_compare_num_layers():
    config = NeuralPredictionConfig()
    config.experiment_name = 'poyo_compare_num_layers'
    config.model_label = 'POCO'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = dataset_list
    config_ranges['decoder_num_layers'] = [1, 4, 8, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multi_species():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multi_species'
    config.dataset_label = [
        'zebrafish_pc', 'zebrafishahrens_pc', 'mice_pc', 
        'mice', 'celegans', 'celegansflavell', 
    ]
    config.max_batch = 20000
    config_ranges = OrderedDict()
    config_ranges['model_label'] = ['POCO', ]
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
        # ['zebrafish_pc', 'zebrafishahrens_pc', 'zebrafishjain_pc'], 
        ['zebrafish_pc', 'zebrafishahrens_pc', ],
    ]
    config_ranges['model_label'] = ['POCO', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
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
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

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
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

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
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

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
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

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
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

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

training_set_size_models = ['POCO', 'NLinear', 'TexFilter']

def compare_models_multiple_splits_celegans_flavell():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multiple_splits'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', 'lowpass', ]
    config_ranges['dataset_label'] = create_splits(40, [1, 2, 3, 5, 8, 12, 20, 40], 'celegansflavell')
    config_ranges['model_label'] = ['POCO', 'TexFilter']
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multiple_splits_mice():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multiple_splits'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', 'lowpass', ]
    config_ranges['dataset_label'] = create_splits(12, [1, 2, 3, 4, 6, 12], 'mice')
    config_ranges['model_label'] = ['POCO', 'TexFilter']
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multiple_splits_zebrafish():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multiple_splits'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = \
        create_splits(19, [1, 2, 3, 5, 10, 19], 'zebrafish') + \
        create_splits(19, [1, 2, 3, 5, 10, 19], 'zebrafish_pc')
    config_ranges['model_label'] = ['POCO', 'TexFilter']
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multiple_splits_zebrafish_ahrens():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multiple_splits'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', ]
    config_ranges['dataset_label'] = \
        create_splits(15, [1, 2, 3, 5, 8, 15], 'zebrafishahrens') + \
        create_splits(15, [1, 2, 3, 5, 8, 15], 'zebrafishahrens_pc')
    config_ranges['model_label'] = ['POCO', 'TexFilter']
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_multiple_splits_zebrafish_ahrens_lowpass():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_multiple_splits'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['lowpass', ]
    config_ranges['dataset_label'] = \
        create_splits(15, [1, 2, 3, 5, 8, 15], 'zebrafishahrens') + \
        create_splits(15, [1, 2, 3, 5, 8, 15], 'zebrafishahrens_pc')
    config_ranges['model_label'] = ['POCO', 'TexFilter']
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_train_length_celegans_flavell():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_train_length'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', 'lowpass', ]
    config_ranges['dataset_label'] = ['celegansflavell', ]
    config_ranges['model_label'] = training_set_size_models
    config_ranges['train_data_length'] = [256, 512, 768, 1024, 1536, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_train_length_mice():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_train_length'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', 'lowpass', ]
    config_ranges['dataset_label'] = ['mice', ]
    config_ranges['model_label'] = training_set_size_models
    config_ranges['train_data_length'] = [256, 512, 1024, 2048, 4096, 8192, 16384, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_train_length_zebrafish():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_train_length'

    config_ranges = OrderedDict()
    config_ranges['dataset_label'] = ['zebrafish', 'zebrafish_pc', ]
    config_ranges['model_label'] = training_set_size_models
    config_ranges['train_data_length'] = [256, 512, 1024, 1536, 2048, 3072]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=2)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_train_length_zebrafish_ahrens():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_train_length'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', 'lowpass', ]
    config_ranges['dataset_label'] = ['zebrafishahrens_pc', ]
    config_ranges['model_label'] = training_set_size_models
    config_ranges['train_data_length'] = [256, 512, 1024, 1536, 2048, 3072]

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
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

selected_models = ['POCO', ]

def compare_models_sim_multi_session():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_sim_multi_session'

    config_ranges = OrderedDict()
    config_ranges['connectivity_noise'] = [0, 0.05, 0.5, 1]
    config_ranges['dataset_label'] = [f'sim_{n}' for n in [300]]
    config_ranges['model_label'] = selected_models
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=16)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_sim():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_sim'
    config.max_batch = 5000

    config_ranges = OrderedDict()
    config_ranges['connectivity_noise'] = [0, 0.05, 0.5, 1]
    config_ranges['dataset_label'] = [f'sim_{n}-{seed}' for n in [300] for seed in range(16)]
    config_ranges['model_label'] = selected_models

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=16)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_models_sim_all():
    config = NeuralPredictionConfig()
    config.experiment_name = 'compare_models_sim'
    config.max_batch = 5000

    config_ranges = OrderedDict()
    config_ranges['connectivity_noise'] = [0,]
    config_ranges['dataset_label'] = [f'sim_{n}-{seed}' for n in [150, 300] for seed in range(1)]
    config_ranges['model_label'] = single_session_model_list

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=16)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def basemodels(): # train the base model on some sessions/datasets, used for finetuning
    config = NeuralPredictionConfig()
    config.experiment_name = 'basemodels'

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', ]
    config_ranges['dataset_label'] = [
        ['zebrafish_pc-*', ], ['zebrafish_pc'],
        ['zebrafishahrens_pc-0-11', ], ['zebrafishahrens_pc'],
        ['zebrafish_pc-*', 'zebrafishahrens_pc', ],
        ['zebrafish_pc', 'zebrafishahrens_pc-0-11', ],
        ['zebrafish_pc', 'zebrafishahrens_pc'],
        ['mice-0-10'],
    ]
    config_ranges['model_label'] = ['POCO', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)
    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def finetuning(): # compare finetuning (all), only train embedding, train from scratch, linear/mlp train from scratch
    config = NeuralPredictionConfig()
    config.experiment_name = 'finetuning'
    config.log_every = 20
    config.max_batch = 2000

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', ]
    config_ranges['dataset_label'] = (
        [f'zebrafish_pc-{id}' for id in [4, 9, 14, 18]] + 
        [f'zebrafishahrens_pc-{id}' for id in [11, 12, 13, 14]] +
        [f'mice-{id}' for id in [10, 11]]
    )
    config_ranges['model_label'] = ['Pre-POCO(Full)', 'Pre-POCO(UI)', 'Pre-POCO(UI+MLP)', 'POCO', 'NLinear', 'MLP', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)

    pretrained_model_id_dict = {
        'zebrafish_pc': 0,
        'zebrafishahrens_pc': 2,
        'mice': 7,
    } # corresponding to the dataset_id in the basemodels function

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            if config.model_label.startswith('Pre-POCO'):
                config.finetuning = True
                model_id = pretrained_model_id_dict[config.dataset_label.split('-')[0]]
                config.load_path = os.path.join(basemodels()[seed][model_id].save_path, 'net_best.pth')
                if config.model_label == 'Pre-POCO(UI)':
                    config.freeze_backbone = True
                    config.freeze_conditioned_net = True
                elif config.model_label == 'Pre-POCO(UI+MLP)':
                    config.freeze_backbone = True
                config.model_label = 'POCO'

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def finetuning_test_time():
    config = NeuralPredictionConfig()
    config.experiment_name = 'finetuning_test'
    config.log_every = 50
    config.max_batch = 200
    config.batch_size = 1

    config.load_path = os.path.join(basemodels()[0][0].save_path, 'net_best.pth')
    config.freeze_backbone = True
    config.freeze_conditioned_net = True
    config.finetuning = True

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', ]
    config_ranges['dataset_label'] = [f'zebrafish_pc-{id}' for id in [4, ]]
    config_ranges['model_label'] = ['POCO', ]
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)

    configs = configure_models(configs)
    configs = configure_dataset(configs)
    return configs

def compare_pretraining_dataset():
    config = NeuralPredictionConfig()
    config.experiment_name = 'finetuning'
    config.log_every = 20
    config.max_batch = 2000

    config_ranges = OrderedDict()
    config_ranges['dataset_filter'] = ['none', ]
    config_ranges['dataset_label'] = (
        ['zebrafishjain_pc'] +
        [f'zebrafish_pc-{id}' for id in [4, 9, 14, 18]] +
        [f'zebrafishahrens_pc-{id}' for id in [11, 12, 13, 14]]
    )
    config_ranges['model_label'] = ['Pre-POCO(Ahrens)', 'Pre-POCO(Deisseroth)', 'Pre-POCO(Both Datasets)', 'POCO', 'NLinear', 'MLP']
    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=3)

    pretrained_model_id_dict = {
        '(Deisseroth)': {'zebrafish_pc': 0, 'zebrafishahrens_pc': 1, 'zebrafishjain_pc': 1},
        '(Ahrens)': {'zebrafish_pc': 3, 'zebrafishahrens_pc': 2, 'zebrafishjain_pc': 3},
        '(Both Datasets)': {'zebrafish_pc': 4, 'zebrafishahrens_pc': 5, 'zebrafishjain_pc': 6},
    }

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            if config.model_label.startswith('Pre-POCO'):
                config.finetuning = True
                model_id = pretrained_model_id_dict[config.model_label[8: ]][config.dataset_label.split('-')[0]]
                config.load_path = os.path.join(basemodels()[seed][model_id].save_path, 'net_best.pth')
                config.freeze_backbone = True
                config.freeze_conditioned_net = True
                config.model_label = 'POCO'

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