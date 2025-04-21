import os
import os.path as osp
import logging
from configs.configs import NeuralPredictionConfig, DatasetConfig

def configure_dataset(configs: dict):
    """
    Add dataset configurations to the NeuralPredictionConfig objects according to the dataset label

    The dataset label has the following format: {dataset_name}_{dataset_type}(-{session_id}), where:
    - dataset_name: the name of the dataset (e.g. zebrafish, celegans, mice, sim)
    - dataset_type: the type of the dataset (e.g. pc, fc, avg)
    - session_id: the id of the session (e.g. 0, 1, 2), if not specified, all sessions are used
        alternatively, the session_id can be a range (e.g. 0-2 represents sessions 0, 1)
    """

    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            labels = config.dataset_label
            config.dataset = []

            if isinstance(labels, str):
                labels = [labels]
                config.dataset_label = labels

            for label in labels:
            
                session_id = label.split('-')[1] if len(label.split('-')) > 1 else None
                session_id_2 = label.split('-')[2] if len(label.split('-')) > 2 else None
                label_prefix = label.split('-')[0]
                dataset_name = label_prefix.split('_')[0]
                dataset_type = label_prefix.split('_')[1] if len(label.split('_')) > 1 else None
                dataset_config = DatasetConfig()

                dataset_config.batch_size = config.batch_size
                dataset_config.num_workers = config.num_workers
                dataset_config.pred_length = config.pred_length
                dataset_config.seq_length = config.seq_length
                dataset_config.train_data_length = config.train_data_length if hasattr(config, 'train_data_length') else int(1e9)

                if session_id is not None:
                    if session_id == '*':
                        assert dataset_name == 'zebrafish'
                        dataset_config.session_ids = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17] 
                        # all sessions except 4, 9, 14, 18; left out for testing fine-tuning
                    elif session_id_2 is not None:
                        dataset_config.session_ids = list(range(int(session_id), int(session_id_2)))
                    else:
                        dataset_config.session_ids = [int(session_id)]

                if dataset_name == 'zebrafish':
                    config.dataset.append('zebrafish')
                    if dataset_type == 'pc':
                        dataset_config.pc_dim = 512
                    elif dataset_type == 'fc':
                        dataset_config.fc_dim = 500
                    elif dataset_type == None:
                        dataset_config.pc_dim = None
                        dataset_config.test_set_window_stride = 8
                        dataset_config.batch_size = 8
                        config.mem = max(config.mem, 128)
                    else:
                        raise ValueError(f'Unknown dataset type: {dataset_type}')
                    
                elif dataset_name == 'zebrafishahrens' or dataset_name == 'zebrafishjain':
                    config.dataset.append(dataset_name)
                    dataset_config.filter_type = 'lowpass'
                    if dataset_type == 'pc':
                        dataset_config.pc_dim = 512
                    elif dataset_type == None:
                        dataset_config.pc_dim = None
                        dataset_config.test_set_window_stride = 32
                        dataset_config.batch_size = 4
                        config.mem = max(config.mem, 256)
                    else:
                        raise ValueError(f'Unknown dataset type: {dataset_type}')
                
                elif dataset_name == 'zebrafishstim':
                    config.dataset.append('zebrafishstim')
                    dataset_config.exp_types = ['control']
                    if dataset_type == 'pc':
                        dataset_config.pc_dim = 512
                    elif dataset_type == 'avg':
                        dataset_config.brain_regions = 'average'
                        dataset_config.pc_dim = None
                    elif dataset_type == None:
                        dataset_config.pc_dim = None
                        dataset_config.test_set_window_stride = 32
                        dataset_config.batch_size = 8
                        config.mem = max(config.mem, 128)
                    else:
                        raise ValueError(f'Unknown dataset type: {dataset_type}')
                
                elif dataset_name in ['celegans', 'celegansflavell']:
                    config.dataset.append(dataset_name)
                    dataset_config.filter_type = 'lowpass'
                    if dataset_type == 'pc':
                        dataset_config.pc_dim = 100
                    elif dataset_type == None:
                        dataset_config.pc_dim = None
                    else:
                        raise ValueError(f'Unknown dataset type: {dataset_type}')
                
                elif dataset_name == 'mice':
                    config.dataset.append('mice')
                    dataset_config.filter_type = 'lowpass'
                    if dataset_type == 'pc':
                        dataset_config.pc_dim = 512
                    elif dataset_type == None:
                        dataset_config.pc_dim = None
                    else:
                        raise ValueError(f'Unknown dataset type: {dataset_type}')
                
                elif dataset_name == 'sim':
                    config.dataset.append('simulation')
                    dataset_config.n_neurons = int(label_prefix.split('_')[-1])
                    dataset_type = label_prefix.split('_')[1] if len(label.split('_')) > 2 else None
                    dataset_config.sampling_freq = 5
                    config.mem = 128
                    if dataset_type == 'pc':
                        dataset_config.pc_dim = 512
                    elif dataset_type == None:
                        dataset_config.pc_dim = None
                        dataset_config.batch_size = 16
                    else:
                        raise ValueError(f'Unknown dataset type: {dataset_type}')
                    
                else:
                    raise ValueError(f'Unknown dataset label: {label}')
                
                if config.dataset_filter is not None: # overwrite the filter type, if specified
                    dataset_config.filter_type = config.dataset_filter

                dataset_config.dataset = config.dataset[-1]
                config.dataset_config[label] = dataset_config

            config.mod_w = [1] * len(config.dataset)

    return configs

def configure_models(configs: dict):
    """
    Add model configurations to the NeuralPredictionConfig objects according to the model label
    """
    for seed, config_list in configs.items():
        for config in config_list:
            config: NeuralPredictionConfig
            label = config.model_label

            model_type = label.split('_')[0]
            sub_model_type = label.split('_')[1] if len(label.split('_')) > 1 else None
            
            if model_type in ['AR', 'MultiAR']:
                config.loss_mode = 'autoregressive'
                config.model_type = 'Autoregressive' if model_type == 'AR' else 'MultiAutoregressive'
                config.rnn_type = sub_model_type
                config.num_layers = 4 if config.rnn_type in ['S4', 'Transformer'] else 1
            elif model_type == 'Latent':
                config.loss_mode = 'autoregressive'
                config.model_type = 'LatentModel'
                config.rnn_type = sub_model_type
                if config.rnn_type == 'RNN':
                    config.rnn_type = 'CTRNNCell'
                elif config.rnn_type[: 5] == 'LRRNN':
                    config.rnn_rank = int(config.rnn_type[5:])
                    config.rnn_type = 'LRRNN'
                config.tf_interval = 4
            elif model_type in ['Linear', 'MLP', 'Transformer', 'POYO', 'POCO', 'TACO']:
                config.model_type = 'Decoder'
                config.loss_mode = 'prediction'
                config.decoder_type = model_type
                config.tokenizer_dir = None
                config.tokenizer_type = 'none'
                config.separate_projs = False

                if config.compression_factor is None:
                    config.compression_factor = config.decoder_context_length if config.decoder_context_length is not None else config.seq_length - config.pred_length

                if model_type == 'POCO':
                    config.conditioning = 'mlp'
                    config.decoder_type = 'POYO'
                elif model_type == 'MLP':
                    config.decoder_hidden_size = 1024
                elif model_type == 'TACO':
                    config.conditioning = 'mlp'
                    config.decoder_type = 'Transformer'

                if sub_model_type == 'TOTEM':
                    data_label = config.dataset_label
                    config.tokenizer_type = 'vqvae'
                    config.tokenizer_dir = \
                        f'experiments/vqvae/model_dataset_label{data_label}_compression_factor{config.compression_factor}_s{seed}'
                    config.tokenizer_state_dict_file = 'net_20000.pth'
                elif sub_model_type == 'cnn':
                    config.tokenizer_type = 'cnn'
                    config.kernel_size = 16
                    config.conv_channels = 128
                elif sub_model_type == 'pop':
                    config.population_token = True
                    config.population_token_dim = 512
            elif model_type == 'NetFormer':
                config.loss_mode = 'autoregressive'
                config.model_type = 'NetFormer'
                config.normalize_input = True
                config.mu_module_mode = config.std_module_mode = 'original'
            else:
                config.loss_mode = 'prediction'
                config.model_type = model_type
                assert sub_model_type is None

    return configs