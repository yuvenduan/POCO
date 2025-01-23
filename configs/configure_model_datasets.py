import os
import os.path as osp
import logging
from configs.configs import NeuralPredictionConfig

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
            elif label == 'spontaneous_fc':
                config.dataset = 'zebrafish'
                config.pc_dim = None
                config.fc_dim = 500
                config.normalize_mode = 'zscore'
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
            elif label == 'stim_avg':
                config.dataset = 'zebrafish_stim'
                config.pc_dim = None
                config.normalize_mode = 'zscore'
                config.brain_regions = 'average'
                config.exp_types = ['control']
            elif label == 'stim_pc':
                config.dataset = 'zebrafish_stim'
                config.pc_dim = 512
                config.exp_types = ['control']
            elif label == 'stim':
                config.dataset = 'zebrafish_stim'
                config.pc_dim = None
                config.exp_types = ['control']
                config.batch_size = 4
                config.mem = 128
                config.test_set_window_stride = 32
            elif label == 'celegans_pc':
                config.dataset = 'celegans'
                config.pc_dim = 100
            elif label == 'celegans':
                config.dataset = 'celegans'
                config.pc_dim = None
            elif label == 'mice_pc':
                config.dataset = 'mice'
                config.pc_dim = 512
            elif label == 'mice':
                config.dataset = 'mice'
                config.pc_dim = None
            elif label[:6] == 'sim_pc':
                config.dataset = 'simulation'
                config.pc_dim = 512
                config.sampling_freq = 5
                config.n_neurons = int(label[6:])
                config.mem = 64
            elif label[:3] == 'sim':
                config.dataset = 'simulation'
                config.pc_dim = None
                config.sampling_freq = 5
                config.n_neurons = int(label[3:])
                config.mem = 128
                if config.n_neurons > 512:
                    config.batch_size = 16
            else:
                raise ValueError(f'Unknown dataset label: {label}')

    return configs

def configure_models(configs: dict, customize=False):
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
                config.encoder_type = 'none'

                if not customize:
                    config.separate_projs = True
                    if config.decoder_type == 'POYO' and config.poyo_output_mode != 'latent':
                        config.separate_projs = False

                if label[-5: ] == 'TOTEM':
                    data_label = config.dataset_label
                    config.encoder_type = 'vqvae'
                    config.encoder_dir = \
                        f'experiments/vqvae/model_dataset_label{data_label}_compression_factor{config.compression_factor}_s{seed}'
                    config.encoder_state_dict_file = 'net_20000.pth'
                elif label[-3: ] == 'cnn':
                    config.encoder_type = 'cnn'
                    config.kernel_size = 16
                    config.conv_channels = 128
                elif label[-3: ] == 'pop':
                    config.population_token = True
                    config.population_token_dim = 512

            else:
                raise ValueError(f'Unknown model label: {label}')

    return configs