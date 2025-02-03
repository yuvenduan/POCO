"""
Configurations for the project
format adapted from https://github.com/gyyang/olfaction_evolution

Be aware that the each field in the configuration must be in basic data type that
jason save and load can preserve. each field cannot be complicated data type
"""

from configs.config_global import EXP_TYPES

class BaseConfig(object):

    def __init__(self):
        """
        model_type: model type, eg. "ConvRNNBL"
        task_type: task type, eg. "n_back"
        """
        self.experiment_name = None
        self.model_name = None
        self.model_type = None
        self.task_type = None
        self.dataset = None
        self.save_path = None
        self.model_label = None
        self.dataset_label = None
        self.seed = 0

        # Weight for each dataset
        self.mod_w = [1.0, ]

        # Specify required resources (useful when running on cluster)
        self.hours = 24
        self.mem = 32
        self.cpu = 2
        self.num_workers = 0

        # basic evaluation parameters
        self.store_states = False

        # if not None, load model from the designated path
        self.load_path = None

        # if overwrite=False and training log is complete, skip training
        self.overwrite = True

        self.config_mode = 'train'
        self.training_mode = 'supervised'

        self.input_shape = None
        self.model_outsize = None

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)

class SupervisedLearningBaseConfig(BaseConfig):

    def __init__(self):
        super().__init__()

        self.training_mode = 'supervised'
        # max norm of grad clipping, eg. 1.0 or None
        self.grad_clip = 5

        # optimizer
        self.batch_size = 64
        self.optimizer_type = 'AdamW'
        self.lr = 3e-4
        self.wdecay = 1e-4

        # scheduler
        self.use_lr_scheduler = False
        self.scheduler_type = 'CosineAnnealing'

        # training
        self.num_ep = 100000
        self.max_batch = 5000

        # evaluation
        self.perform_val = True
        self.perform_test = True
        self.log_every = 100
        self.save_every = 50000

        # autoregressive model config
        self.print_mode = 'error'
        self.model_type = 'Autoregressive'
        self.rnn_type = 'LSTM'
        self.num_layers = 1
        self.teacher_forcing = True
        self.shared_backbone = True
        self.rnn_layernorm = False

        # rnn config
        self.hidden_size = 512
        self.rnn_alpha = 0.05 # only for LRRNN, CTRNN, PLRNN
        self.rnn_rank = 4 # only for Low-Rank RNN (LRRNN)
        self.learnable_alpha = False
        self.rnn_residual_connection = True # only for Low-Rank RNN (LRRNN)

        # TCN model config
        self.stem_ratio = 6
        self.downsample_ratio = 1
        self.ffn_ratio = 1
        self.patch_size = 4
        self.patch_stride = 4
        self.num_blocks = [1]
        self.large_size = [17]
        self.small_size = [5]
        self.dims = [64, 64, 64, 64]
        self.dw_dims = [64, 64, 64, 64]
        self.small_kernel_merged = False
        self.call_structural_reparam = False
        self.use_multi_scale = False

        self.dropout = 0.3
        self.kernel_size = 25
        self.individual = 0
        self.freq = 'h'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.head_dropout = 0
        self.decomposition = 0

        # vqvae model config
        self.block_hidden_size = 128
        self.vqvae_embedding_dim = 64
        self.num_embeddings = 256
        self.commitment_cost = 0.25
        self.compression_factor = 16
        self.num_residual_layers = 2
        self.res_hidden_size = 64

        # cnn model config
        self.kernel_size
        self.conv_channels = 64
        self.conv_stride = 4

        # latent model config
        self.tf_interval = 5 
        self.hidden_size
        self.rnn_type # only support RNNs without input: CTRNN, PLRNN, ...
        self.latent_to_ob = 'linear' # or 'identity'

        # decoder config
        self.tokenizer_dir = None
        self.tokenizer_type = 'none' # or 'cnn', 'vqvae'
        self.population_token = False # if True, use a single token for the whole population, otherwise use individual tokens
        self.population_token_dim = 512
        self.tokenizer_state_dict_file = f'net_{self.max_batch}.pth'
        self.decoder_type = 'Transformer' # or 'MLP', 'Linear'
        self.decoder_hidden_size = 256
        self.separate_projs = True
        self.decoder_proj_init = 'fan_in'
        self.normalize_input = True # if True, normalize the input to the decoder
        self.mu_std_loss_coef = 0
        self.mu_std_module_mode = 'combined' # original mean / std; learned mean / std
        self.mu_std_separate_projs = False

        # poyo config
        self.poyo_num_latents = 8
        self.poyo_query_mode = 'single' # or 'multi'
        self.poyo_output_mode = 'query' # or 'latent'
        self.decoder_num_layers = 4
        self.decoder_num_heads = 8
        self.poyo_unit_dropout = 0
        self.rotary_attention_tmax = 100

        self.do_analysis = False

class DatasetConfig:

    def __init__(self):
        self.dataset = None # set in configure_model_datasets.py
        self.batch_size = None 
        self.num_workers = None
        self.pred_length = None
        self.seq_length = None
        self.show_chance = True

        self.train_split = 0.6
        self.val_split = 0.2
        self.test_split = 0.2
        self.patch_length = 1000 # train and val sets will be divided in each patch_length segment
        self.session_ids = None # if not None, only use data from these sessions
        self.animal_ids = None # if not None, only use data from these animals
        self.pc_dim = 512 # number of principal components to used for training, if None, predict original data
        self.fc_dim = None # number of functional clusters to used for training; pc_dim will be ignored if this is not None
        self.normalize_mode = 'none' # 'minmax' (target will be [-1, 1]) or 'zscore' (target will zero-mean and unit variance) or 'none'
        self.sampling_freq = 1 # downsample the data by this factor
        self.sampling_mode = 'avg' 
        self.test_set_window_stride = 1 # larger stride will make the test set smaller but faster to evaluate

        # config for simulated data
        self.n_neurons = 512 # only used for simulated dataset: the number of neurons used for training
        self.n_regions = 1
        self.ga = 1.8
        self.sim_noise_std = 0
        self.sim_seed = 0
        self.portion_observable_neurons = 1 # only used for simulated dataset: the portion of neurons that are observable
        self.train_data_length = 1000000
        self.sparsity = 1

        # config for visual fish data
        self.use_stimuli = False
        self.stimuli_dim = 0
        self.use_eye_movements = False
        self.use_motor = False

        # only available for zebrafish data and when pc_dim is None, could be any brain region name, 'all',
        # or 'average' (in which case we will average the neural activity within each brain region)
        self.brain_regions = 'all' 
        self.exp_types = EXP_TYPES

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dict__}
    
    def __str__(self):
        return str(self.__dict__)

class NeuralPredictionConfig(SupervisedLearningBaseConfig):

    def __init__(self):
        super().__init__()

        self.task_type = 'neural_prediction'

        self.seq_length = 64 # the total length of a sequence
        self.pred_length = 16 # the last pred_length frames will be predicted
        self.target_mode = 'raw' # or 'derivative'
        self.wdecay = 1e-4
        self.perform_test = True
        self.perform_val = True
        self.loss_mode = 'autoregressive'
        # or 'prediction', for the latter the target will be the next frame

        # config for data
        self.dataset = 'zebrafish' # or a list of dataset names
        self.dataset_config = {}

        self.max_batch = 10000
        self.test_batch = 100000 # test on all available data 
        self.mem = 32
        self.do_analysis = False

    def to_dict(self):
        # Convert to a JSON-serializable dictionary
        config_dict = {key: getattr(self, key) for key in self.__dict__}
        # Convert dataset_config dictionary values to dicts
        if isinstance(self.dataset_config, dict):
            config_dict['dataset_config'] = {
                k: v.to_dict() if isinstance(v, DatasetConfig) else v
                for k, v in self.dataset_config.items()
            }
        return config_dict