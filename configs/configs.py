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
        self.seed = 0

        # Weight for each dataset
        self.mod_w = [1.0, ]

        # Specify required resources (useful when running on cluster)
        self.hours = 24
        self.mem = 16
        self.cpu = 2 
        self.num_workers = 1

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
        self.lr = 1e-3
        self.wdecay = 1e-4

        # scheduler
        self.use_lr_scheduler = False
        self.scheduler_type = None

        # training
        self.num_ep = 100000
        self.max_batch = 10000

        # evaluation
        self.perform_val = True
        self.perform_test = True
        self.log_every = 100
        self.save_every = 100000
        self.test_batch = 100

        # autoregressive model config
        self.print_mode = 'error'
        self.model_type = 'SimpleRNN'
        self.rnn_type = 'LSTM'
        self.num_layers = 1
        self.teacher_forcing = True
        self.shared_backbone = True

        # rnn config
        self.hidden_size = 512
        self.alpha = 0.1
        self.learnable_alpha = False

        # meta-rnn model config
        self.inner_wd = 0
        self.inner_lr = 10
        self.inner_train_step = 1
        self.inner_test_time_train_step = 1
        self.algorithm = 'maml'
        self.use_low_dim_rnn = True
        self.shared_rnn = False

        # seq vae model config
        self.encoder_rnn_type = 'BiGRU'
        self.encoder_hidden_size = 512 
        self.decoder_rnn_type = 'CTRNN'
        self.decoder_hidden_size = 512
        self.kl_loss_coef = 0.002

        # only used for latent models
        self.tf_interval = 5 

        self.do_analysis = False


class NeuralPredictionConfig(SupervisedLearningBaseConfig):

    def __init__(self):
        super().__init__()

        self.task_type = 'neural_prediction'
        self.dataset = 'zebrafish' # or simulation
        self.exp_types = EXP_TYPES

        self.seq_length = 50 # the total length of a sequence
        self.pred_length = 10 # the last pred_length frames will be predicted
        self.train_split = 0.7
        self.val_split = 0.3
        self.target_mode = 'raw' # or 'derivative'
        self.wdecay = 1e-4
        self.perform_test = False
        self.perform_val = True
        self.patch_length = 500 # train and val sets will be devided in each patch_length segment
        self.loss_mode = 'autoregressive' 
        # or 'prediction', for the latter the target will be the next frame

        # config for data
        self.animal_ids = 'all'
        # self.train_fish_ids = [] # all neural data (instead of just first 70%) of fish in this list will be used for training
        self.pc_dim = 512 # number of principal components to used for training, if None, predict original data
        self.normalize_mode = 'none' # 'minmax' (target will be [-1, 1]) or 'zscore' (target will zero-mean and unit variance) or 'none'
        self.sampling_rate = 10 # downsample the data to this rate, should be 1 for real neural data and 10 for simulated data

        # only available for zebrafish data and when pc_dim is None, could be any brain region name, 'all',
        # or 'average' (in which case we will average the neural activity within each brain region)
        self.brain_regions = 'all' 
        
        # config for simulated data
        self.n_neurons = 512 # only used for simulated dataset: the number of neurons used for training
        self.n_regions = 1
        self.ga = 2.0
        self.sim_noise_std = 0
        self.portion_observable_neurons = 1 # only used for simulated dataset: the portion of neurons that are observable
        self.train_data_length = 1000000
        self.sparsity = 1

        # config for visual fish data
        self.use_stimuli = False
        self.stimuli_dim = 0
        self.use_eye_movements = False
        self.use_motor = False

        self.max_batch = 5000
        self.test_batch = 10000 # test on all available data 
        self.mem = 32
        self.do_analysis = False
