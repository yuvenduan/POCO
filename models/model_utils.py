import torch
import torch.nn as nn
import models
import numpy as np
import torch
import math

from configs.config_global import ROOT_DIR, DEVICE
from configs.configs import SupervisedLearningBaseConfig
from utils.config_utils import load_config
from configs.configs import BaseConfig
from models.rnns import PLRNN_Step, CTRNNCell, CTRNN

def get_rnn_from_config(config: SupervisedLearningBaseConfig, **kwargs):
    rnn = get_rnn(
        rnn_type=config.rnn_type,
        rnn_in_size=kwargs.get('rnn_in_size', config.hidden_size),
        hidden_size=kwargs.get('hidden_size', config.hidden_size),
        alpha=config.rnn_alpha,
        rank=config.rnn_rank,
        num_layers=config.num_layers,
        residual=kwargs.get('residual', config.rnn_residual_connection),
    )
    return rnn

def get_rnn(rnn_type, rnn_in_size, hidden_size, alpha=0.1, rank=2, num_layers=1, residual=False, **kwargs):
    # TODO: support rnn_in_size for transformer and ssm
    """
    Get RNN based on rnn_type
    """

    if rnn_type == 'RNN':
        rnn = nn.RNN(rnn_in_size, hidden_size, num_layers=num_layers, **kwargs)
    elif rnn_type == 'LSTM':
        rnn = nn.LSTM(rnn_in_size, hidden_size, num_layers=num_layers, **kwargs)
    elif rnn_type == 'GRU':
        rnn = nn.GRU(rnn_in_size, hidden_size, num_layers=num_layers, **kwargs)
    elif rnn_type == 'CTRNNCell':
        rnn = CTRNNCell(input_size=rnn_in_size, hidden_size=hidden_size, alpha=alpha, **kwargs)
        assert kwargs.get('num_layers', 1) == 1
    elif rnn_type == 'PLRNN':
        assert rnn_in_size == None
        rnn = PLRNN_Step(hidden_size, alpha=alpha, **kwargs)
    elif rnn_type == 'CTRNN':
        rnn = CTRNN(rnn_in_size, hidden_size, alpha=alpha, num_layers=num_layers, **kwargs)
    elif rnn_type == 'LRRNN':
        rnn = CTRNNCell(rnn_in_size, hidden_size, alpha=alpha, rank=rank, residual=residual, **kwargs)
    elif rnn_type == 'BiGRU':
        rnn = nn.GRU(rnn_in_size, hidden_size, bidirectional=True, num_layers=num_layers, **kwargs)
    elif rnn_type == 'Transformer':
        assert rnn_in_size == hidden_size, 'Input size must be equal to hidden size for transformer'
        kwargs['num_layers'] = kwargs.get('num_layers', 3)
        rnn = models.transformer.CausalTransformerEncoder(
            hidden_size,
            models.transformer.CausalTransformerEncoderLayer(
                hidden_size, 
                nhead=8, 
                dim_feedforward=hidden_size * 2,
                dropout=0.1
            ), 
            **kwargs
        )
    elif rnn_type == 'Linear':
        rnn = nn.Sequential(
            nn.Linear(rnn_in_size, hidden_size),
            nn.Tanh()
        )
        assert kwargs.get('num_layers', 1) == 1
    elif rnn_type == 'S4':
        rnn = models.s4.S4(hidden_size, num_layers=num_layers, **kwargs)
    else:
        raise NotImplementedError('RNN not implemented')
    return rnn

def get_pretrained_model(model_path, config_path, datum_size=None, eval_mode=True):
    config = load_config(config_path)
    model = model_init(config, datum_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    if eval_mode:
        model.eval()
        model.requires_grad_(False)
    return model

def model_init(config_: BaseConfig, datum_size):
    if config_.model_type == 'Autoregressive':
        model = models.AutoregressiveModel(config_, datum_size=datum_size)
    elif config_.model_type == 'SeqVAE':
        model = models.MultiFishSeqVAE(config_, datum_size=datum_size)
    elif config_.model_type == 'Linear':
        model = models.Linear(config_, datum_size=datum_size)
    elif config_.model_type == 'LatentModel':
        model = models.LatentModel(config_, datum_size=datum_size)
    elif config_.model_type == 'TCN':
        model = models.MultiTCN(config_, datum_size=datum_size)
    elif config_.model_type == 'VQVAE':
        model = models.vqvae(config_, datum_size=datum_size)
    elif config_.model_type == 'Decoder':
        model = models.Decoder(config_, datum_size=datum_size)
    elif config_.model_type == 'Mixed':
        model = models.MixedModel(config_, datum_size=datum_size)
    else:
        raise NotImplementedError("Model not Implemented")

    if config_.load_path is not None:
        model.load_state_dict(torch.load(config_.load_path))
        
    model.to(DEVICE)
    return model