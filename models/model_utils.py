import torch
import torch.nn as nn
import numpy as np
import torch
import models.multi_session_models as multi_session_models

from configs.config_global import DEVICE
from configs.configs import SupervisedLearningBaseConfig
from utils.config_utils import load_config
from configs.configs import BaseConfig
from models.layers.rnns import PLRNN_Step, CTRNNCell, CTRNN
from models.layers.transformer import CausalTransformerEncoder, CausalTransformerEncoderLayer

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
    elif rnn_type == 'Transformer':
        assert rnn_in_size == hidden_size, 'Input size must be equal to hidden size for transformer'
        kwargs['num_layers'] = kwargs.get('num_layers', 4)
        rnn = CausalTransformerEncoder(
            hidden_size,
            CausalTransformerEncoderLayer(
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
    else:
        raise NotImplementedError('RNN not implemented')
    return rnn

def get_pretrained_model(model_path, config_path, datum_size=None, eval_mode=True):
    config = load_config(config_path)
    model: nn.Module = model_init(config, datum_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    if eval_mode:
        model.eval()
        model.requires_grad_(False)
    return model

def model_init(config: BaseConfig, datum_size, unit_type=None):
    model_class = eval("multi_session_models." + config.model_type)

    if config.model_type == 'Decoder': # only Decoder might use unit_type infomation
        model = model_class(config, datum_size, unit_type)
    else:
        model = model_class(config, datum_size)
    model: torch.nn.Module

    if config.finetuning:
        assert config.load_path is not None, 'load_path must be specified for finetuning'
        assert hasattr(model, 'load_pretrained'), 'model must have load_pretrained method for finetuning'
        model.load_pretrained(torch.load(config.load_path, weights_only=True))
    elif config.load_path is not None:
        model.load_state_dict(torch.load(config.load_path, weights_only=True))
        
    model.to(DEVICE)
    return model

def count_parameters(model: nn.Module, trainable_only: bool = True):
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
        trainable_only (bool): If True, only counts parameters with requires_grad=True.

    Returns:
        int: Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())