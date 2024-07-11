import torch
import torch.nn as nn
import models
import numpy as np
import torch
import math

from configs.config_global import ROOT_DIR, DEVICE, MAP_LOC
from models.rnns import PLRNN_Step, CTRNNCell, CTRNN

def get_rnn(rnn_type, rnn_in_size, hidden_size, **kwargs):
    # TODO: support rnn_in_size for transformer and ssm

    if rnn_type == 'RNN':
        rnn = nn.RNN(rnn_in_size, hidden_size, **kwargs)
    elif rnn_type == 'LSTM':
        rnn = nn.LSTM(rnn_in_size, hidden_size, **kwargs)
    elif rnn_type == 'GRU':
        rnn = nn.GRU(rnn_in_size, hidden_size, **kwargs)
    elif rnn_type == 'CTRNNCell':
        rnn = CTRNNCell(input_size=rnn_in_size, hidden_size=hidden_size, **kwargs)
        assert kwargs.get('num_layers', 1) == 1
    elif rnn_type == 'PLRNN':
        assert rnn_in_size == None
        rnn = PLRNN_Step(hidden_size, **kwargs)
    elif rnn_type == 'CTRNN':
        rnn = CTRNN(rnn_in_size, hidden_size, **kwargs)
    elif rnn_type == 'BiGRU':
        rnn = nn.GRU(rnn_in_size, hidden_size, bidirectional=True, **kwargs)
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
        rnn = models.s4.S4(hidden_size, **kwargs)
    else:
        raise NotImplementedError('RNN not implemented')
    return rnn