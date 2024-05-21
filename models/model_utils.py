import torch
import torch.nn as nn
import models
import numpy as np
import torch
import math

from configs.config_global import ROOT_DIR, DEVICE, MAP_LOC

class CTRNNCell(nn.Module):
    
    def __init__(
            self, 
            input_size=None, 
            hidden_size=128, 
            nonlinearity="tanh", 
            alpha=0.1,
            mul_rnn_noise=0,
            add_rnn_noise=0,
            learnable_alpha=False
        ):

        super(CTRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.alpha = alpha
        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        self.mul_rnn_noise = mul_rnn_noise
        self.add_rnn_noise = add_rnn_noise

        self.weight_ih = nn.Parameter(torch.zeros((input_size, hidden_size))) if input_size != None else None
        self.weight_hh = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

        if self.nonlinearity == "tanh":
            self.act = torch.tanh
        elif self.nonlinearity == "relu":
            self.act = torch.relu
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inp=None, hidden_in=None, fast_weights=None):

        # add fast weights, if given
        weight_ih = self.weight_ih
        weight_hh = self.weight_hh
        bias = self.bias
        alpha = self.alpha
        if fast_weights is not None:
            if self.learnable_alpha:
                alpha = alpha + fast_weights[0]
            if self.input_size is not None:
                weight_ih = weight_ih + fast_weights[-3]
            weight_hh = weight_hh + fast_weights[-2]
            bias = bias + fast_weights[-1]

        if self.learnable_alpha:
            alpha = torch.clip(alpha, 0.05, 0.99)

        # compute pre-activation
        pre_act = torch.matmul(hidden_in, weight_hh) + bias
        if self.input_size is not None:
            pre_act = torch.matmul(inp, weight_ih)

        # add noise
        if self.add_rnn_noise > 0:
            pre_act = pre_act + torch.randn_like(pre_act) * np.sqrt(2 / alpha) * self.add_rnn_noise
        hidden_out = (1 - alpha) * hidden_in + alpha * self.act(pre_act)
        if self.mul_rnn_noise > 0:
            hidden_out = hidden_out + torch.randn_like(hidden_out) * self.mul_rnn_noise
        
        return hidden_out

    def init_hidden(self, batch_s):
        return torch.zeros(batch_s, self.hidden_size).to(DEVICE)

class CTRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, **kwargs):
        super(CTRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.ModuleList([CTRNNCell(input_size if n == 0 else hidden_size, hidden_size, **kwargs) for n in range(num_layers)])

    def forward(self, inp=None, hidden_in=None, fast_weights=None, time_steps=0):
        # inp: (seq_len, batch_size, input_size)
        # hidden_in: (num_layers, batch_size, hidden_size)
        # fast_weights: list of fast weights for each layer
        if fast_weights is not None:
            assert self.num_layers == 1, 'Fast weights only supported for single layer RNN'
        if self.num_layers == 1:
            hidden_in = [hidden_in]
        if hidden_in is None:
            hidden_in = [self.rnn[i].init_hidden(inp.shape[1]) for i in range(self.num_layers)]
        if inp is None:
            assert time_steps > 0, 'Time steps must be given if no input is provided'
            inp = [None] * time_steps
        else:
            time_steps = inp.shape[1]
        
        for i in range(self.num_layers):
            layer_out = []
            for j in range(time_steps):
                hidden_in[i] = self.rnn[i](inp[j], hidden_in[i], fast_weights)
                layer_out.append(hidden_in[i])
            inp = torch.stack(layer_out)
        return inp, hidden_in

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