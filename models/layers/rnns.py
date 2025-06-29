from typing import Optional, Tuple
from configs.config_global import DEVICE
import torch.nn as nn
import numpy as np
import torch
import math

class CTRNNCell(nn.Module):
    
    def __init__(
            self, 
            input_size=None, 
            hidden_size=128, 
            nonlinearity="tanh", 
            alpha=0.1,
            mul_rnn_noise=0,
            add_rnn_noise=0,
            learnable_alpha=False,
            rank=None,
            residual=True
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
        self.rank = rank
        self.residual = residual
        if rank is None:    
            self.weight_hh = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        else:
            self.weight_hh_n = nn.Parameter(torch.zeros((hidden_size, rank)))
            self.weight_hh_m = nn.Parameter(torch.zeros((rank, hidden_size)))

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
        if self.rank is None:
            weight_hh = self.weight_hh
        else:
            weight_hh = (self.weight_hh_n @ self.weight_hh_m) * np.sqrt(self.hidden_size / self.rank)
            if self.residual:
                identity = torch.eye(self.hidden_size, device=DEVICE)
                normalized_hh_n = self.weight_hh_n / torch.linalg.norm(self.weight_hh_n, dim=0, ord=2)
                residual = identity - normalized_hh_n @ normalized_hh_n.T
                weight_hh = weight_hh + residual

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
            pre_act = pre_act + torch.matmul(inp, weight_ih)
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

    def forward(self, inp=None, hidden_in=None, fast_weights=None, time_steps=1):
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

# PLRNN adapted from https://github.com/DurstewitzLab/dendPLRNN/blob/main/BPTT_TF/bptt/PLRNN_model.py
class Latent_Step(nn.Module):
    def __init__(self, dz, clip_range=None, layer_norm=False):
        super(Latent_Step, self).__init__()
        self.clip_range = clip_range
        #self.nonlinearity = nn.ReLU()
        self.dz = dz

        if layer_norm:
            self.norm = lambda z: z - z.mean(dim=1, keepdim=True)
        else:
            self.norm = nn.Identity()

    def init_AW_random_max_ev(self):
        AW = torch.eye(self.dz) + 0.1 * torch.randn(self.dz, self.dz)
        max_ev = torch.max(torch.abs(torch.linalg.eigvals(AW)))
        return nn.Parameter(AW / max_ev, requires_grad=True)

    def init_uniform(self, shape: Tuple[int]) -> nn.Parameter:
        # empty tensor
        tensor = torch.empty(*shape)
        # value range
        r = 1 / math.sqrt(shape[0])
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)

    def init_AW(self):
        '''
        Talathi & Vartak 2016: Improving Performance of Recurrent Neural Network
        with ReLU Nonlinearity https://arxiv.org/abs/1511.03771.
        '''
        matrix_random = torch.randn(self.dz, self.dz)
        matrix_positive_normal = (1 / self.dz) * matrix_random.T @ matrix_random
        matrix = torch.eye(self.dz) + matrix_positive_normal
        max_ev = torch.max(torch.abs(torch.linalg.eigvals(matrix)))
        matrix_spectral_norm_one = matrix / max_ev
        return nn.Parameter(matrix_spectral_norm_one, requires_grad=True)

    def clip_z_to_range(self, z):
        if self.clip_range is not None:
            torch.clip_(z, -self.clip_range, self.clip_range)
        return z

class PLRNN_Step(Latent_Step):

    def __init__(self, *args, alpha=1, **kwargs):
        super(PLRNN_Step, self).__init__(*args, **kwargs)
        self.AW = self.init_AW()
        self.h = self.init_uniform((self.dz, ))
        self.alpha = alpha

    def get_latent_parameters(self):
        '''
        Split the AW matrix and return A, W, h.
        A is returned as a 1d vector!
        '''
        AW = self.AW
        A = torch.diag(AW)
        W = AW - torch.diag(A)
        h = self.h
        return A, W, h

    def forward(self, hidden_in):
        z = hidden_in
        A, W, h = self.get_latent_parameters()
        z_activated = torch.relu(self.norm(z))
        z = self.alpha * (A * z + z_activated @ W.t() + h) + (1 - self.alpha) * z
        return self.clip_z_to_range(z)