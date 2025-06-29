__all__ = ['MuStdModel', 'MuStdWrapper', 'BatchedLinear', 'RevIN']

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from configs.configs import SupervisedLearningBaseConfig, NeuralPredictionConfig

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class BatchedLinear(nn.Module):

    def __init__(self, n_channels, in_size, out_size, bias=False, init='fan_in'):
        # weight: n * out_size * in_size
        # input: batch * n * in_size -> n * in_size * batch
        # out: n * out_size * batch -> batch * n * out_size
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_channels, out_size, in_size))
        if init == 'one_hot':
            nn.init.zeros_(self.weight)
            self.weight.data[:, :, -1] = 1
        elif init == 'fan_in':
            bound = 1 / math.sqrt(in_size)
            nn.init.uniform_(self.weight, -bound, bound)
            # self.weight.data = self.weight[:, 0: 1, :] * torch.ones_like(self.weight)
        elif init == 'fan_out':
            bound = 1 / math.sqrt(out_size)
            nn.init.uniform_(self.weight, -bound, bound)
            # self.weight.data = self.weight[:, 0: 1, :] * torch.ones_like(self.weight)
        elif init == 'zero':
            nn.init.zeros_(self.weight)
        else:
            raise ValueError(f"Unknown init {init}")

        if bias:
            self.bias = nn.Parameter(torch.zeros(n_channels, out_size))

    def forward(self, x):
        # x: batch * n * in_size
        # return: batch * n * out_size
        x = x.permute(1, 2, 0)
        x = torch.bmm(self.weight, x)
        if hasattr(self, 'bias'):
            x = x + self.bias.unsqueeze(2)
        x = x.permute(2, 0, 1)
        return x

class MuStdModel(nn.Module):
    def __init__(self, Tin, hidden_dim=64, datum_size=None, separate_projs=False):
        super(MuStdModel, self).__init__()

        self.linear = nn.Linear(Tin + 2, hidden_dim)
        if not separate_projs:
            self.proj = nn.Linear(hidden_dim, 2)
        else:
            self.proj = nn.ModuleList([BatchedLinear(size, hidden_dim, 2) for size in datum_size])
        self.separate_projs = separate_projs

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                nn.init.zeros_(p)

    def forward(self, x_list: torch.Tensor) -> torch.Tensor:
        """
        x: list of tensors, each of shape (L, B, D)
        return: a tensor of shape (sum(B * D), 2), representing the predicted mean and std
        """
        results = []
        for i, x in enumerate(x_list):
            if x.size(1) == 0:
                results.append(torch.zeros((0, 2), device=x.device))
                continue
            mean = torch.mean(x, dim=0, keepdim=True)
            std = torch.sqrt(torch.var(x, dim=0, keepdim=True, unbiased=False) + 1e-5)
            x = torch.cat((x, mean, std), dim=0) # L + 2, B, D
            x = x.permute(1, 2, 0) # B, D, L + 2
            x = self.linear(x)
            x = F.relu(x)
            proj = self.proj if not self.separate_projs else self.proj[i]
            x = proj(x) # B, D, 2
            results.append(x.reshape(-1, 2))
        return torch.cat(results, dim=0)

class MuStdWrapper(nn.Module):

    def __init__(self, config: NeuralPredictionConfig, datum_size):
        super(MuStdWrapper, self).__init__()

        self.Tin = config.seq_length - config.pred_length
        self.normalize_input = config.normalize_input
        self.mu_module_mode = config.mu_module_mode
        self.std_module_mode = config.std_module_mode

        self.datum_size = datum_size
        print(f"Normalizer: normalize_input {self.normalize_input}, mu_module_mode {self.mu_module_mode}, std_module_mode {self.std_module_mode}")

        self.mustd = MuStdModel(self.Tin, datum_size=datum_size, separate_projs=config.mu_std_separate_projs)

    def normalize(self, x_list, concat_output=False):
        """
        x_list: list of tensors, each of shape (L, B, D)
        concat_output: if True, return a single tensor of shape (sum(B * D), L), otherwise return a list of tensors, each of shape (L, B, D)
        """     

        self.bsz = bsz = [xx.size(1) for xx in x_list]
        self.L = L = x_list[0].size(0)
        x = torch.cat([xx.reshape(L, b * d) for xx, b, d in zip(x_list, bsz, self.datum_size)], dim=1).transpose(0, 1) # sum(B * D), L

        if not self.normalize_input and self.mu_module_mode == 'none' and self.std_module_mode == 'none':
            return x if concat_output else x_list

        mu, std = self.mustd(x_list).chunk(2, dim=1)
        x_mean, x_std = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True) # x_mean, x_std: sum(B * D), 1

        if self.mu_module_mode == 'last':
            mu = x[:, -1: ]
        elif self.mu_module_mode == 'original':
            mu = x_mean
        elif self.mu_module_mode == 'combined':
            mu = mu + x_mean
        elif self.mu_module_mode == 'last_combined':
            mu = mu + x[:, -1: ]
        elif self.mu_module_mode == 'learned':
            mu = mu
        elif self.mu_module_mode == 'none':
            mu = torch.zeros_like(mu)
        else:
            raise ValueError(f"Unknown mu_module_mode {self.mu_module_mode}")
        
        if self.normalize_input:
            x = (x - x_mean) / (x_std + 1e-6)
        
        if self.std_module_mode == 'original':
            std = x_std
        elif self.std_module_mode == 'learned':
            std = std
        elif self.std_module_mode == 'learned_exp':
            std = torch.exp(std)
        elif self.std_module_mode == 'learned_softplus':
            std = F.softplus(std)
        elif self.std_module_mode == 'combined':
            std = std + x_std
        elif self.std_module_mode == 'combined_exp':
            std = torch.exp(std + torch.log(x_std + 1e-6))
        elif self.std_module_mode == 'combined_softplus':
            val = torch.log(torch.exp(x_std) - 1 + 1e-6)
            std = F.softplus(std + val)
        elif self.std_module_mode == 'none':
            std = torch.ones_like(std)
        else:
            raise ValueError(f"Unknown std_module_mode {self.std_module_mode}")

        
        self.params = (mu, std)
        if not concat_output:
            split_size = [b * d for b, d in zip(bsz, self.datum_size)]
            x = torch.split(x.transpose(0, 1), split_size, dim=1) # L, sum(B * D) -> [L, B * D]
            x = [xx.reshape(L, b, d) for xx, b, d in zip(x, bsz, self.datum_size)]

        return x
    
    def unnormalize(self, x):
        """
        x: list of tensors, each of shape (L, B, D)
        return: a list of unnormalized tensors, each of shape (L, B, D)
        """

        if not self.normalize_input and self.mu_module_mode == 'none' and self.std_module_mode == 'none':
            return x

        bsz, L = self.bsz, self.L
        mu, std = self.params

        split_size = [b * d for b, d in zip(bsz, self.datum_size)]
        mu = torch.split(mu, split_size, dim=0) # [B * D, 1]
        mu = [xx.reshape(b, d) for xx, b, d in zip(mu, bsz, self.datum_size)]
        std = torch.split(std, split_size, dim=0) # [B * D, 1]
        std = [xx.reshape(b, d) for xx, b, d in zip(std, bsz, self.datum_size)]

        preds = []
        for xx, m, s, d in zip(x, mu, std, self.datum_size):
            preds.append(xx * s + m)

        self.params = None
        return preds

    def set_mode(self, mode):
        pass