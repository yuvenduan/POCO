__all__ = ['MuStdModel', 'MuStdWrapper', 'BatchedLinear']

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from configs.configs import SupervisedLearningBaseConfig, NeuralPredictionConfig

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
        self.mu_std_module_mode = config.mu_std_module_mode if self.normalize_input else 'none'
        self.datum_size = datum_size
        print(f"Normalizer: normalize_input {self.normalize_input}, mu_std_module_mode {self.mu_std_module_mode}")

        self.mustd = MuStdModel(self.Tin, datum_size=datum_size, separate_projs=config.mu_std_separate_projs)

    def normalize(self, x, concat_output=False):
        """
        x: list of tensors, each of shape (L, B, D)
        concat_output: if True, return a single tensor of shape (sum(B * D), L), otherwise return a list of tensors, each of shape (L, B, D)
        """

        self.bsz = bsz = [xx.size(1) for xx in x]
        self.L = L = x[0].size(0)

        mu, std = self.mustd(x).chunk(2, dim=1)

        x = torch.cat([xx.reshape(L, b * d) for xx, b, d in zip(x, bsz, self.datum_size)], dim=1).transpose(0, 1) # sum(B * D), L
        x_mean, x_std = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True) # x_mean, x_std: sum(B * D), 1

        if self.normalize_input:
            x = (x - x_mean) / (x_std + 1e-6)

        if self.mu_std_module_mode == 'original':
            mu, std = x_mean, x_std
        elif self.mu_std_module_mode == 'learned':
            pass
        elif self.mu_std_module_mode == 'combined':
            mu, std = mu + x_mean, std + x_std
        elif self.mu_std_module_mode == 'combined_mu_only':
            mu = mu + x_mean
            std = torch.ones_like(std)
        elif self.mu_std_module_mode == 'none':
            mu = torch.zeros_like(mu)
            std = torch.ones_like(std)
        else:
            raise ValueError(f"Unknown mu_std_module_mode {self.mu_std_module_mode}")
        
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