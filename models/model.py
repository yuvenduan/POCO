__all__ = ['SimpleRNN', ]

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from configs.configs import SupervisedLearningBaseConfig

from models.model_utils import get_rnn, CTRNNCell
from configs.config_global import DEVICE
import models

class SimpleRNN(nn.Module):

    def __init__(self, config: SupervisedLearningBaseConfig, datum_size):
        super().__init__()

        self.in_proj = nn.ModuleList([nn.Linear(size, config.hidden_size) for size in datum_size])
        
        if config.shared_backbone:
            self.rnn = get_rnn(config.rnn_type, config.hidden_size, config.hidden_size, num_layers=config.num_layers)
        else:
            self.rnns = nn.ModuleList([get_rnn(config.rnn_type, config.hidden_size, size) for size in datum_size])
        self.shared_backbone = config.shared_backbone

        self.out_proj = nn.ModuleList([nn.Linear(config.hidden_size, size) for size in datum_size])
        self.out_sizes = datum_size
        self.teacher_forcing = config.teacher_forcing
        assert self.teacher_forcing, "Only support teacher forcing for now"

    def forward(self, x, pred_step=1):
        """
        :param x: list of tensors, each of shape (L, B, D)
        """

        bsz = [xx.size(1) for xx in x]
        x = [proj(xx) for proj, xx in zip(self.in_proj, x)]

        if self.shared_backbone:
            x = torch.cat(x, dim=1) 
            for step in range(pred_step):
                ret = self.rnn(x)
                if isinstance(ret, tuple):
                    new_x = ret[0]
                else:
                    new_x = ret
                x = torch.cat([x, new_x[-1: ]], dim=0)
            x = x[1: ]
            x = torch.split(x, bsz, dim=1)
        else:
            for idx in range(len(x)):
                xx = x[idx]
                if xx.shape[1] == 0:
                    x[idx] = torch.zeros((xx.shape[0] + pred_step - 1, 0, xx.shape[2]), device=xx.device)
                    continue
                for step in range(pred_step):
                    new_xx = self.rnns[idx](xx)
                    xx = torch.cat([xx, new_xx[-1: ]], dim=0)
                x[idx] = xx[1: ]

        x = [proj(xx) for proj, xx in zip(self.out_proj, x)]

        return x
    
    def set_mode(self, mode):
        pass

class MAML(nn.Module):

    def __init__(self, config: SupervisedLearningBaseConfig, datum_size):
        super().__init__()
        self.config = config
        self.rnns = nn.ModuleList(
            [get_rnn('CTRNNCell', None, size, alpha=config.alpha, learnable_alpha=config.learnable_alpha) 
             for size in datum_size])
        self.datum_size = datum_size
        
    def get_inner_params(self, idx):
        fast_weights = [nn.Parameter(torch.zeros_like(p)) for p in self.rnns[idx].parameters()]
        fast_weights = nn.ParameterList(fast_weights).to(DEVICE)
        return fast_weights
    
    def forward(self, x, idx, inner_params, pred_step=1, teacher_forcing=True):
        # x: (1: L, D) -> (2: L + pred_step, D)
        # if np.random.rand() < 0.001:
        #    print('inner param:', inner_params[0].data, 'init alpha', self.rnns[idx].alpha.data, flush=True)
        total_length = x.shape[0] + pred_step - 1
        if teacher_forcing:
            x = self.rnns[idx](None, x, inner_params)
        else:
            x = self.rnns[idx](None, x[: 1], inner_params)

        last_hidden = x[-1]
        pred_list = []
        for step in range(total_length - x.shape[0]):
            pred = self.rnns[idx](None, last_hidden, inner_params)
            pred_list.append(pred)
            last_hidden = pred

        if len(pred_list) > 0:
            x = torch.cat([x, torch.stack(pred_list)], dim=0)
        return x
    
class RNNBaseline(nn.Module):

    def __init__(self, config: SupervisedLearningBaseConfig, datum_size):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.config = config
        self.datum_size = datum_size
        
    def get_inner_params(self, idx):
        rnn = get_rnn(
            'CTRNNCell', None, self.datum_size[idx], 
            alpha=self.config.alpha, learnable_alpha=self.config.learnable_alpha
        ).to(DEVICE)
        return rnn
    
    def forward(self, x, idx, inner_params, pred_step=1, teacher_forcing=True):
        # x: (L, D)
        total_length = x.shape[0] + pred_step - 1
        if teacher_forcing:
            x = inner_params(None, x)
        else:
            x = inner_params(None, x[: 1])

        last_hidden = x[-1]
        pred_list = []
        for step in range(total_length - x.shape[0]):
            pred = inner_params(None, last_hidden)
            pred_list.append(pred)
            last_hidden = pred

        if len(pred_list) > 0:
            x = torch.cat([x, torch.stack(pred_list)], dim=0)
        return x

class MetaRNN(nn.Module):

    def __init__(self, config: SupervisedLearningBaseConfig, datum_size):
        super().__init__()

        if config.algorithm == 'maml':
            self.model = MAML(config, datum_size)
        elif config.algorithm == 'none':
            self.model = RNNBaseline(config, datum_size)
        else:
            raise ValueError(f"Unknown algorithm {config.algorithm}")
        self.datum_size = datum_size

        self.lr = config.inner_lr
        self.wd = config.inner_wd
        self.train_step = config.inner_train_step
        self.test_time_train_step = config.inner_test_time_train_step
        self.teacher_forcing = config.teacher_forcing

    def forward(self, x, pred_step=1):
        ret = []
        for id, acts in enumerate(x):
            # (L, B)
            if acts.size(1) == 0:
                ret.append(torch.zeros_like(acts))
                continue

            pred_list = []
            for idx in range(acts.size(1)):
                pred = self.train_rnn(acts[:, idx], id, pred_step=pred_step)
                pred_list.append(pred)
            pred_list = torch.stack(pred_list, dim=1)
            ret.append(pred_list)

        return ret
    
    def train_rnn(self, activity: torch.Tensor, id, pred_step=1):

        train_step = self.train_step if self.training else self.test_time_train_step
        
        with torch.set_grad_enabled(True):

            module: nn.Module = self.model.get_inner_params(id)
            loss_list = []

            for step in range(train_step):
                pred = self.model(activity[: -1], id, module, pred_step=1, teacher_forcing=self.teacher_forcing)
                target = activity[1:]
                loss = F.mse_loss(pred, target)
                grad = torch.autograd.grad(loss, module.parameters(), create_graph=True)
                loss_list.append(loss.item())

                # do gd
                for p, g in zip(module.parameters(), grad):
                    # print(p.shape, p.norm(), step, flush=True)
                    p.data = p.data - self.lr * g - self.wd * self.lr * p.data

        test_time_teacher_forcing = True
        if self.training and not self.teacher_forcing:
            test_time_teacher_forcing = False
        pred = self.model(activity, id, module, pred_step=pred_step, teacher_forcing=test_time_teacher_forcing)
        return pred
    
    def set_mode(self, mode):
        pass

class SeqVAE(nn.Module):
    
    def __init__(self, config: SupervisedLearningBaseConfig):
        super().__init__()
        assert config.encoder_rnn_type == 'BiGRU'
        assert config.decoder_rnn_type == 'CTRNN'
        self.initial_hidden = nn.Parameter(torch.zeros(2, config.encoder_hidden_size))
        self.encoder = get_rnn(config.encoder_rnn_type, config.encoder_hidden_size, config.encoder_hidden_size)
        self.mu_proj = nn.Linear(config.encoder_hidden_size * 2, config.decoder_hidden_size)
        self.logvar_proj = nn.Linear(config.encoder_hidden_size * 2, config.decoder_hidden_size)
        self.decoder = get_rnn(config.decoder_rnn_type, None, config.decoder_hidden_size)

    def forward(self, x, pred_step=1):
        """
        :param x: a tensor of shape (L, B, D)
        """
        L, B, D = x.shape
        out, h = self.encoder(x, self.initial_hidden.unsqueeze(1).repeat(1, B, 1))
        h = h.permute(1, 0, 2).reshape(B, -1)
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        out, h = self.decoder(inp=None, hidden_in=z, time_steps=L + pred_step)
        return out, kl_loss
    
    def set_mode(self, mode):
        pass

class MultiFishSeqVAE(nn.Module):

    def __init__(self, config: SupervisedLearningBaseConfig, datum_size):
        super().__init__()

        self.in_proj = nn.ModuleList([nn.Linear(size, config.encoder_hidden_size) for size in datum_size])
        if config.shared_backbone:
            self.vae = SeqVAE(config)
        else:
            self.vaes = nn.ModuleList([SeqVAE(config) for _ in datum_size])
        self.shared_backbone = config.shared_backbone
        self.out_proj = nn.ModuleList([nn.Linear(config.decoder_hidden_size, size) for size in datum_size])
        self.decoder_hidden_size = config.decoder_hidden_size

    def forward(self, x, pred_step=1):
        """
        :param x: list of tensors, each of shape (L, B, D)
        """

        bsz = [xx.size(1) for xx in x]
        x = [proj(xx) for proj, xx in zip(self.in_proj, x)]

        if self.shared_backbone:
            x = torch.cat(x, dim=1) 
            out, kl_loss = self.vae(x, pred_step=pred_step)
            out = torch.split(out, bsz, dim=1)
        else:
            out = []
            kl_loss = 0
            for idx in range(len(x)):
                xx = x[idx]
                if xx.shape[1] == 0:
                    out.append(torch.zeros((xx.shape[0] + pred_step, xx.shape[1], self.decoder_hidden_size), device=xx.device))
                    continue
                out_, kl_loss_ = self.vaes[idx](xx, pred_step=pred_step)
                out.append(out_)
                kl_loss += kl_loss_
        out = [proj(xx) for proj, xx in zip(self.out_proj, out)]
        self.kl_loss = kl_loss

        return out
    
    def set_mode(self, mode):
        pass

class Conv(nn.Module):

    def __init__(self, config: SupervisedLearningBaseConfig, datum_size):
        super().__init__()
        self.kernel_size = config.kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, bias=False)
        self.conv.weight.data.fill_(0)
        self.conv.weight.data[:, :, -1] = 1
        self.config = config

    def forward(self, x, pred_step=1):
        ret = []
        for acts in x:
            L, B, D = acts.shape
            assert L >= self.kernel_size
            for step in range(pred_step):
                x = acts[-self.kernel_size: ]
                x = x.permute(1, 2, 0).unsqueeze(2)
                x = x.view(B * D, 1, self.kernel_size)
                x = self.conv(x)
                x = x.reshape(B, D, 1)
                x = x.permute(2, 0, 1)
                acts = torch.cat([acts, x], dim=0)
            ret.append(acts[1: ])

        return ret
    
    def set_mode(self, mode):
        pass