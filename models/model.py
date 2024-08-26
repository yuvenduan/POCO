__all__ = ['AutoregressiveModel', 'LatentModel', 'SeqVAE', 'Linear', 'MultiFishSeqVAE', 'Decoder', 'MixedModel']

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import itertools
import os
from configs.configs import SupervisedLearningBaseConfig, NeuralPredictionConfig

from models.model_utils import get_rnn, get_pretrained_model, get_rnn_from_config
from models.TOTEM.models.decode import MuStdModel, XcodeYtimeDecoder
from models.poyo import POYO
from configs.config_global import DEVICE

class AutoregressiveModel(nn.Module):
    """
    Au autoregressive model that predicts the next step based on the previous steps
    """

    def __init__(self, config: NeuralPredictionConfig, datum_size):
        """
        datum_size: list of integers. For multi-fish neural prediction, it is the number of observed 
            channels of each fish. if pc_dim is not None (only predict the PCs), datum_size = [pc_dim] * num_fish
        """
        super().__init__()

        self.in_proj = nn.ModuleList([nn.Linear(size, config.hidden_size) for size in datum_size])
        
        if config.shared_backbone:
            self.rnn = get_rnn_from_config(config, rnn_in_size=config.hidden_size)
        else:
            self.rnns = nn.ModuleList([get_rnn_from_config(config, rnn_in_size=config.hidden_size) for size in datum_size])
        
        self.stimuli_dim = config.stimuli_dim
        if config.stimuli_dim > 0:
            self.stimuli_proj = nn.Linear(config.stimuli_dim, config.hidden_size)
        self.shared_backbone = config.shared_backbone

        self.out_proj = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(config.hidden_size) if config.rnn_layernorm else nn.Identity(), 
            nn.Linear(config.hidden_size, size)
        ) for size in datum_size])
        self.out_sizes = datum_size
        self.teacher_forcing = config.teacher_forcing
        self.target_mode = config.target_mode
        assert self.teacher_forcing, "Only support teacher forcing for now"

    def _forward(self, input):
        """
        One step forward
        input: list of tensors, each of shape (L, B, D)
        return: list of tensors, each of shape (L, B, D)
        """
        bsz = [xx.size(1) for xx in input]
        x = [proj(xx[:, :, : xx.shape[2] - self.stimuli_dim]) for proj, xx in zip(self.in_proj, input)]
        if self.stimuli_dim > 0:
            stimuli = [self.stimuli_proj(xx[:, :, -self.stimuli_dim: ]) for xx in input]
            x = [xx + stim for xx, stim in zip(x, stimuli)]

        if self.shared_backbone:
            x = torch.cat(x, dim=1)
            ret = self.rnn(x)
            new_x = ret[0] if isinstance(ret, tuple) else ret
            x = torch.split(new_x, bsz, dim=1)
        else:
            for idx in range(len(x)):
                xx = x[idx]
                if xx.shape[1] == 0:
                    x[idx] = torch.zeros((xx.shape[0], 0, xx.shape[2]), device=xx.device)
                    continue
                new_xx = self.rnns[idx](xx)
                x[idx] = new_xx

        x = [proj(xx) for proj, xx in zip(self.out_proj, x)]
        return x

    def forward(self, input, pred_step=1):
        """
        :param x: list of tensors, each of shape (L, B, D)
        :param pred_step: number of steps to predict
        :return: list of tensors, each of shape (L + pred_step - 1, B, D)
        """
        for step in range(pred_step):
            preds = self._forward(input)
            # concatenate the prediction to the input, and use it as the input for the next step
            if self.target_mode == 'raw':
                input = [torch.cat([x, pred[-1: ]], dim=0) for x, pred in zip(input, preds)]
            elif self.target_mode == 'derivative':
                input = [torch.cat([x, pred[-1: ] + x[-1: ]], dim=0) for x, pred in zip(input, preds)]
            else:
                raise ValueError(f"Unknown target mode {self.target_mode}")

        return preds
    
    def set_mode(self, mode):
        pass

class Latent_to_Obs(nn.Module):

    def __init__(self, config: NeuralPredictionConfig, datum_size):
        super().__init__()
        self.latent_to_ob_mode = config.latent_to_ob
        if config.latent_to_ob == 'linear':
            self.projs = nn.ModuleList([nn.Linear(config.hidden_size, size) for size in datum_size])
        elif config.latent_to_ob == 'identity':
            self.projs = [nn.Identity() for _ in datum_size]
        else:
            raise ValueError(f"Unknown latent_to_ob mode {config.latent_to_ob}")
        self.out_sizes = datum_size

    def latent_to_obs(self, z):
        """
        :param z: list of tensors, each of shape (L, B, D)
        """
        x = [proj(xx) for proj, xx in zip(self.projs, z)]
        return x
        
    def forward(self, z):
        return self.latent_to_obs(z)
    
    def obs_to_latent(self, x):
        """
        Infer latent states from observations via the pseudo-inverse of the projection matrix
        :param x: list of tensors, each of shape (L, B, D)
        """
        z = []
        for proj, xx in zip(self.projs, x):
            if self.latent_to_ob_mode == 'linear':
                W = proj.weight
                xx = xx - proj.bias
                W_PI = torch.linalg.pinv(W)
                z.append(F.linear(xx, W_PI))
            elif self.latent_to_ob_mode == 'identity':
                z.append(xx)
        return z

class LatentModel(nn.Module):

    def __init__(self, config: NeuralPredictionConfig, datum_size):
        super().__init__()

        self.proj = Latent_to_Obs(config, datum_size)
        if config.shared_backbone:
            assert config.num_layers == 1
            self.rnn = get_rnn_from_config(config, rnn_in_size=None)
        else:
            hidden_sizes = [config.hidden_size] * len(datum_size) if config.latent_to_ob != 'identity' else datum_size
            self.rnns = nn.ModuleList([get_rnn_from_config(config, rnn_in_size=None, hidden_size=size) for size in hidden_sizes])
        self.shared_backbone = config.shared_backbone
        self.out_sizes = datum_size
        self.teacher_forcing = config.teacher_forcing
        self.train_tf_interval = config.tf_interval
        self.target_mode = config.target_mode

        assert self.teacher_forcing, "Only support teacher forcing for now"
        assert config.stimuli_dim == 0, "Stimuli not supported for now"

    def forward(self, input, pred_step=1):
        """
        :param x: list of tensors, each of shape (L, B, D)
        """

        bsz = [xx.size(1) for xx in input]
        z = self.proj.obs_to_latent(input)

        if self.shared_backbone:
            z = torch.cat(z, dim=1)

            # use sparse teacher forcing
            z_list = [z[0]]
            for step in range(z.shape[0] - 1):
                if step % self.train_tf_interval == 0:
                    last_z = z[step]
                ret = self.rnn(hidden_in=last_z)
                last_z = ret[0] if isinstance(ret, tuple) else ret
                z_list.append(last_z)

            last_z = z[-1]    
            for step in range(pred_step):
                ret = self.rnn(hidden_in=last_z)
                last_z = ret[0] if isinstance(ret, tuple) else ret
                z_list.append(last_z)

            z = torch.stack(z_list[1: ])
            z = torch.split(z, bsz, dim=1)
        else:
            for idx in range(len(z)):
                zz = z[idx]
                if zz.shape[1] == 0:
                    z[idx] = torch.zeros((zz.shape[0] + pred_step - 1, 0, zz.shape[2]), device=zz.device)
                    continue
                z_list = [zz[0]]
                for step in range(zz.shape[0] - 1):
                    if step % self.train_tf_interval == 0:
                        last_z = zz[step]
                    new_z = self.rnns[idx](hidden_in=last_z)
                    last_z = new_z
                    z_list.append(last_z)
                last_z = zz[-1]
                for step in range(pred_step):
                    new_z = self.rnns[idx](hidden_in=last_z)
                    last_z = new_z
                    z_list.append(last_z)
                z_list = z_list[1: ]
                z[idx] = torch.stack(z_list)

        x = self.proj.latent_to_obs(z)
        return x
    
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

        assert config.loss_mode[: 14] == 'reconstruction'

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

class BatchedLinear(nn.Module):

    def __init__(self, n_channels, in_size, out_size, bias=False, init='one_hot'):
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
            self.weight.data = self.weight[:, 0: 1, :] * torch.ones_like(self.weight)
        elif init == 'fan_out':
            bound = 1 / math.sqrt(out_size)
            nn.init.uniform_(self.weight, -bound, bound)
            self.weight.data = self.weight[:, 0: 1, :] * torch.ones_like(self.weight)
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

class Linear(nn.Module):

    def __init__(self, config: NeuralPredictionConfig, datum_size):
        super().__init__()

        self.kernel_size = config.kernel_size
        self.datum_size = datum_size
        self.use_bias = config.use_bias
        self.kernel_size = config.kernel_size
        self.in_length = config.linear_input_length
        self.out_length = config.pred_length

        self.shared_backbone = config.shared_backbone
        self.per_channel = config.per_channel

        if config.per_channel:
            if config.shared_backbone:
                self.linear = BatchedLinear(datum_size[0], self.in_length, self.out_length, bias=self.use_bias)
                assert np.equal(datum_size, datum_size[0]).all()
            else:
                self.linear = nn.ModuleList([BatchedLinear(size, self.in_length, self.out_length, bias=self.use_bias) for size in datum_size])
        else:
            assert config.shared_backbone
            self.linear = nn.Linear(self.in_length, self.out_length, bias=self.use_bias)
            self.linear.weight.data.fill_(0)
            self.linear.weight.data[:, -1] = 1

    def forward(self, x, pred_step=1):
        """
        :param x: list of tensors, each of shape (L, B, D)
        """
        ret = []
        for i, xx in enumerate(x):
            xx = xx[-self.in_length: ].permute(1, 2, 0) # B, D, L
            linear_model = self.linear if self.shared_backbone else self.linear[i]
            xx = linear_model(xx)
            xx = xx.permute(2, 0, 1)
            ret.append(xx)
        return ret
    
    def set_mode(self, mode):
        pass

class Decoder(nn.Module):
    """
    Decoder based on pre-trained features
    """

    def __init__(self, config: NeuralPredictionConfig, datum_size):
        super().__init__()

        self.Tin = config.seq_length - config.pred_length
        if config.encoder_dir is not None:
            model_path = os.path.join(config.encoder_dir, config.encoder_state_dict_file)
            self.encoder = get_pretrained_model(model_path, config.encoder_dir, datum_size)
            self.TC = self.Tin // self.encoder.compression_factor
            self.embed_dim = self.encoder.embedding_dim
        else:
            self.encoder = None
            self.TC = self.Tin // config.compression_factor
            self.embed_dim = config.compression_factor
        self.T_step = self.Tin // self.TC
        
        self.datum_size = datum_size
        if config.mu_std_loss_coef is None:
            self.mu_std_loss_coef = 0
            self.normalize_seq = False
        else:
            self.mu_std_loss_coef = config.mu_std_loss_coef
            self.normalize_seq = True
        self.embed_length = 1

        if config.decoder_type == 'Linear':
            self.decoder = nn.Flatten(1, 2)
            self.embedding_dim = self.TC * self.embed_dim
        elif config.decoder_type == 'MLP':
            self.decoder = nn.Sequential(
                nn.Flatten(1, 2), 
                nn.Linear(self.TC * self.embed_dim, config.decoder_hidden_size), 
                nn.ReLU()
            )
            self.embedding_dim = config.decoder_hidden_size
        elif config.decoder_type == 'Transformer':
            self.decoder = XcodeYtimeDecoder(
                self.embed_dim, 
                config.decoder_hidden_size, 
                config.decoder_num_heads,
                config.decoder_hidden_size, 
                config.decoder_num_layers
            )
            self.embedding_dim = config.decoder_hidden_size * self.TC
        elif config.decoder_type == 'POYO':
            poyo_query_mode = config.poyo_query_mode if hasattr(config, 'poyo_query_mode') else 'single'
            self.embed_length = 1 if poyo_query_mode == 'single' else config.pred_length
            self.decoder = POYO(
                input_dim=self.embed_dim, 
                dim=config.decoder_hidden_size, 
                depth=config.decoder_num_layers, 
                self_heads=config.decoder_num_heads,
                datum_size=datum_size,
                num_latents=config.poyo_num_latents,
                query_length=self.embed_length,
                T_step=self.T_step,
                unit_dropout=config.poyo_unit_dropout if hasattr(config, 'poyo_unit_dropout') else 0,
            )
            self.embedding_dim = config.decoder_hidden_size
        else:
            raise NotImplementedError
        
        self.separate_projs = config.separate_projs
        self.linear_out_size = config.pred_length
        if config.decoder_type == 'POYO' and poyo_query_mode == 'multi':
            self.linear_out_size = 1

        if not self.separate_projs:
            self.proj = nn.Linear(self.embedding_dim, self.linear_out_size)
        else:
            self.proj = nn.ModuleList([BatchedLinear(size, self.embedding_dim, self.linear_out_size, init=config.decoder_proj_init) for size in datum_size])

        self.mustd = MuStdModel(self.Tin, config.pred_length, [64])
        
    def forward(self, x, pred_step=1, unit_indices=None, unit_timestamps=None):
        """
        x: list of tensors, each of shape (L, B, D)
        return: list of tensors, each of shape (pred_length, B, D)
        """
        bsz = [xx.size(1) for xx in x]
        L = x[0].size(0)
        x = torch.cat([xx.reshape(L, b * d) for xx, b, d in zip(x, bsz, self.datum_size)], dim=1).transpose(0, 1) # sum(B * D), L

        mu, std = self.mustd(x).chunk(2, dim=1)
        if self.normalize_seq:
            x_mean, x_std = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True) # x_mean, x_std: sum(B * D), 1
            x = (x - x_mean) / (x_std + 1e-6)

        if self.encoder is not None:
            with torch.no_grad():
                out = self.encoder.encode(x) # out: sum(B * D), TC, E
        else:
            out = x.reshape(x.shape[0], self.TC, self.embed_dim) # out: sum(B * D), TC, E

        if isinstance(self.decoder, POYO):
            if unit_indices is None:
                sum_channels = 0
                unit_indices = []
                for b, d in zip(bsz, self.datum_size):
                    indices = torch.arange(d, device=x.device).unsqueeze(0).repeat(b, 1).reshape(-1) # B * D
                    unit_indices.append(indices + sum_channels)
                    sum_channels += d
                unit_indices = torch.cat(unit_indices, dim=0) # sum(B * D)
            if unit_timestamps is None:
                unit_timestamps = torch.zeros_like(unit_indices).unsqueeze(1) + torch.arange(0, self.Tin, self.T_step, device=x.device) # sum(B * D), L

            input_seqlen = torch.cat([torch.full((b, ), d, device=x.device) 
                                            for b, d in zip(bsz, self.datum_size)], dim=0)
            session_index = torch.cat([torch.full((b, d), i, device=x.device).reshape(-1) 
                                            for i, (b, d) in enumerate(zip(bsz, self.datum_size))], dim=0)

            embed = self.decoder(
                out,
                unit_indices=unit_indices,
                unit_timestamps=unit_timestamps,
                input_seqlen=input_seqlen,
                session_index=session_index,
                pred_step=pred_step
            )
        else:
            embed = self.decoder(out) # sum(B * D), embed
        
        if self.normalize_seq:
            mu = x_mean + mu
            std = x_std + std
        self.mu = mu.clone().squeeze(-1)
        self.std = std.clone().squeeze(-1)

        # partition embed, to a list of tensors, each of shape (B, D, embed)
        split_size = [b * d * self.embed_length for b, d in zip(bsz, self.datum_size)]
        embed = torch.split(embed, split_size, dim=0) # [B * D, embed]
        embed = [xx.reshape(b, d, self.embed_length, self.embedding_dim) for xx, b, d in zip(embed, bsz, self.datum_size)]

        split_size = [b * d for b, d in zip(bsz, self.datum_size)]
        mu = torch.split(mu, split_size, dim=0) # [B * D, 1]
        mu = [xx.reshape(b, d) for xx, b, d in zip(mu, bsz, self.datum_size)]
        std = torch.split(std, split_size, dim=0) # [B * D, 1]
        std = [xx.reshape(b, d) for xx, b, d in zip(std, bsz, self.datum_size)]

        preds = []
        for i, (e, m, s) in enumerate(zip(embed, mu, std)):
            if self.separate_projs:
                proj = self.proj[i]
            else:
                proj = self.proj
            
            if e.shape[2] > 1:
                b, d = e.shape[0], e.shape[1]
                e = e.permute(0, 2, 1, 3)
                e = e.reshape(-1, e.shape[2], e.shape[3]) # B * pred_length, D, embed_dim

                pred: torch.Tensor = proj(e) # B * pred_length, D, 1
                pred = pred.reshape(b, pred_step, d).permute(0, 2, 1) # B, D, pred_length
            else:
                pred: torch.Tensor = proj(e.squeeze(2)) # B, D, pred_length
            assert pred.shape[2] == pred_step 
            
            # pred = (pred - pred.mean(dim=-1, keepdims=True)) / pred.std(dim=-1, keepdims=True) # normalized
            if self.normalize_seq:
                pred = pred * s.unsqueeze(-1) + m.unsqueeze(-1)
            preds.append(pred.permute(2, 0, 1))

        return preds

class MixedModel(nn.Module):

    def __init__(self, config: NeuralPredictionConfig, datum_size):
        super().__init__()

        self.latent_model = LatentModel(config, datum_size)

        config.shared_backbone = False
        config.per_channel = True
        self.linear_model = Linear(config, datum_size)

    def forward(self, x, pred_step=1):
        """
        :param x: list of tensors, each of shape (L, B, D)
        """
        linear_out = self.linear_model(x, pred_step=pred_step)
        latent_out = self.latent_model(x, pred_step=pred_step)
        for i in range(len(linear_out)):
            # pad linear_out to the same length as latent_out
            padding = torch.zeros((latent_out[i].shape[0] - linear_out[i].shape[0], linear_out[i].shape[1], linear_out[i].shape[2]), device=linear_out[i].device)
            padded_linear_out = torch.cat([linear_out[i], padding], dim=0)
            x[i] = padded_linear_out + latent_out[i]
        return x