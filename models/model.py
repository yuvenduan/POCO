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
from models.TOTEM.models.decode import XcodeYtimeDecoder
from models.poyo import POYO
from models.normalizer import MuStdWrapper, BatchedLinear
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
        self.normalizer = MuStdWrapper(config, datum_size)

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
        input = self.normalizer.normalize(input)

        for step in range(pred_step):
            preds = self._forward(input)
            # concatenate the prediction to the input, and use it as the input for the next step
            if self.target_mode == 'raw':
                input = [torch.cat([x, pred[-1: ]], dim=0) for x, pred in zip(input, preds)]
            elif self.target_mode == 'derivative':
                input = [torch.cat([x, pred[-1: ] + x[-1: ]], dim=0) for x, pred in zip(input, preds)]
            else:
                raise ValueError(f"Unknown target mode {self.target_mode}")

        preds = self.normalizer.unnormalize(preds)
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
        self.normalizer = MuStdWrapper(config, datum_size)

        assert self.teacher_forcing, "Only support teacher forcing for now"
        assert config.stimuli_dim == 0, "Stimuli not supported for now"

    def forward(self, input, pred_step=1):
        """
        :param x: list of tensors, each of shape (L, B, D)
        """

        bsz = [xx.size(1) for xx in input]
        input = self.normalizer.normalize(input)
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
        x = self.normalizer.unnormalize(x)
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
        if config.encoder_type == 'vqvae':
            assert config.encoder_dir is not None, "Pre-trained encoder dir must be provided for vqvae"
            model_path = os.path.join(config.encoder_dir, config.encoder_state_dict_file)
            self.encoder = get_pretrained_model(model_path, config.encoder_dir, datum_size)
            self.TC = self.Tin // self.encoder.compression_factor
            self.encoder_type = 'vqvae'
            self.embed_dim = self.encoder.embedding_dim
        elif config.encoder_type == 'cnn':
            self.encoder = nn.Sequential(nn.Conv1d(1, config.conv_channels, config.kernel_size, config.conv_stride), nn.ReLU())
            self.TC = (self.Tin - config.kernel_size - 1) // config.conv_stride + 1
            self.encoder_type = 'cnn'
            self.embed_dim = config.conv_channels
        else:
            self.encoder = None
            self.TC = self.Tin // config.compression_factor
            self.encoder_type = 'none'
            self.embed_dim = config.compression_factor
        self.T_step = self.Tin // self.TC

        if config.population_token:
            self.population_projs = nn.ModuleList([
                nn.Sequential(nn.Linear(size * self.embed_dim, config.population_token_dim), nn.ReLU()) 
            for size in datum_size])
            self.pre_embed_dim = self.embed_dim
            self.embed_dim = config.population_token_dim

        self.datum_size = datum_size
        self.mu_std_module_mode = config.mu_std_module_mode
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
            poyo_query_mode = config.poyo_query_mode
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
                unit_dropout=config.poyo_unit_dropout,
                output_latent=config.poyo_output_mode == 'latent',
                t_max=config.rotary_attention_tmax
            )
            if config.poyo_output_mode == 'latent':
                self.embedding_dim = config.poyo_num_latents * config.decoder_hidden_size
            else:
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

        self.normalizer = MuStdWrapper(config, datum_size)
        
    def forward(self, x, pred_step=1, unit_indices=None, unit_timestamps=None):
        """
        x: list of tensors, each of shape (L, B, D)
        return: list of tensors, each of shape (pred_length, B, D)
        """
        # TODO: delete embed length and population token, if they don't work

        bsz = [xx.size(1) for xx in x]
        L = x[0].size(0)
        x = self.normalizer.normalize(x, concat_output=True)

        if self.encoder_type == 'vqvae':
            with torch.no_grad():
                out = self.encoder.encode(x) # out: sum(B * D), TC, E
        elif self.encoder_type == 'cnn':
            x = x.unsqueeze(1)
            out = self.encoder(x) # out: sum(B * D), C, TC
            out = out.permute(0, 2, 1) # out: sum(B * D), TC, C
        elif self.encoder_type == 'none':
            out = x.reshape(x.shape[0], self.TC, L // self.TC) # out: sum(B * D), TC, E
        else:
            raise ValueError(f"Unknown encoder type {self.encoder_type}")
        
        if hasattr(self, 'population_projs'):
            split_size = [b * d for b, d in zip(bsz, self.datum_size)]
            E = out.shape[-1]
            out = torch.split(out, split_size, dim=0) # [B * D, TC, E]
            out = [x.reshape(b, d, self.TC, E).permute(0, 2, 1, 3).reshape(b, self.TC, d * E) for x, b, d in zip(out, bsz, self.datum_size)] # [B, TC, D * E]
            out = [proj(xx) for proj, xx in zip(self.population_projs, out)] # [B, TC, E]
            out = torch.cat(out, dim=0) # sum(B), TC, E
            d_list = [1 for _ in self.datum_size]
        else:
            d_list = self.datum_size

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
                unit_timestamps = torch.zeros_like(unit_indices).unsqueeze(1) + torch.arange(0, self.Tin, self.T_step, device=x.device) # sum(B * D), TC

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
                pred_step=pred_step,
            )
            if embed.dim() == 3:
                embed = embed.reshape(embed.shape[0], -1) # sum(B * D), pred_length
                d_list = [1 for _ in self.datum_size]

        else:
            embed = self.decoder(out) # sum(B * D), embed

        # partition embed to a list of tensors, each of shape (B, D, embed)
        split_size = [b * d * self.embed_length for b, d in zip(bsz, d_list)]
        embed = torch.split(embed, split_size, dim=0) # [B * D, embed]
        embed = [xx.reshape(b, d, self.embed_length, self.embedding_dim) for xx, b, d in zip(embed, bsz, d_list)]

        preds = []
        for i, (e, d) in enumerate(zip(embed, self.datum_size)):
            if e.shape[1] < d:
                assert e.shape[1] == 1
                e = e.repeat(1, d, 1, 1)

            if self.separate_projs:
                proj = self.proj[i]
            else:
                proj = self.proj
            
            if e.shape[2] > 1:
                b = e.shape[0]
                e = e.permute(0, 2, 1, 3)
                e = e.reshape(-1, e.shape[2], e.shape[3]) # B * pred_length, D, embed_dim

                pred: torch.Tensor = proj(e) # B * pred_length, D, 1
                pred = pred.reshape(b, pred_step, d).permute(0, 2, 1) # B, D, pred_length
            else:
                pred: torch.Tensor = proj(e.squeeze(2)) # B, D, pred_length
            assert pred.shape[2] == pred_step 
            
            preds.append(pred.permute(2, 0, 1))

        preds = self.normalizer.unnormalize(preds)
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