__all__ = [
    'MultiSessionWrapper', 'MultiSessionSharedWrapper', 
    'Autoregressive', 'MultiAutoregressive', 
    'TCN', 'MultiTCN', 
    'DLinear', 'MultiDLinear', 
    'TexFilter', 'PaiFilter', 
    'Latent_to_Obs', 'LatentModel', 
    'Decoder', 'NetFormer', 'ARMLP'
]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import models.single_session_models as single_models
import itertools

from configs.configs import SupervisedLearningBaseConfig, NeuralPredictionConfig
from models.model_utils import get_rnn, get_pretrained_model, get_rnn_from_config
from models.TOTEM.models.decode import XcodeYtimeDecoder
from models.poyo.poyo import POYO
from models.layers.normalizer import MuStdWrapper, BatchedLinear, RevIN
from models.layers.netformer import BaseNetFormer
from configs.config_global import DEVICE

class MultiSessionWrapper(nn.Module):
    # A wrapper for models that work on a single session of data, define a model for each session

    def __init__(self, config: NeuralPredictionConfig, model_class, input_sizes, model_kwargs={}, add_normalizer=False):
        super().__init__()
        if len(input_sizes) > 1:
            print(f"WARNING: Using separate models for each session, consider using shared backbone instead")
        input_sizes = list(itertools.chain(*input_sizes))
        self.models = nn.ModuleList([model_class(config, input_size, **model_kwargs) for input_size in input_sizes])

        self.add_normalizer = add_normalizer
        if add_normalizer:
            self.normalizer = MuStdWrapper(config, input_sizes)
    
    def forward(self, x, **kwargs):
        if self.add_normalizer:
            x = self.normalizer.normalize(x, concat_output=False)
        preds = [model(xx, **kwargs) for model, xx in zip(self.models, x)]
        if self.add_normalizer:
            preds = self.normalizer.unnormalize(preds)
        return preds
    
class MultiSessionSharedWrapper(nn.Module):
    # a wrapper that assumes some shared latent space and separate linear projections for each session

    def __init__(self, config: NeuralPredictionConfig, model_class, input_sizes, model_kwargs={}, linear_proj=True, add_normalizer=False):
        super().__init__()
        input_sizes = list(itertools.chain(*input_sizes))
        self.shared_model = model_class(config, config.hidden_size, **model_kwargs)
        if linear_proj:
            print(f"Using shared backbone with dim {config.hidden_size}")
            self.in_projs = nn.ModuleList([nn.Linear(size, config.hidden_size) for size in input_sizes])
            self.out_projs = nn.ModuleList([nn.Linear(config.hidden_size, size) for size in input_sizes])
        else:
            print(f"Using shared backbone without linear projection, the model must be agnostic to input size")

        if add_normalizer:
            self.normalizer = MuStdWrapper(config, input_sizes)
        self.add_normalizer = add_normalizer
        
    def forward(self, x, **kwargs):
        # x: list of tensors, each of shape (L, B, D)
        if self.add_normalizer:
            x = self.normalizer.normalize(x, concat_output=False)
        if hasattr(self, 'in_projs'):
            x = [proj(xx) for proj, xx in zip(self.in_projs, x)]
        bsz = [xx.size(1) for xx in x]
        x = torch.cat(x, dim=1)
        x = self.shared_model(x, **kwargs)
        x = torch.split(x, bsz, dim=1)
        if hasattr(self, 'out_projs'):
            x = [proj(xx) for proj, xx in zip(self.out_projs, x)]
        if self.add_normalizer:
            x = self.normalizer.unnormalize(x)
        return x

class Autoregressive(MultiSessionWrapper):
    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, single_models.Autoregressive, input_size, add_normalizer=True)

class MultiAutoregressive(MultiSessionSharedWrapper):
    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, single_models.Autoregressive, input_size, model_kwargs={'omit_linear': True}, add_normalizer=True)

class TCN(MultiSessionWrapper):
    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, single_models.TCN, input_size)

class MultiTCN(MultiSessionSharedWrapper):
    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, single_models.TCN, input_size)

class DLinear(MultiSessionWrapper):
    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, single_models.DLinear, input_size)

class MultiDLinear(MultiSessionSharedWrapper):
    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, single_models.DLinear, input_size, linear_proj=False)

class MultiPaiFilter(MultiSessionSharedWrapper):
    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, single_models.PaiFilter, input_size, linear_proj=False)

class TSMixer(MultiSessionWrapper):
    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, single_models.TSMixer, input_size)

class MultiTSMixer(MultiSessionSharedWrapper):
    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, single_models.TSMixer, input_size)

class PaiFilter(nn.Module):

    def __init__(self, configs: NeuralPredictionConfig, input_size):
        super().__init__()
        input_size = list(itertools.chain(*input_size))
        self.seq_len = configs.seq_length - configs.pred_length
        self.pred_len = configs.pred_length
        self.scale = 0.02
        self.revin_layer = nn.ModuleList([RevIN(size, affine=True, subtract_last=False) for size in input_size])

        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size
        
        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x_list):
        pred_list = []

        for revin_layer, x in zip(self.revin_layer, x_list):
            x = x.permute(1, 0, 2) # L, B, D -> B, L, D

            if x.shape[0] == 0:
                pred_list.append(torch.zeros(0, x.shape[1], self.pred_len, device=x.device))
                continue

            z = x
            z = revin_layer(z, 'norm')
            x = z

            x = x.permute(0, 2, 1)
            x = self.circular_convolution(x, self.w.to(x.device))  # B, N, D

            x = self.fc(x)
            x = x.permute(0, 2, 1)

            z = x
            z = revin_layer(z, 'denorm')
            x = z

            x = x.permute(1, 0, 2) # B, L, D -> L, B, D
            pred_list.append(x)

        return pred_list

class TexFilter(nn.Module):

    def __init__(self, configs: NeuralPredictionConfig, input_size):
        super().__init__()
        input_size = list(itertools.chain(*input_size))
        self.seq_len = configs.seq_length - configs.pred_length
        self.pred_len = configs.pred_length
        self.embed_size = configs.filter_embed_size
        self.hidden_size = configs.hidden_size
        self.dropout = configs.dropout
        self.band_width = self.seq_len
        self.scale = 0.02
        self.sparsity_threshold = 0.01

        self.revin_layer = nn.ModuleList([RevIN(size, affine=True, subtract_last=False) for size in input_size])
        self.embedding = nn.Linear(self.seq_len, self.embed_size)
        self.token = nn.Conv1d(in_channels=self.seq_len, out_channels=self.embed_size, kernel_size=(1,))

        self.w = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))

        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size)
        )

        self.output = nn.Linear(self.embed_size, self.pred_len)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.layernorm1 = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(self.dropout)

    def tokenEmbed(self, x):
        x = self.token(x)
        return x

    def texfilter(self, x):
        B, N, _ = x.shape
        o1_real = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device=x.device)

        o2_real = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device=x.device)
        o2_imag = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bid,d->bid', x.real, self.w[0]) - \
            torch.einsum('bid,d->bid', x.imag, self.w[1]) + \
            self.rb1
        )

        o1_imag = F.relu(
            torch.einsum('bid,d->bid', x.imag, self.w[0]) + \
            torch.einsum('bid,d->bid', x.real, self.w[1]) + \
            self.ib1
        )

        o2_real = (
                torch.einsum('bid,d->bid', o1_real, self.w1[0]) - \
                torch.einsum('bid,d->bid', o1_imag, self.w1[1]) + \
                self.rb2
        )

        o2_imag = (
                torch.einsum('bid,d->bid', o1_imag, self.w1[0]) + \
                torch.einsum('bid,d->bid', o1_real, self.w1[1]) + \
                self.ib2
        )

        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x_list):
        pred_list = []
        for revin_layer, x in zip(self.revin_layer, x_list):
            x = x.permute(1, 0, 2)
            # x: [Batch, Input length, Channel]
            if x.shape[0] == 0:
                pred_list.append(torch.zeros(0, x.shape[1], self.pred_len, device=x.device))
                continue

            B, L, N = x.shape
            z = x
            z = revin_layer(z, 'norm')
            x = z

            x = x.permute(0, 2, 1)
            x = self.embedding(x)  # B, N, D
            x = self.layernorm(x)
            x = torch.fft.rfft(x, dim=1, norm='ortho')

            weight = self.texfilter(x)
            x = x * weight
            x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
            x = self.layernorm1(x)
            x = self.dropout(x)
            x = self.fc(x)
            x = self.output(x)
            x = x.permute(0, 2, 1)

            z = x
            z = revin_layer(z, 'denorm')
            x = z
            x = x.permute(1, 0, 2)
            pred_list.append(x)

        return pred_list

class Latent_to_Obs(nn.Module):

    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__()
        self.latent_to_ob_mode = config.latent_to_ob
        if config.latent_to_ob == 'linear':
            self.projs = nn.ModuleList([nn.Linear(config.hidden_size, size) for size in input_size])
        elif config.latent_to_ob == 'identity':
            self.projs = [nn.Identity() for _ in input_size]
        else:
            raise ValueError(f"Unknown latent_to_ob mode {config.latent_to_ob}")
        self.out_sizes = input_size

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

    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__()

        input_size = list(itertools.chain(*input_size))
        self.proj = Latent_to_Obs(config, input_size)
        assert config.num_layers == 1, "Only support single layer for now"
        self.rnn = get_rnn_from_config(config, rnn_in_size=None)
        self.out_sizes = input_size
        self.teacher_forcing = config.teacher_forcing
        self.train_tf_interval = config.tf_interval
        self.target_mode = config.target_mode
        self.normalizer = MuStdWrapper(config, input_size)
        self.pred_step = config.pred_length

        assert self.teacher_forcing, "Only support teacher forcing for now"

    def forward(self, input):
        """
        :param x: list of tensors, each of shape (L, B, D)
        """
        bsz = [xx.size(1) for xx in input]
        input = self.normalizer.normalize(input)
        z = self.proj.obs_to_latent(input)
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
        for step in range(self.pred_step):
            ret = self.rnn(hidden_in=last_z)
            last_z = ret[0] if isinstance(ret, tuple) else ret
            z_list.append(last_z)

        z = torch.stack(z_list[1: ])
        z = torch.split(z, bsz, dim=1)

        x = self.proj.latent_to_obs(z)
        x = self.normalizer.unnormalize(x)
        return x
    
    def set_mode(self, mode):
        pass

class Decoder(nn.Module):
    """
    Tokenizer + Decoder
    TODO: delete embed length (multi query mode for POYO), if it doesn't work
          delete TOTEM, if it doesn't make a difference
    """

    def __init__(self, config: NeuralPredictionConfig, input_size, unit_types):
        super().__init__()

        self.Tin = config.decoder_context_length if config.decoder_context_length is not None else config.seq_length - config.pred_length
        self.dataset_idx = [] # the dataset index for each session
        for i_dataset, size in enumerate(input_size):
            self.dataset_idx += [i_dataset] * len(size)
        self.num_datasets = len(input_size)
        self.num_unit_types = [np.concatenate(unit_type).max() + 1 for unit_type in unit_types]
        input_size = list(itertools.chain(*input_size))
        self.unit_types = list(itertools.chain(*unit_types))
        self.unit_types = [torch.from_numpy(unit_type).to(DEVICE) for unit_type in self.unit_types]

        if config.tokenizer_type == 'vqvae':
            assert config.tokenizer_dir is not None, "Pre-trained tokenizer dir must be provided for vqvae"
            model_path = os.path.join(config.tokenizer_dir, config.tokenizer_state_dict_file)
            self.tokenizer = get_pretrained_model(model_path, config.tokenizer_dir, input_size)
            self.TC = self.Tin // self.tokenizer.compression_factor
            self.tokenizer_type = 'vqvae'
            self.token_dim = self.tokenizer.embedding_dim
        elif config.tokenizer_type == 'cnn':
            self.tokenizer = nn.Sequential(nn.Conv1d(1, config.conv_channels, config.kernel_size, config.conv_stride), nn.ReLU())
            self.TC = (self.Tin - config.kernel_size - 1) // config.conv_stride + 1
            self.tokenizer_type = 'cnn'
            self.token_dim = config.conv_channels
        else:
            self.tokenizer = None
            self.TC = self.Tin // config.compression_factor
            self.tokenizer_type = 'none'
            self.token_dim = config.compression_factor
        self.T_step = self.Tin // self.TC

        self.input_size = input_size
        self.embedding_length = 1
        self.pred_step = config.pred_length

        if config.decoder_type == 'Linear':
            self.decoder = nn.Flatten(1, 2)
            self.embedding_dim = self.TC * self.token_dim
        elif config.decoder_type == 'MLP':
            self.decoder = nn.Sequential(
                nn.Flatten(1, 2), 
                nn.Linear(self.TC * self.token_dim, config.decoder_hidden_size), 
                nn.ReLU()
            )
            self.embedding_dim = config.decoder_hidden_size
        elif config.decoder_type == 'Transformer':
            self.decoder = XcodeYtimeDecoder(
                self.token_dim, 
                config.decoder_hidden_size, 
                config.decoder_num_heads,
                config.decoder_hidden_size, 
                config.decoder_num_layers
            )
            self.embedding_dim = config.decoder_hidden_size * self.TC
        elif config.decoder_type == 'POYO':
            poyo_query_mode = config.poyo_query_mode
            self.embedding_length = 1 if poyo_query_mode == 'single' else config.pred_length
            self.decoder = POYO(
                input_dim=self.token_dim, 
                dim=config.decoder_hidden_size,
                depth=config.decoder_num_layers, 
                self_heads=config.decoder_num_heads,
                input_size=input_size,
                num_latents=config.poyo_num_latents,
                query_length=self.embedding_length,
                T_step=self.T_step,
                unit_dropout=config.poyo_unit_dropout,
                output_latent=config.poyo_output_mode == 'latent',
                t_max=config.rotary_attention_tmax,
                num_datasets=self.num_datasets,
                unit_embedding_components=config.unit_embedding_components,
                latent_session_embedding=config.latent_session_embedding,
                num_unit_types=self.num_unit_types,
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

        self.conditioning = config.conditioning
        self.conditioning_dim = config.conditioning_dim
        if config.conditioning == 'mlp':
            self.in_proj = nn.Sequential(nn.Linear(self.Tin, config.conditioning_dim), nn.ReLU())
            
            self.conditioning_alpha = nn.Linear(self.embedding_dim, config.conditioning_dim)
            self.conditioning_beta = nn.Linear(self.embedding_dim, config.conditioning_dim)
            
            # init as zeros
            self.conditioning_alpha.weight.data.zero_()
            self.conditioning_alpha.bias.data.zero_()
            self.conditioning_beta.weight.data.zero_()
            self.conditioning_beta.bias.data.zero_()

            self.out_proj = nn.Linear(config.conditioning_dim, self.linear_out_size)
            assert not self.separate_projs, "Separate projs not supported for MLP conditioning"
            assert self.linear_out_size == config.pred_length, "Not implemented for now"
            assert self.embedding_length == 1, "Not implemented for now"
        elif config.conditioning == 'none':
            if not self.separate_projs:
                self.proj = nn.Linear(self.embedding_dim, self.linear_out_size)
            else:
                self.proj = nn.ModuleList([BatchedLinear(size, self.embedding_dim, self.linear_out_size, init=config.decoder_proj_init) for size in input_size])
        else:
            raise NotImplementedError(f"Unknown conditioning model {config.conditioning}")

        self.normalizer = MuStdWrapper(config, input_size)

        # freeze parts of the model for finetuning
        if config.freeze_backbone:
            for param in self.decoder.parameters():
                param.requires_grad = False
            if config.decoder_type == 'POYO':
                # allow all embedding layers to be trained
                self.decoder.embedding_requires_grad(True)
            self.normalizer.requires_grad_(False)
            if self.tokenizer_type != 'none':
                self.tokenizer.requires_grad_(False)

        if config.freeze_conditioned_net:
            assert config.conditioning == 'mlp', "Only support freezing conditioned net for MLP conditioning"
            self.in_proj.requires_grad_(False)
            self.conditioning_alpha.requires_grad_(False)
            self.conditioning_beta.requires_grad_(False)
            self.out_proj.requires_grad_(False)
        
    def forward(self, x, unit_indices=None, unit_timestamps=None):
        """
        x: list of tensors, each of shape (L, B, D)
        return: list of tensors, each of shape (pred_length, B, D)
        """

        bsz = [xx.size(1) for xx in x]
        L = x[0].size(0)
        pred_step = self.pred_step
        x_list = self.normalizer.normalize(x, concat_output=False) # [L, B, D]
        x = self.normalizer.normalize(x, concat_output=True) # sum(B * D), L

        # only use the last Tin steps
        if L != self.Tin:
            x_list = [xx[-self.Tin: ] for xx in x_list]
            x = x[:, -self.Tin: ]

        # Tokenize the input sequence
        if self.tokenizer_type == 'vqvae':
            with torch.no_grad():
                out = self.tokenizer.encode(x) # out: sum(B * D), TC, E
        elif self.tokenizer_type == 'cnn':
            out = x.unsqueeze(1)
            out = self.tokenizer(out) # out: sum(B * D), C, TC
            out = out.permute(0, 2, 1) # out: sum(B * D), TC, C
        elif self.tokenizer_type == 'none':
            out = x.reshape(x.shape[0], self.TC, self.Tin // self.TC) # out: sum(B * D), TC, E
        else:
            raise ValueError(f"Unknown tokenizer type {self.tokenizer_type}")
        d_list = self.input_size

        if isinstance(self.decoder, POYO):
            if unit_indices is None:
                sum_channels = 0
                unit_indices = []
                for b, d in zip(bsz, self.input_size):
                    indices = torch.arange(d, device=x.device).unsqueeze(0).repeat(b, 1).reshape(-1) # B * D
                    unit_indices.append(indices + sum_channels)
                    sum_channels += d
                unit_indices = torch.cat(unit_indices, dim=0) # sum(B * D)
            if unit_timestamps is None:
                unit_timestamps = torch.zeros_like(unit_indices).unsqueeze(1) + torch.arange(0, self.Tin, self.T_step, device=x.device) # sum(B * D), TC

            input_seqlen = torch.cat([torch.full((b, ), d, device=x.device) 
                                            for b, d in zip(bsz, self.input_size)], dim=0)
            session_index = torch.cat([torch.full((b, ), i, device=x.device) 
                                            for i, b in enumerate(bsz)], dim=0)
            dataset_index = torch.cat([torch.full((b, ), self.dataset_idx[i], device=x.device)
                                            for i, b in enumerate(bsz)], dim=0)
            unit_types = torch.cat([unit_type.repeat(b, 1).reshape(-1) 
                                            for b, unit_type in zip(bsz, self.unit_types)], dim=0) # sum(B * D)

            embed = self.decoder(
                out,
                unit_indices=unit_indices,
                unit_timestamps=unit_timestamps,
                input_seqlen=input_seqlen,
                session_index=session_index,
                dataset_index=dataset_index,
                unit_type=unit_types
            ) # sum(B * D), embedding_dim; or sum(B * D), pred_length, embedding_dim

            if embed.dim() == 3:
                embed = embed.reshape(embed.shape[0], -1) # sum(B * D), pred_length
                d_list = [1 for _ in self.input_size]

        else:
            embed = self.decoder(out) # sum(B * D), embedding_dim

        # partition embed to a list of tensors, each of shape (B, D, 1, embedding_dim)
        split_size = [b * d * self.embedding_length for b, d in zip(bsz, d_list)]
        embed = torch.split(embed, split_size, dim=0)
        embed = [xx.reshape(b, d, self.embedding_length, self.embedding_dim) for xx, b, d in zip(embed, bsz, d_list)] # (B, D, 1, E)

        preds = []
        for i, (e, d, input) in enumerate(zip(embed, self.input_size, x_list)):
            if e.shape[1] < d:
                assert e.shape[1] == 1
                e = e.repeat(1, d, 1, 1)

            if self.conditioning == 'none':
                if self.separate_projs:
                    proj = self.proj[i]
                else:
                    proj = self.proj
            
            if e.shape[2] > 1:
                b = e.shape[0]
                e = e.permute(0, 2, 1, 3)
                e = e.reshape(-1, e.shape[2], e.shape[3]) # B * pred_length, D, token_dim

                pred: torch.Tensor = proj(e) # B * pred_length, D, 1
                pred = pred.reshape(b, pred_step, d).permute(0, 2, 1) # B, D, pred_length
            else:
                if self.conditioning == 'mlp':
                    alpha = self.conditioning_alpha(e.squeeze(2)) # B, D, cond_dim
                    beta = self.conditioning_beta(e.squeeze(2)) # B, D, cond_dim
                    input = input.permute(1, 2, 0) # B, D, L
                    weights = self.in_proj(input) * alpha + beta # B, D, cond_dim
                    pred = self.out_proj(weights) # B, D, pred_length
                elif self.conditioning == 'none':
                    pred: torch.Tensor = proj(e.squeeze(2)) # B, D, pred_length
                else:
                    raise ValueError(f"Unknown conditioning mode {self.conditioning}")
            assert pred.shape[2] == pred_step 
            
            preds.append(pred.permute(2, 0, 1))

        preds = self.normalizer.unnormalize(preds)
        return preds
    
    def load_pretrained(self, state_dict):
        own_state = self.state_dict()

        # copy the pretrained weights to the model
        for name, param in state_dict.items():
            if name in own_state and param.shape == own_state[name].shape:
                own_state[name].copy_(param)

        if hasattr(self.decoder, 'reset_for_finetuning'):
            self.decoder.reset_for_finetuning()

class NetFormer(nn.Module):
    """
    NetFormer model
    """

    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__()
        
        input_size = list(itertools.chain(*input_size))
        self.pred_step = config.pred_length
        sum_channels = sum(input_size)

        self.input_size = input_size
        self.netformer = BaseNetFormer(
            sum_channels, 
            config.seq_length,
            config.pred_length,
        )

        self.normalizer = MuStdWrapper(config, input_size)

    def forward(self, input):
        """
        :param x: list of tensors, each of shape (L, B, D)
        """
        x_list = self.normalizer.normalize(input, concat_output=False) # [L, B, D]
        pred_list = []

        for i in range(len(x_list)):
            x = x_list[i].permute(1, 2, 0) # (B, D, L)
            neuron_ids = torch.arange(sum(self.input_size[: i]), sum(self.input_size[: i + 1]), device=x.device).unsqueeze(0)
            pred = self.netformer(x, neuron_ids)
            pred = pred.permute(2, 0, 1) # (B, D, L) -> (L, B, D)
            pred_list.append(pred)

        pred_list = self.normalizer.unnormalize(pred_list)
        return pred_list

from zapbench_scripts.standalone_poco import POCO

class POCOtest(POCO):
    """
    Use the standalone version of POCO
    """

    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__(config, input_size)

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs: NeuralPredictionConfig, input_size):
        super().__init__()

        input_size = list(itertools.chain(*input_size))
        self.task_name = "short_term_forecast"
        self.seq_len = configs.seq_length - configs.pred_length
        self.pred_len = configs.pred_length
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_list):
        preds = []
        for x in x_list:
            # x: (L, B, D)
            seq_last = x[-1:, :, :].detach()
            x = x - seq_last
            x = self.Linear(x.permute(1, 2, 0)).permute(2, 0, 1)
            x = x + seq_last
            preds.append(x)
        return preds
    
class ARMLP(nn.Module):
    """
    MLP model for next-step prediction
    """
    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__()
        
        input_size = list(itertools.chain(*input_size))
        # check that all input sizes are the same
        assert len(set(input_size)) == 1, "All input sizes must be the same for MLP"

        self.pred_step = config.pred_length
        self.mlp = nn.Sequential(
            nn.Linear(input_size[0], config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, input_size[0])
        )

    def forward(self, input):
        """
        :param input: list of tensors, each of shape (L, B, D)
        """
        bsz = [x.shape[1] for x in input]
        state = torch.cat(input, dim=1)[-1] # (B, D)
        preds = []

        for step in range(self.pred_step):
            state = self.mlp(state)
            preds.append(state)

        preds = torch.stack(preds, dim=0) # (pred_step, B, D)
        preds = torch.split(preds, bsz, dim=1)
        return preds