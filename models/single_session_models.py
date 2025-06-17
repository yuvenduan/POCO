# A collection of models that only work a single session of data (when the input size is fixed)
# for all models here, the input is a tensor of shape (batch_size, seq_len, num_neurons)
# output is a tensor of shape (batch_size, preds_len, num_neurons)

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.configs import SupervisedLearningBaseConfig, NeuralPredictionConfig

from models.model_utils import get_rnn_from_config
from models.layers import tcn
from models.layers.normalizer import RevIN, MuStdWrapper
from models.layers.autoformer import series_decomp
import models.layers.tsmixer as tsmixer

class Autoregressive(nn.Module):
    """
    An autoregressive model that predicts the next step based on the previous steps
    """

    def __init__(self, config: NeuralPredictionConfig, input_size, omit_linear=False):
        """
        input_size: number of neurons to predict
        """
        super().__init__()

        if omit_linear:
            assert input_size == config.hidden_size, "If omitting linear layer, input size must be equal to hidden_size"
        
        self.in_proj = nn.Linear(input_size, config.hidden_size) if not omit_linear else nn.Identity()
        self.rnn = get_rnn_from_config(config, rnn_in_size=config.hidden_size)
        self.pred_step = config.pred_length

        self.out_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_size) if config.rnn_layernorm else nn.Identity(), 
            nn.Linear(config.hidden_size, input_size) if not omit_linear else nn.Identity()
        )
        
        self.out_sizes = input_size
        self.teacher_forcing = config.teacher_forcing
        self.target_mode = config.target_mode
        
        assert self.teacher_forcing, "Only support teacher forcing for now"

    def _forward(self, input):
        """
        One step forward
        input: tensor of shape (L, B, D)
        return: tensors of shape (L, B, D)
        """
        input = self.in_proj(input)
        ret = self.rnn(input)
        new_x = ret[0] if isinstance(ret, tuple) else ret
        new_x = self.out_proj(new_x)
        return new_x

    def forward(self, input):
        """
        :param x: tensor of shape (L, B, D)
        :return: tensor of shape (L + pred_step - 1, B, D)
        """

        for step in range(self.pred_step):
            pred = self._forward(input)
            # concatenate the prediction to the input, and use it as the input for the next step
            if self.target_mode == 'raw':
                input = torch.cat([input, pred[-1: ]], dim=0)
            elif self.target_mode == 'derivative':
                input = torch.cat([input, pred[-1: ] + input[-1: ]], dim=0)
            else:
                raise ValueError(f"Unknown target mode {self.target_mode}")

        return pred
    
class TCN(nn.Module):

    def __init__(self, configs, input_size):
        super().__init__()
        self.model = tcn.TCN(configs, input_size)

    def forward(self, x):
        # x: L, B, D
        x = x.permute(1, 0, 2)
        x = self.model(x)
        x = x.permute(1, 0, 2)
        return x

class DLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs: NeuralPredictionConfig, input_size):
        """
        individual: Bool, whether shared model among different variates.
        """
        super().__init__()
        self.task_name = "short_term_forecast"
        self.seq_len = configs.seq_length - configs.pred_length
        self.pred_len = configs.pred_length
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(self.seq_len // 4 * 2 + 1)
        self.individual = configs.separate_projs
        self.channels = input_size

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output
    
    def forward(self, x_enc): # [L, B, D]
        x_enc = x_enc.permute(1, 0, 2) # [B, L, D]
        dec_out = self.forecast(x_enc)
        dec_out = dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return dec_out.permute(1, 0, 2) # [L, B, D]
    
class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs: NeuralPredictionConfig, input_size):
        super().__init__()
        self.task_name = "short_term_forecast"
        self.seq_len = configs.seq_length - configs.pred_length
        self.pred_len = configs.pred_length
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = input_size
        self.individual = configs.separate_projs
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(1,0,2)
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x.permute(1,0,2) # [Batch, Output length, Channel]
    
class PaiFilter(nn.Module):

    """
    See https://github.com/aikunyi/FilterNet
    """

    def __init__(self, configs: NeuralPredictionConfig, input_size):
        super().__init__()
        self.seq_len = configs.seq_length - configs.pred_length
        self.pred_len = configs.pred_length
        self.scale = 0.02
        self.revin_layer = RevIN(input_size, affine=True, subtract_last=False)

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

    def forward(self, x):
        x = x.permute(1, 0, 2)
        z = x
        z = self.revin_layer(z, 'norm')
        x = z

        x = x.permute(0, 2, 1)
        x = self.circular_convolution(x, self.w.to(x.device))  # B, N, D

        x = self.fc(x)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z
        x = x.permute(1, 0, 2)
        return x

class TSMixer(tsmixer.TSMixer):

    def __init__(self, configs: NeuralPredictionConfig, input_size):
        super().__init__(
            sequence_length=configs.seq_length - configs.pred_length,
            prediction_length=configs.pred_length,
            input_channels=input_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2) # [L, B, D] -> [B, L, D]
        x = super().forward(x)
        x = x.permute(1, 0, 2)
        return x