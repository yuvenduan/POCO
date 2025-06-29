import torch
import torch.nn as nn
import math
from typing import Optional
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CausalTransformerEncoder(nn.TransformerEncoder):

    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positional_embedding = PositionalEncoding(hidden_size, max_len=100)

    def forward(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src (Tensor): current_len_output x bsz x hidden_dim
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
        """

        # output = self.positional_embedding(src)
        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_key_padding_mask=src_key_padding_mask,
            )

        return output
    
class CausalTransformerEncoderLayer(nn.TransformerEncoderLayer):

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            see CausalTransformerEncoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim
        """

        return super().forward(
            src,
            src_mask=generate_square_subsequent_mask(src.size(0), src.device),
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True
        )

def generate_square_subsequent_mask(sz: int, device: str = "cpu") -> torch.Tensor:
    """ Generate the attention mask for causal decoding """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask

# Adapted from https://github.com/SaberaTalukder/TOTEM/blob/master/forecasting/lib/models/decode.py
class XcodeYtimeDecoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        seq_in_len: int = 5000,
        seq_out_len: int = 5000,
        dropout: float = 0.0,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super(XcodeYtimeDecoder, self).__init__()
        self.model_type = "Transformer"
        self.d_model = d_model

        self.has_linear_in = d_in != d_model
        if self.has_linear_in:
            self.linear_in = nn.Linear(d_in, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_in_len)

        encoder_layers = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            x: tensor of shape (batch_size, seq_in_len, d_in)
            x_mask: tensor of shape (seq_in_len, seq_in_len)

        Returns:
            y: tensor of shape (batch_size, seq_out_len)
        """
        x = x.transpose(0, 1) # (seq_in_len, batch, d_in)
        if self.has_linear_in:
            x = self.linear_in(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, x_mask)  # (seq_in_len, batch, d_model)

        x = torch.permute(x, (1, 0, 2))
        x = x.flatten(start_dim=1)

        return x