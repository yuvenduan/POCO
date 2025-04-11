# https://github.com/NeuroAIHub/NetFormer/blob/main/NetFormer/layers.py

import torch
from torch import nn, einsum
from torch.nn import functional as F

class Attention(nn.Module):
    """
    Attention layer in NetFormer, which takes input tensor x and entity tensor e, and returns the output tensor.
    """

    def __init__(
        self,
        dim_X,
        dim_E,
        *,
        dropout=0.0,
        activation='none', # 'sigmoid' or 'tanh' or 'softmax' or 'none'
    ):
        super().__init__()
        self.activation = activation

        self.scale = (dim_X + dim_E) ** -0.5

        # Q, K

        self.query_linear = nn.Linear(dim_X + dim_E, dim_X + dim_E, bias=False)
        self.key_linear = nn.Linear(dim_X + dim_E, dim_X + dim_E, bias=False)

        # dropouts

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, e, pred_step=1):

        x_e = torch.cat((x, e), dim=-1)

        batch_size, n, t = x.shape

        # We_Q_We_KT: (dim_E, dim_E)

        # We_Q_We_KT = (self.query_linear.weight.clone().detach().T)[t:] @ (self.key_linear.weight.clone().detach().T)[t:].T
        # attn3 = einsum("b n e, b m e -> b n m", e @ We_Q_We_KT, e)

        # Q, K

        queries = self.query_linear(x_e)
        keys = self.key_linear(x_e)

        logits = einsum("b n d, b m d -> b n m", queries, keys)
        if self.activation == 'softmax':
            attn = logits.softmax(dim=-1)
        elif self.activation == 'sigmoid':
            attn = F.sigmoid(logits)
        elif self.activation == 'tanh':
            attn = F.tanh(logits)
        elif self.activation == 'none':
            attn = logits

        attn = self.attn_dropout(attn)
        attn = attn * self.scale

        v = x  # identity mapping
        out = einsum("b n m, b m t -> b n t", attn, v)
        out = out + x   # residual connection

        for step in range(pred_step - 1):
            next_step = einsum("b n m, b m t -> b n t", attn, out[:, :, -1: ]) + out[:, :, -1: ]
            out = torch.cat((out, next_step), dim=-1)

        return out, attn

class BaseNetFormer(nn.Module):
    # adpated from https://github.com/NeuroAIHub/NetFormer/blob/main/NetFormer/models.py

    def __init__(
        self,
        num_unqiue_neurons,
        window_size=200,
        predict_window_size = 1,
        attention_activation="softmax", # "softmax" or "sigmoid" or "tanh", "none"
        dim_E=30,
    ):
        super().__init__()

        # self.cell_type_level_constraint = nn.Parameter(torch.FloatTensor(num_cell_types, num_cell_types).uniform_(-1, 1))
        # self.cell_type_level_var = nn.Parameter(torch.ones(num_cell_types, num_cell_types), requires_grad=True)

        self.predict_window_size = predict_window_size
        dim_X = window_size - predict_window_size

        # Embedding
        self.embedding_table = nn.Embedding(
            num_embeddings=num_unqiue_neurons, embedding_dim=dim_E   # global unique neuron lookup table
        )
        self.layer_norm = nn.LayerNorm(dim_X + dim_E)

        # Attention
        self.attentionlayer = Attention(
            dim_X=dim_X,
            dim_E=dim_E,
            activation=attention_activation,
        )

        self.layer_norm2 = nn.LayerNorm(dim_X + predict_window_size - 1)

    def forward(self, x, neuron_ids): # x: (batch_size, neuron_num, time), neuron_ids: (batch_size, neuron_num)

        e = self.embedding_table(neuron_ids[0])
        e = e.repeat(x.shape[0], 1, 1)  # (m, e) to (b, m, e)

        x_e = self.layer_norm(torch.cat((x, e), dim=-1))
        # Split x and e
        x = x_e[:, :, :x.shape[-1]]
        e = x_e[:, :, x.shape[-1]:]

        x, attn = self.attentionlayer(x, e, self.predict_window_size)
        x = self.layer_norm2(x)

        return x