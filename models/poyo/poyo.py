# Adapted from POTO repo
import numpy as np
import torch
import torch.nn as nn

from models.poyo import (
    Embedding,
    InfiniteVocabEmbedding,
    PerceiverRotary,
    compute_loss_or_metric,
)

class POYO(nn.Module):

    def __init__(
        self,
        *,
        input_dim=1,
        dim=512,
        dim_head=64,
        num_latents=64,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        emb_init_scale=0.02,
        use_memory_efficient_attn=True,
        input_size=None,
        query_length=1,
        T_step=1,
        unit_dropout=0.0,
        output_latent=False, # if True, return the latent representation, else return the query representation
        t_max=100,
        num_datasets=1,
        unit_embedding_components=['session', ],
        latent_session_embedding=False,
        num_unit_types=None,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, dim)
        self.unit_emb = Embedding(sum(input_size), dim, init_scale=emb_init_scale)
        self.session_emb = Embedding(len(input_size), dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)
        self.dataset_emb = Embedding(num_datasets, dim, init_scale=emb_init_scale)

        if num_unit_types is not None:
            self.unit_type_emb = nn.ModuleList([
                Embedding(num_unit_type, dim, init_scale=emb_init_scale) 
            for num_unit_type in num_unit_types])

        self.num_latents = num_latents
        self.query_length = query_length
        self.unit_dropout = unit_dropout

        self.perceiver_io = PerceiverRotary(
            dim=dim,
            dim_head=dim_head,
            depth=depth,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
            use_memory_efficient_attn=use_memory_efficient_attn,
            t_max=t_max,
        )

        self.dim = dim
        self.T_step = T_step
        self.using_memory_efficient_attn = self.perceiver_io.using_memory_efficient_attn
        self.output_latent = output_latent
        self.unit_embedding_components = unit_embedding_components
        self.latent_session_embedding = latent_session_embedding

    def forward(
        self,
        # input sequence
        x: torch.Tensor, # sum(B * D), L, input_dim
        unit_indices, # sum(B * D)
        unit_timestamps, # sum(B * D), L
        input_seqlen, # (B, )
        # output sequence
        session_index,  # (B, )
        dataset_index, # (B, )
        unit_type, # sum(B * D)
    ):

        # input
        L = x.shape[1]
        B = input_seqlen.shape[0]
        T = L * self.T_step
        unit_embedding = self.unit_emb(unit_indices)

        session_indices = torch.concatenate([torch.full((input_seqlen[i], ), session_index[i], device=x.device) for i in range(B)]) # sum(B * D)
        dataset_indices = torch.concatenate([torch.full((input_seqlen[i], ), dataset_index[i], device=x.device) for i in range(B)]) # sum(B * D)

        if 'session' in self.unit_embedding_components:
            unit_embedding = unit_embedding + self.session_emb(session_indices) + self.dataset_emb(dataset_indices) # sum(B * D), dim
        if 'unit_type' in self.unit_embedding_components: 
            dataset_id = dataset_index[0]
            assert torch.all(dataset_index == dataset_id), 'currently, embedding for unit_type is only supported when each batch contains only one dataset'
            unit_embedding = unit_embedding + self.unit_type_emb[dataset_id](unit_type)

        inputs = unit_embedding.unsqueeze(1) + self.input_proj(x) # sum(B * D), L, dim

        if self.training and self.unit_dropout > 0.0 and np.random.rand() < 0.8:
            mask = torch.rand(inputs.shape[0], device=x.device) < 1 - self.unit_dropout
            inputs = inputs[mask]
            unit_timestamps = unit_timestamps[mask]

            # compute adjusted input_seqlen
            new_input_seqlen = torch.zeros(B, device=x.device, dtype=torch.long)
            pos = 0
            for i in range(B):
                new_input_seqlen[i] = torch.sum(mask[pos: pos + input_seqlen[i]])
                pos += input_seqlen[i]
            input_seqlen = new_input_seqlen

        unit_timestamps = unit_timestamps.reshape(-1)
        inputs = inputs.reshape(-1, inputs.shape[2])

        # latents
        latent_index = torch.arange(self.num_latents, device=x.device)
        latents = self.latent_emb(latent_index)
        latents = latents.repeat(B, 1, 1) # B, N_latent, dim
        if self.latent_session_embedding:
            latents = latents + self.session_emb(session_index).unsqueeze(1) + self.dataset_emb(dataset_index).unsqueeze(1) # B, N_latent, dim
        latents = latents.reshape(-1, latents.shape[2])
        latent_seqlen = torch.full((B, ), self.num_latents, device=x.device)
        latent_timestamps = torch.arange(0, T, step=T // self.num_latents, device=x.device)
        latent_timestamps = latent_timestamps.repeat(B) # B * N_latent

        # outputs
        output_queries = unit_embedding
        sumD = output_queries.shape[0]
        output_queries = output_queries.repeat_interleave(self.query_length, dim=0) # sum(B * D * q_len), dim
        output_timestamps = torch.arange(self.query_length, device=x.device).repeat(sumD) + T # sum(B * D * q_len)

        # feed into perceiver
        output_latents = self.perceiver_io(
            inputs=inputs,
            latents=latents,
            output_queries=output_queries,
            input_timestamps=unit_timestamps,
            latent_timestamps=latent_timestamps,
            output_query_timestamps=output_timestamps,
            input_mask=None,
            input_seqlen=input_seqlen * L,
            latent_seqlen=latent_seqlen,
            output_query_seqlen=input_seqlen,
            output_latent=self.output_latent,
        )

        return output_latents