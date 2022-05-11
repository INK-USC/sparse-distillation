import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

class MultipleLayerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dims,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()

        inner_dims = list(map(int, inner_dims.split(",")))
        assert len(inner_dims) > 0
        
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)

        self.dense_layers = [nn.Linear(input_dim, inner_dims[0])]
        for i in range(1, len(inner_dims)):
            in_dim = inner_dims[i-1]
            out_dim = inner_dims[i]
            # self.dense_layers.append(self.dropout)
            self.dense_layers.append(nn.Linear(in_dim, out_dim))
            # self.dense_layers.append(self.activation_fn)
        self.dense_layers = nn.ModuleList(self.dense_layers)
        
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dims[-1], num_classes), q_noise, qn_block_size
        )
        
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features

        for layer in self.dense_layers:
            x = self.dropout(x)
            x = layer(x)
            x = self.activation_fn(x)

        x = self.dropout(x)
        x = self.out_proj(x)
        return x