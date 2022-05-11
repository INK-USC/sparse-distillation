import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from fairseq.model_parallel.megatron.mpu.initialize import get_model_parallel_rank
from fairseq.model_parallel.megatron.mpu.initialize import get_model_parallel_world_size
from fairseq.model_parallel.megatron.mpu.mappings import copy_to_model_parallel_region
from fairseq.model_parallel.megatron.mpu.mappings import gather_from_model_parallel_region
from fairseq.model_parallel.megatron.mpu.mappings import reduce_from_model_parallel_region
from fairseq.model_parallel.megatron.mpu.mappings import scatter_to_model_parallel_region
from fairseq.model_parallel.megatron.mpu.random import get_cuda_rng_tracker
from fairseq.model_parallel.megatron.mpu.utils import divide
from fairseq.model_parallel.megatron.mpu.utils import split_tensor_along_last_dim
from fairseq.model_parallel.megatron.mpu.utils import VocabUtility
from fairseq.model_parallel.megatron.mpu.layers import _initialize_affine_weight

class CustomParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.
    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """
    def __init__(self, args, num_embeddings, embedding_dim,
                 init_method=init.xavier_uniform_,
                 keep_master_weight_for_test=False):
        super(CustomParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.args = args
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set some detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # Divide the weight matrix along the embedding dimension.
        world_size = get_model_parallel_world_size()
        self.embedding_dim_per_partition = divide(self.embedding_dim,
                                                  world_size)

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings,
                                             self.embedding_dim_per_partition))
        self.weight.model_parallel = True
        # And initialize.
        _initialize_affine_weight(
            self.weight, self.num_embeddings, self.embedding_dim,
            self.embedding_dim_per_partition, 1, init_method,
            stride=1, return_master_weight=False)

    def forward(self, input_, src_lengths):
        input_parallel = copy_to_model_parallel_region(input_)
        x = F.embedding(input_parallel, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)

        # average pooling
        x = x.sum(dim=1)

        coeff = (1.0 / src_lengths).to(x.dtype) # avoid some fp16 issue

        # this will divide x by the number of features (indicated by src_lengths)
        # reference: https://stackoverflow.com/questions/53987906/how-to-multiply-a-tensor-row-wise-by-a-vector-in-pytorch
        x = x * coeff[:, None]

        output = gather_from_model_parallel_region(x)
        return output