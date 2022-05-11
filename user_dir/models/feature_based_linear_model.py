import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


from fairseq.models.roberta import RobertaModel, RobertaEncoder, RobertaClassificationHead
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.model_parallel.megatron.mpu.layers import VocabParallelEmbedding, ParallelEmbedding
from fairseq.model_parallel.megatron.mpu.layers import ColumnParallelLinear
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .hub_interface import FeatureBasedLinearModelHubInterface
from ..modules.multi_layer_classification_head import MultipleLayerClassificationHead
from ..modules.parallel_layers import CustomParallelEmbedding
from ..modules.attentive_pooling import AttentivePooling

logger = logging.getLogger(__name__)

@register_model("feature_based_linear_model")
class FeatureBasedLinearModel(RobertaModel):
    def __init__(self, args, encoder):
        # use grandparent class init
        FairseqEncoderModel.__init__(self, encoder) 

        self.args = args
        self.classification_heads = nn.ModuleDict()
    
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--hidden-dim",
            type=int,
            metavar="H2",
            help="output of linear layer, dimension",
        )
        parser.add_argument(
            "--attn-dim",
            type=int,
            metavar="H2",
            help="output of linear layer, dimension",
        )
        parser.add_argument(
            "--feature-dropout", type=float, metavar="D1", help="dropout applied to input features"
        )
        parser.add_argument(
            "--pooler-dropout", type=float, metavar="D2", help="dropout applied before classfier"
        )
        parser.add_argument(
            "--embed-bag-mode", type=str, help="pooling method for embedding bag layer"
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )

        parser.add_argument(
            "--hidden-dims",
            type=str,
            metavar="H2",
            help="a list of hidden dimensions",
        )
        parser.add_argument(
            "--multi-layer-classification-head",
            action="store_true",
            default=False,
            help="use MLP classification head",
        )
        parser.add_argument(
            "--vocab-parallel-embedding",
            action="store_true",
            default=False,
            help="use vocab parallel embedding",
        )
        parser.add_argument(
            "--parallel-embedding",
            action="store_true",
            default=False,
            help="use parallel embedding",
        )
        parser.add_argument(
            "--parallel-dense",
            action="store_true",
            default=False,
            help="use parallel dense layer",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = FeatureBasedLinearEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(
        self,
        src_tokens,
        src_lengths,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        x = self.encoder(src_tokens, src_lengths, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        
        # sentence_prediction task asks for 2 outputs
        return x, None

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        if self.args.multi_layer_classification_head:
            self.classification_heads[name] = MultipleLayerClassificationHead(
                input_dim=self.args.encoder_embed_dim,
                inner_dims=self.args.hidden_dims,
                num_classes=num_classes,
                activation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
                q_noise=self.args.quant_noise_pq,
                qn_block_size=self.args.quant_noise_pq_block_size,
                do_spectral_norm=self.args.spectral_norm_classification_head,
            )
        else:
            self.classification_heads[name] = FeatureBasedLinearModelClassificationHead(
                args=self.args,
                input_dim=self.args.encoder_embed_dim,
                inner_dim=inner_dim or self.args.hidden_dim,
                num_classes=num_classes,
                activation_fn=self.args.pooler_activation_fn,
                pooler_dropout=self.args.pooler_dropout,
                q_noise=self.args.quant_noise_pq,
                qn_block_size=self.args.quant_noise_pq_block_size,
                do_spectral_norm=self.args.spectral_norm_classification_head,
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )

        logger.info(x["args"])
        return FeatureBasedLinearModelHubInterface(x["args"], x["task"], x["models"][0])


    def upgrade_state_dict_named(self, state_dict, name):
        # super().upgrade_state_dict_named(state_dict, name)
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        current_dict_size = len(self.encoder.dictionary)
        if loaded_dict_size != current_dict_size:
            state_dict["encoder.embed_tokens.weight"] = state_dict["encoder.embed_tokens.weight"][:current_dict_size, :]

class FeatureBasedLinearModelClassificationHead(nn.Module):
    def __init__(self,
        args,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.args = args

        if getattr(args, "parallel_dense", False):
            self.dense = ColumnParallelLinear(input_dim, inner_dim)
        else:
            self.dense = nn.Linear(input_dim, inner_dim)

        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)    

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class FeatureBasedLinearEncoder(RobertaEncoder):

    def __init__(self, args, dictionary):
        FairseqEncoder.__init__(self, dictionary)

        base_architecture(args)
        self.args = args

        # self.embed_bag = nn.EmbeddingBag(len(dictionary), args.encoder_embed_dim, mode=args.embed_bag_mode)
        
        use_sparse = ((args.optimizer == "mixed_adam") or (args.optimizer == "sgd")) and (not getattr(args, "vocab_parallel_embedding", False)) and (not getattr(args, "parallel_embedding", False))
        logger.info("use sparse? {}".format(use_sparse))
        self.embed_tokens = self.build_embedding(
            len(dictionary), args.encoder_embed_dim, dictionary.pad(), use_sparse
        )

        for p in self.embed_tokens.parameters():
            p.param_group = "sparse"

        if self.args.embed_bag_mode == "attn":
            self.attn = AttentivePooling(args)

    def build_embedding(self, vocab_size, embedding_dim, padding_idx, sparse):
        if getattr(self.args, "parallel_embedding", False):
            return CustomParallelEmbedding(self.args, vocab_size, embedding_dim)
        elif getattr(self.args, "vocab_parallel_embedding", False):
            return VocabParallelEmbedding(vocab_size, embedding_dim, padding_idx)
        else:
            return nn.Embedding(vocab_size, embedding_dim, padding_idx, sparse=sparse)

    def forward(
        self,
        src_tokens,
        src_lengths,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        # a hack to avoid NaN in all padding sequences
        src_lengths = torch.max(src_lengths, torch.tensor([1]).to(src_lengths.device))

        if getattr(self.args, "parallel_embedding", False):
            x = self.embed_tokens(src_tokens, src_lengths)
            return x

        x = self.embed_tokens(src_tokens)

        if self.args.embed_bag_mode == "max":
            x = x.max(dim=1).values
        elif self.args.embed_bag_mode == "mean":
            x = x.sum(dim=1)
            # print(torch.isnan(x).any())
            
            coeff = (1.0 / src_lengths).to(x.dtype) # avoid some fp16 issue
            # print(coeff)

            # this will divide x by the number of features (indicated by src_lengths)
            # reference: https://stackoverflow.com/questions/53987906/how-to-multiply-a-tensor-row-wise-by-a-vector-in-pytorch
            x = x * coeff[:, None]
        elif self.args.embed_bag_mode == "sum":
            x = x.sum(dim=1)
        elif self.args.embed_bag_mode == "attn":
            x_avg = x.sum(dim=1)
            coeff = (1.0 / src_lengths).to(x.dtype) # avoid some fp16 issue
            x_avg = x_avg * coeff[:, None]
            x_mask = torch.eq(src_tokens, self.args.pad)
            x = self.attn(x, x_mask, x_avg)

        return x


@register_model_architecture("feature_based_linear_model", "feature_based_linear_model")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 50)
    args.attn_dim = getattr(args, "attn_dim", 50)
    args.feature_dropout = getattr(args, "feature_dropout", 0.2)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.1)
    args.embed_bag_mode = getattr(args, "embed_bag_mode", "mean")

    # if hidden_dim is specified, use the specified value
    # elif encoder_embed_dim is specified, use that value or 256, whichever is smaller
    backup_hidden_dim = getattr(args, "encoder_embed_dim")
    args.hidden_dim = getattr(args, "hidden_dim", backup_hidden_dim)

    args.multi_layer_classification_head = getattr(args, "multi_layer_classification_head", False)

    # keep some roberta args
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
    args.spectral_norm_classification_head = getattr(args, "spectral_norm_classification_head", False)
    args.max_source_positions = getattr(args, "max_positions", 512)
