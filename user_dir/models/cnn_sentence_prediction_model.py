import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import logging

from typing import Any, Dict, List, Optional, Tuple

from fairseq import utils, checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import LSTMEncoder
from fairseq.models.roberta.model import RobertaModel
from fairseq.models.fairseq_model import BaseFairseqModel

from .feature_based_linear_model import FeatureBasedLinearModelClassificationHead
from .hub_interface import FeatureBasedLinearModelHubInterface

logger = logging.getLogger(__name__)

def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

@register_model('cnn_sentence_prediction')
class CNNSentencePredictionModel(BaseFairseqModel):
    def __init__(self, dictionary, embed, args):
        super().__init__()

        self.args = args

        self.classification_heads = nn.ModuleDict()

        if '<mask>' not in dictionary:
            dictionary.add_symbol('<mask>')
        self.dictionary = dictionary

        if getattr(args, 'embed_file', None) is not None and os.path.exists(args.embed_file):
            self.embed_tokens = self.load_embed(args.embed_file)
        else:
            self.embed_tokens = nn.Embedding(len(dictionary), args.encoder_embed_dim)

        window = args.context_window
        assert window % 2 == 1
        padding = int((window - 1) / 2)

        self.conv = nn.Conv1d(self.args.encoder_embed_dim, self.args.cnn_hidden_dim, window, stride=1, padding=padding)


    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--cnn-hidden-dim",
            type=int,
            metavar="H1",
            help="cnn hidden dimension",
        )
        parser.add_argument(
            "--embed-file",
            type=str,
            help="embedding file",
            default=None,
        )

    @classmethod
    def build_model(cls, args, task):
        dictionary = task.dictionary
        embed = None
        cnn_sentence_prediction(args)
        return cls(dictionary, embed, args)

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

    def load_embed(self, filename):
        roberta = RobertaModel.from_pretrained(filename)
        embed = roberta.model.encoder.sentence_encoder.embed_tokens
        # self.dictionary = roberta.model.encoder.dictionary
        logger.info('Loading embedding from {}, size: {}'.format(filename, embed))
        logger.info('Embed params: {}'.format(embed.weight))
        del roberta
        return embed

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

        # print(src_tokens.shape)
        x = self.embed_tokens(src_tokens)
        # print(x.shape)
        x = x.transpose(1, 2)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = x.transpose(1, 2)
        # print(x.shape)

        # mean pooling
        x = x.sum(dim=1)
        # print(x.shape)

        coeff = (1.0 / src_lengths).to(x.dtype) # avoid some fp16 issue

        # this will divide x by the number of features (indicated by src_lengths)
        # reference: https://stackoverflow.com/questions/53987906/how-to-multiply-a-tensor-row-wise-by-a-vector-in-pytorch
        x = x * coeff[:, None]

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

        self.classification_heads[name] = FeatureBasedLinearModelClassificationHead(
            args=self.args,
            input_dim=self.args.cnn_hidden_dim,
            inner_dim=inner_dim or self.args.hidden_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

@register_model_architecture('cnn_sentence_prediction', 'cnn_sentence_prediction')
def cnn_sentence_prediction(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.cnn_hidden_dim = getattr(args, 'cnn_hidden_dim', 256)
    # args.lstm_num_layers = getattr(args, 'lstm_num_layers', 2)
    # args.lstm_dropout_in = getattr(args, 'lstm_dropout_in', 0.1)
    # args.lstm_dropout_out = getattr(args, 'lstm_dropout_out', 0.1)
    # args.lstm_bidirectional = getattr(args, 'lstm_bidirectional', True)
    # args.lstm_left_pad = getattr(args, 'lstm_left_pad', True)
    # args.lstm_bias = not getattr(args, 'lstm_no_bias', False)

    # backup_hidden_dim = getattr(args, "encoder_embed_dim")
    args.hidden_dim = getattr(args, "hidden_dim", 256)

    args.pooler_dropout = getattr(args, "pooler_dropout", 0.1)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")

    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
    args.spectral_norm_classification_head = getattr(args, "spectral_norm_classification_head", False)
    args.max_source_positions = getattr(args, "max_positions", 512)
    args.max_positions = getattr(args, "max_positions", 512)
    args.feature_dropout = getattr(args, "feature_dropout", 0.0)
    args.context_window = getattr(args, "context_window", 7)