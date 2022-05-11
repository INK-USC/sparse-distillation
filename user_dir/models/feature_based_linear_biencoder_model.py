import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture,
)

from .hub_interface import FeatureBasedLinearBiencoderModelHubInterface
from .feature_based_linear_model import FeatureBasedLinearModel, FeatureBasedLinearModelClassificationHead
from ..modules.multi_layer_classification_head import MultipleLayerClassificationHead

logger = logging.getLogger(__name__)

@register_model("feature_based_linear_biencoder_model")
class FeatureBasedLinearBiencoderModel(FeatureBasedLinearModel):
    
    def forward(
        self,
        input0_tokens,
        input0_lengths,
        input1_tokens,
        input1_lengths,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True
        
        # print(input0_tokens)
        # print(input0_lengths)
        # print(input1_tokens)
        # print(input1_lengths)

        x0 = self.encoder(input0_tokens, input0_lengths, features_only, return_all_hiddens, **kwargs)
        x1 = self.encoder(input1_tokens, input1_lengths, features_only, return_all_hiddens, **kwargs)

        # print(x0.shape)
        # print(x1.shape)

        # a standard concatenate–compare operation (Wang et al., 2018) 
        product = torch.mul(x0, x1)
        diff = torch.abs(x0 - x1)
        x = torch.cat((x0, x1, product, diff), 1)

        # print(x.shape)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)

        # print(x.shape)

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
                input_dim=self.args.encoder_embed_dim * 4, # concate-compare module
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
                input_dim=self.args.encoder_embed_dim * 4, # concate-compare module
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
        return FeatureBasedLinearBiencoderModelHubInterface(x["args"], x["task"], x["models"][0])

@register_model_architecture("feature_based_linear_biencoder_model", "feature_based_linear_biencoder_model")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 50)
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
