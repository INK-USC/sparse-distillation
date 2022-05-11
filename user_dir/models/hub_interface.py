import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders

from fairseq.models.roberta.hub_interface import RobertaHubInterface

class FeatureBasedLinearModelHubInterface(RobertaHubInterface):
    def extract_features(
        self, tokens: torch.LongTensor, src_lengths: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        # if tokens.size(-1) > self.model.max_positions():
        #     raise ValueError(
        #         "tokens exceeds maximum length: {} > {}".format(
        #             tokens.size(-1), self.model.max_positions()
        #         )
        #     )
        features, extra = self.model(
            tokens.to(device=self.device),
            src_lengths.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def predict(self, head: str, tokens: torch.LongTensor, src_lengths: torch.LongTensor, return_logits: bool = False):
        features = self.extract_features(tokens.to(device=self.device), src_lengths.to(device=self.device))
        logits = self.model.classification_heads[head](features)
        if return_logits:
            return logits
        return F.softmax(logits, dim=-1)

class FeatureBasedLinearBiencoderModelHubInterface(RobertaHubInterface):
    def extract_features(
        self, tokens0: torch.LongTensor, src_lengths0: torch.LongTensor, tokens1: torch.LongTensor, src_lengths1: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens0.dim() == 1:
            tokens0 = tokens0.unsqueeze(0)
            tokens1 = tokens1.unsqueeze(0)

        features, extra = self.model(
            tokens0.to(device=self.device),
            src_lengths0.to(device=self.device),
            tokens1.to(device=self.device),
            src_lengths1.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def predict(self, head: str, tokens0: torch.LongTensor, src_lengths0: torch.LongTensor, tokens1: torch.LongTensor, src_lengths1: torch.LongTensor, return_logits: bool = False):
        features = self.extract_features(tokens0.to(device=self.device), src_lengths0.to(device=self.device), tokens1.to(device=self.device), src_lengths1.to(device=self.device))
        logits = self.model.classification_heads[head](features)
        if return_logits:
            return logits
        return F.softmax(logits, dim=-1)