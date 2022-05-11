from fairseq.criterions.sentence_prediction import SentencePredictionCriterion
from fairseq.criterions import register_criterion

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@register_criterion("distillation_multitask")
class DistillationMultitaskCriterion(SentencePredictionCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task, classification_head_name, regression_target)
        self.temperature = 1.0

    def forward(self, model, sample, mode="supervised", reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=distillation_multitask"

        if mode == "supervised":
            logits, _ = model(
                **sample["net_input"],
                features_only=True,
                classification_head_name=self.classification_head_name,
            )
            targets = model.get_targets(sample, [logits]).view(-1)
            sample_size = targets.numel()

            lprobs = F.log_softmax(logits, dim=-1)
            supervised_loss = F.nll_loss(lprobs, targets, reduction="sum")

            logging_output = {
                "supervised_loss": supervised_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample_size,
                "sample_size": sample_size,
            }

            if not self.regression_target:
                preds = logits.argmax(dim=1)
                logging_output["ncorrect"] = (preds == targets).sum()

            loss = supervised_loss

        elif mode == "distillation":
            logits, _ = model(
                **sample["net_input"],
                features_only=True,
                classification_head_name=self.classification_head_name,
            )
            # according to https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # teacher output gets F.softmax; student output gets F.log_softmax
            lprobs = F.log_softmax(logits, dim=-1)
            # distill_lprobs = F.log_softmax(logits / self.temperature, dim=-1)
            distill_target = F.softmax(sample["distill_target"] / self.temperature, dim=-1)
            distill_loss = F.kl_div(lprobs, distill_target, reduction="sum") * self.temperature * self.temperature

            sample_size = logits.size()[-1]

            logging_output = {
                "distill_loss": distill_loss.data,
            }

            loss = distill_loss

        else:
            raise NotImplementedError

        return loss, sample_size, logging_output