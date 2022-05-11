from fairseq.criterions.sentence_prediction import SentencePredictionCriterion
from fairseq.criterions import register_criterion

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@register_criterion("distillation")
class DistillationCriterion(SentencePredictionCriterion):

    def __init__(self, task, classification_head_name, distillation_temperature, distillation_alpha, focal_pow, regression_target):
        super().__init__(task, classification_head_name, regression_target)
        self.temperature = distillation_temperature
        self.alpha = distillation_alpha
        self.focal_pow = focal_pow

    @staticmethod
    def add_args(parser):
        SentencePredictionCriterion.add_args(parser)
        parser.add_argument(
            "--distillation-temperature", default=1.0, type=float,
            help="temperature used in distillation loss"
        )
        parser.add_argument(
            "--distillation-alpha", default=0.2, type=float,
            help="coeff to balance gold label and distilation loss"
        )        
        parser.add_argument(
            "--focal-pow", default=0.5, type=float,
            help="power used in `focal` loss"
        )        

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # assert (
        #     hasattr(model, "classification_heads")
        #     and self.classification_head_name in model.classification_heads
        # ), "model must provide sentence classification head for --criterion=distillation"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        lprobs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(lprobs, targets, reduction="none")

        # according to https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
        # teacher output gets F.softmax; student output gets F.log_softmax
        distill_lprobs = F.log_softmax(logits / self.temperature , dim=-1)
        distill_target = F.softmax(sample["distill_target"] / self.temperature, dim=-1)
        distill_loss = F.kl_div(lprobs, distill_target, reduction="none") * self.temperature * self.temperature

        # focal loss: https://arxiv.org/pdf/1708.02002.pdf
        weights = torch.pow((1 - sample["freq"]), self.focal_pow)

        # print(sample["freq"])
        # print(weights)
        # print(distill_loss)

        # loss = (1.0 - self.alpha) * torch.dot(weights, loss) + self.alpha * torch.dot(weights, distill_loss)
        # print(distill_loss.shape)
        # print(weights.shape)
        loss = self.alpha * torch.sum(distill_loss.t() * weights)


        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_output