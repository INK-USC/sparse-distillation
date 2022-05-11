import logging
import math

from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List, Dict

import torch
import torch.distributed as dist
import torch.optim

from fairseq import utils
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.optim.fused_adam import get_fused_adam_class
from fairseq.optim.adam import FairseqAdamConfig, Adam
from fairseq.optim.composite import CompositeOptimizer, FairseqCompositeOptimizer
from omegaconf import II, OmegaConf

logger = logging.getLogger(__name__)

@register_optimizer("mixed_adam", dataclass=FairseqAdamConfig)
class FairseqMixedAdamOptimizer(FairseqCompositeOptimizer):

    # optimizers: Dict[str, FairseqOptimizer] = {}

    def __init__(self, cfg: FairseqAdamConfig, params):
        FairseqOptimizer.__init__(self, cfg)

        groupped_params = defaultdict(list)
        for p in params:
            group = getattr(p, "param_group", "default")
            groupped_params[group].append(p)

        fused_adam_cls = get_fused_adam_class()
        use_fused_adam = (
            not getattr(cfg, "use_old_adam", False)
            and fused_adam_cls is not None
            and torch.cuda.is_available()
        )
        if getattr(cfg, "tpu", False):
            # on TPUs we use the Adam defined here, since it
            # automatically casts gradients to FP32
            self.dense_optimizer = Adam(groupped_params["default"], **self.optimizer_config)
        elif use_fused_adam:
            logger.info("using FusedAdam")
            self.dense_optimizer = fused_adam_cls(groupped_params["default"], **self.optimizer_config)
        else:
            self.dense_optimizer = Adam(groupped_params["default"], **self.optimizer_config)

        self.sparse_optimizer = MySparseAdam(groupped_params["sparse"], **self.optimizer_config)

        self.optimizers = {
            "default": self.dense_optimizer,
            "sparse": self.sparse_optimizer,
        }
        self._optimizer = CompositeOptimizer(self.optimizers)
        self.groupped_params = groupped_params

    @property
    def optimizer_config(self):
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.adam_betas)
            if isinstance(self.cfg.adam_betas, str)
            else OmegaConf.to_container(self.cfg.adam_betas),
            "eps": self.cfg.adam_eps,
            "weight_decay": self.cfg.weight_decay,
        }

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm.
        Exclude sparse parameters (cannot compute grad norm?)"""
        return utils.clip_grad_norm_(self.groupped_params["default"], max_norm, aggregate_norm_fn)

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an LR scheduler state dict."""
        for k, state in state_dict.items():
            if k not in self.optimizers:
                # skip extra keys like "loss_scale" added by fp16 optimizer
                continue
            self.optimizers[k].load_state_dict(state)

class MySparseAdam(torch.optim.SparseAdam):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,        
    ):
        super().__init__(params, lr, betas, eps)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return False

    # def load_state_dict(self, state_dict, optimizer_overrides=None):
    #     """Load an optimizer state dict.
    #     In general we should prefer the configuration of the existing optimizer
    #     instance (e.g., learning rate) over that found in the state_dict. This
    #     allows us to resume training from a checkpoint using a new set of
    #     optimizer args.
    #     """
    #     logger.info("using mysparseadam load_state_dict")
    #     super().load_state_dict(state_dict) # use parent class methods

    #     if optimizer_overrides is not None and len(optimizer_overrides) > 0:
    #         # override learning rate, momentum, etc. with latest values
    #         for group in self.param_groups:
    #             group.update(optimizer_overrides)
