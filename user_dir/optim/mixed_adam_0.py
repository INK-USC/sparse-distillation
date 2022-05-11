import logging
from fairseq.optim import register_optimizer
from fairseq.optim.composite import (
    OptimizerAndSchedulerConfig,
    CompositeOptimizerConfig,
    FairseqCompositeOptimizer,
)
from fairseq.optim.adam import FairseqAdamConfig
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import PolynomialDecayLRScheduleConfig
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

adam_cfg = OmegaConf.structured(FairseqAdamConfig(adam_betas=(0.9, 0.98), adam_eps=1e-6, weight_decay=0.1, lr=[1e-3]))
scheduler_cfg = OmegaConf.structured(PolynomialDecayLRScheduleConfig(total_num_update=10000))
default_group_cfg = OmegaConf.structured(OptimizerAndSchedulerConfig(optimizer=adam_cfg, lr=[1e-3], lr_scheduler=scheduler_cfg))
sparse_adam_cfg = OmegaConf.structured(FairseqAdamConfig(adam_betas=(0.9, 0.98), adam_eps=1e-6, weight_decay=0.1, lr=[1e-3]))
embed_group_cfg = OmegaConf.structured(OptimizerAndSchedulerConfig(optimizer=sparse_adam_cfg, lr=[1e-3], lr_scheduler=scheduler_cfg))
hardcoded_cfg = OmegaConf.structured(CompositeOptimizerConfig(groups={"default": default_group_cfg, "embed": embed_group_cfg}))

@register_optimizer("mixed_adam_0", dataclass=CompositeOptimizerConfig)
class FairseqMixedAdamOptimizer0(FairseqCompositeOptimizer):
    def __init__(self, cfg: FairseqCompositeOptimizer, params):
        super().__init__(hardcoded_cfg, params)