from typing import Dict, List, Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Independent, Normal
from torch.optim.optimizer import Optimizer

from taowm.utils.distributions import TanhNormal
from taowm.utils.episode_utils import get_state_info_on_idx, get_task_info_of_sequence
from taowm.utils.gymnasium_utils import make_env


class PlayGCBC(pl.LightningModule):
    """
    Implementation of Play-supervised Goal Conditioned Behavior Cloning (GCBC) algorithm from
    https://arxiv.org/abs/1903.01973
    """

    def __init__(
        self,
        env: DictConfig = {},
        actor: DictConfig = {},
        perceptual_encoder: DictConfig = {},
        goal_encoder: DictConfig = {},
        transform_manager: DictConfig = {},
        dataloader: DictConfig = {},
        lr: float = 1e-4,
        real_world: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.real_world = real_world
        if not real_world:
            self.env_cfg = env
            self.env = make_env(self.env_cfg)
