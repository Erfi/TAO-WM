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
        observation_modalities: List[str] = [],
        goal_modalities: List[str] = [],
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
        self.lr = lr
        self.actor_cfg = actor
        self.goal_encoder_cfg = goal_encoder
        self.perceptual_encoder_cfg = perceptual_encoder
        self.observation_modalities = observation_modalities
        self.goal_modalities = goal_modalities

        self.build_networks()

    def build_networks(self):
        pass

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        breakpoint()
        return {}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        breakpoint()
        return {}

    def configure_optimizers(self) -> List[Optimizer]:
        """Configure optimizers for the model."""
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]
