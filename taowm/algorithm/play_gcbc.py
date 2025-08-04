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

    pass
