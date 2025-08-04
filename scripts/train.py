import logging
import os
from pathlib import Path
import sys

import hydra
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer

from lumos.utils.info_utils import (
    print_system_env_info,
    setup_tensor_cores,
    get_last_checkpoint,
)


# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def train(cfg: DictConfig) -> None:
    # Set model directories and logistics
    if cfg.exp_dir is None:
        cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    model_dir = Path(cfg.exp_dir) / "model_weights/"
    cfg.callbacks.checkpoint.dirpath = model_dir
    os.makedirs(model_dir, exist_ok=True)
    log_rank_0(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())

    seed_everything(cfg.seed, workers=True)

    if setup_tensor_cores(enabled=cfg.tc_enabled, precision=cfg.tc_precision):
        log_rank_0("Tensor Cores detected and configured for optimal performance")
        log_rank_0(f"Using matmul precision: {cfg.tc_precision}")

    # Load the datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Load the checkpoint if it exists, otherwise initialize a new model
    breakpoint()
    chk = get_last_checkpoint(model_dir)
    if chk is not None:
        # TODO: Get the correct model class (based on configuration) and load it
        from lumos.world_models.dreamer_v2 import DreamerV2  # Example

        model = DreamerV2.load_from_checkpoint(chk.as_posix())
    else:
        model = hydra.utils.instantiate(cfg.tao_model)

    # Initialize the trainer

    # Start the training process


@rank_zero_only
def log_rank_0(*args, **kwargs):
    """
    Log the information using the logger at rank 0.
    """
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    train()
