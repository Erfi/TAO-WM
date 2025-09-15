from pathlib import Path

import torch
import hydra
from omegaconf import OmegaConf

from taowm.models import MODEL_REGISTRY


def get_model_env_taskchecker_datamodule(train_folder, dataset_path, checkpoint, env=None, device_id=0, show_gui=False):
    """
    Modified version of calvin.calvin_models.calvin_agent.evaluation.utils.get_default_model_and_env
    The modification is because unlike original Calvin Benchmark that uses vision and language goals, we are
    using vision only goals.
    """
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    cfg = OmegaConf.load(train_cfg_path)
    # we don't want to use shm dataset for evaluation so use vision_only.yaml
    with hydra.initialize(config_path="../../config/datamodule/datasets", job_name="evaluate_tao", version_base="1.3"):
        datasets_cfg = hydra.compose("vision_only")
        # since we don't use the trainer during inference, manually set up data_module
        cfg.datamodule.datasets = datasets_cfg
        cfg.datamodule.root_data_dir = dataset_path
        data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
        data_module.prepare_data()
        data_module.setup()
        dataset = data_module.val_datasets["vis"]
        device = torch.device(f"cuda:{device_id}")

    if env is None:
        # rollout callbacks might not be present in train cfg, so we will look into the rollout
        rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "config/callbacks/rollout/default_vision.yaml")
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, dataset, device, show_gui=show_gui)
    # create a task checker
    tasks_cfg = OmegaConf.load(Path(__file__).parents[2] / "config/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    tasks_checker = hydra.utils.instantiate(tasks_cfg)

    # import the model class that was used for the training
    model_cls = MODEL_REGISTRY.get(cfg.model._target_, None)
    assert model_cls is not None, f"Model class {cfg.model._target_} not found in MODEL_REGISTRY"
    model = model_cls.load_from_checkpoint(checkpoint)
    model.freeze()
    if cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
    model = model.cuda(device)
    return model, env, tasks_checker, data_module
