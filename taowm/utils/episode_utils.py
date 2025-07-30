from pathlib import Path
from typing import Dict

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf


def get_state_info_dict(
    episode: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    :param episode: episode loaded by dataset loader
    :return: info dict of full robot and scene state (for env resets)
    """
    return {
        "state_info": {
            "robot_obs": torch.from_numpy(episode["robot_obs"]),
            "scene_obs": torch.from_numpy(episode["scene_obs"]),
        }
    }


def load_dataset_statistics(train_dataset_dir, transforms):
    """
    Tries to load statistics.yaml in train dataset folder in order to
    update the transforms hardcoded in the hydra config file.
    If no statistics.yaml exists, nothing is changed

    Args:
        train_dataset_dir: path of the training folder
        transforms: transforms loaded from hydra conf
    Returns:
        transforms: potentially updated transforms
    """
    statistics_path = Path(train_dataset_dir) / "statistics.yaml"
    if not statistics_path.is_file():
        return transforms

    statistics = OmegaConf.load(statistics_path)
    for transf_key in ["train", "validation"]:
        for modality in transforms[transf_key]:
            if modality in statistics:
                conf_transforms = transforms[transf_key][modality]
                dataset_transforms = statistics[modality]
                for dataset_trans in dataset_transforms:
                    # Use transforms from tacorl not calvin_agent
                    dataset_trans["_target_"] = dataset_trans["_target_"].replace("calvin_agent", "tacorl")
                    exists = False
                    for i, conf_trans in enumerate(conf_transforms):
                        if dataset_trans["_target_"] == conf_trans["_target_"]:
                            exists = True
                            transforms[transf_key][modality][i] = dataset_trans
                            break
                    if not exists:
                        transforms[transf_key][modality] = ListConfig([*conf_transforms, dataset_trans])
    return transforms
