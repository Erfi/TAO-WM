import logging
import json
import random
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from calvin_agent.datasets.disk_dataset import DiskDataset

logger = logging.getLogger(__name__)


class GoalAugmentedDiskDataset(DiskDataset):
    """
    This dataset class augments the DiskDataset such that every sampled sequence
    is augmented with a sampled goal. The goals (visual) can be from the same episode,
    or taken from a list of negative samples provided via a file mapping each episode to its hard negatives
    (see: https://arxiv.org/abs/2209.08959 -> Latent Plans for Task-Agnostic Offline Reinforcement Learning)
    This is used in algorithms such as TACORL (Latent Plans for Task-Agnostic Offline Reinforcement Learning).

    Args:
        geom_sampling_prob (float): Hyperparameter for geometric sampling (0, 1), higher -> closer goals.
        geom_sampling_window_stride (bool): If true, goal are a few window sizes away from the end of the sequence, otherwise few steps away.
        hard_negatives_ratio (float): Probability of using hard negatives instead of geometric sampling. 0 -> always geometric.
        hard_negatives_file (str): Path to file containing hard negatives (mapping every episode -> [hard_negatives]).
    """

    def __init__(
        self,
        *args,
        geom_sampling_prob: float,
        geom_sampling_window_stride: bool = True,
        hard_negatives_ratio: float = 0.5,
        hard_negatives_file: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.geom_sampling_prob = geom_sampling_prob
        self.hard_negatives_ratio = hard_negatives_ratio
        self.geom_sampling_window_stride = geom_sampling_window_stride
        self.nn_steps_from_step = self.get_nn_steps_from_step(hard_negatives_file) if hard_negatives_file else {}
        self.ep_start_end_ids = self.get_episode_start_end_ids()
        self.inv_episode_lookup = (
            {step: idx for idx, step in enumerate(self.episode_lookup)} if self.nn_steps_from_step else {}
        )

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = self._get_window_size(idx)
            else:
                logger.error(f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}")
                raise ValueError
        else:
            idx, window_size = idx
        sequence = self._get_sequences(idx, window_size)
        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)

        # Add goal to sequence
        goal = self._sample_goal(idx, window_size)
        for key, value in goal.items():
            sequence["goal_" + key] = value
        return sequence

    def _sample_goal(self, idx: int, window_size: int):
        # select strategy (geometric or hard-negative)
        if self.nn_steps_from_step and random.random() < self.hard_negatives_ratio:
            return self._sample_hard_negative_goal(idx, window_size)
        else:
            return self._sample_geometric_goal(idx, window_size)

    def _sample_geometric_goal(self, idx: int, window_size: int):
        """
        Sample a goal a few window sizes away using a geometric distribution delta.
        """
        seq_start = self.episode_lookup[idx]
        episode_end = self.find_episode_end(seq_start)
        if episode_end is None:
            logger.info("Could not find episode end in geometric sampling. Sampling a random goal.")
            return self._sample_random_goal()
        else:
            episode_end -= self.min_window_size  # last valid start for goal
        delta = np.random.default_rng().geometric(p=self.geom_sampling_prob)
        if self.geom_sampling_window_stride:
            goal_step = seq_start + (window_size - 1) * delta
        else:
            goal_step = seq_start + (window_size - 1) + delta
        goal_step = min(goal_step, episode_end)
        goal_idx = self.inv_episode_lookup.get(goal_step, None)
        if goal_idx is None:
            logger.info("No valid goal index found in geometric sampling. Sampling a random goal.")
            return self._sample_random_goal()
        return self._get_sequences(idx=goal_idx, window_size=1)

    def _sample_hard_negative_goal(self, idx: int, window_size: int):
        MAX_TRIES = 3
        end_step = self.episode_lookup[idx] + window_size - 1
        negative_samples = self.nn_steps_from_step.get(end_step, [])
        if not negative_samples:
            logger.info("No hard negative samples found. Sampling a random goal.")
            return self._sample_random_goal()
        for _ in range(MAX_TRIES):
            goal_step = np.random.choice(negative_samples)
            goal_idx = self.inv_episode_lookup.get(goal_step, None)
            if goal_idx is not None:
                break
        else:
            logger.info(f"No valid hard_negative goal index found after {MAX_TRIES} tries. Sampling a random goal.")
            return self._sample_random_goal()
        return self._get_sequences(idx=goal_idx, window_size=1)

    def _sample_random_goal(self):
        file_idx = np.random.randint(0, len(self.episode_lookup) - 1)
        return self._get_sequences(idx=file_idx, window_size=1)

    def get_nn_steps_from_step(self, hard_negatives_file: str) -> Dict[int, list[int]]:
        """Find steps with similar robot configuration"""
        logger.info(f"Loading hard negatives from {hard_negatives_file}.")
        nn_steps_from_step = {}
        nn_steps_from_step_path = Path(hard_negatives_file)
        nn_steps_from_step_path = nn_steps_from_step_path.expanduser()
        if nn_steps_from_step_path.is_file():
            with open(nn_steps_from_step_path) as f:
                nn_steps_from_step = json.load(f)
        else:
            logger.warning(f"Hard negatives file not found at {nn_steps_from_step_path}.")
            return {}

        data_type = "validation" if self.validation else "train"
        if data_type in nn_steps_from_step:
            nn_steps_from_step = {int(k): v for k, v in nn_steps_from_step[data_type].items()}  # maps str -> int
        return nn_steps_from_step

    def get_episode_start_end_ids(self) -> Dict[int, Tuple[int, int]]:
        file_path = self.abs_datasets_dir / "ep_start_end_ids.npy"
        if file_path.exists() and file_path.is_file():
            start_end_ids = np.load(self.abs_datasets_dir / "ep_start_end_ids.npy")
            return start_end_ids
        else:
            logger.warning(f"Episode start-end ids file not found at {file_path}.")
            return None

    def find_episode_end(self, step):
        # Efficient vectorized search using numpy
        mask = (self.ep_start_end_ids[:, 0] <= step) & (step <= self.ep_start_end_ids[:, 1])
        idx = np.where(mask)[0]
        if idx.size > 0:
            return self.ep_start_end_ids[idx[0], 1]
        return None
