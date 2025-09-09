import logging
import json
import random
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from calvin_agent.datasets.shm_dataset import ShmDataset

logger = logging.getLogger(__name__)


class GoalAugmentedShmDataset(ShmDataset):
    """
    This dataset class augments the ShmDataset for shared memory such that every sampled sequence
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
        self.inv_episode_lookup = {}

    def setup_shm_lookup(self, shm_lookup: Dict) -> None:
        super().setup_shm_lookup(shm_lookup)
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
        delta = np.random.default_rng().geometric(p=self.geom_sampling_prob)
        if self.geom_sampling_window_stride:
            goal_idx = idx + (window_size - 1) * delta
        else:
            goal_idx = idx + (window_size - 1) + delta
        if (goal_idx >= len(self.episode_lookup)) or (
            self.episode_lookup[goal_idx] - self.episode_lookup[idx] != goal_idx - idx
        ):
            # if goal index is outside of episode
            # or if goal index is from a different episode
            logger.info("Geometric sampling failed. sampling a random goal.")
            return self._sample_random_goal()
        return self._get_sequences(idx=goal_idx, window_size=1)

    def _sample_hard_negative_goal(self, idx: int, window_size: int):
        end_step = self.episode_lookup[idx] + window_size - 1
        negative_samples = self.nn_steps_from_step.get(end_step, [])
        if not negative_samples:
            logger.info("No hard negative samples found. Sampling a random goal.")
            return self._sample_random_goal()
        goal_step = np.random.choice(negative_samples)
        goal_idx = self.inv_episode_lookup.get(goal_step, None)
        if goal_idx is None:
            logger.info("No valid hard_negative goal index found. Sampling a random goal.")
            return self._sample_random_goal()
        return self._get_sequences(idx=goal_idx, window_size=1)

    def _sample_random_goal(self):
        file_idx = np.random.choice(self.episode_lookup)
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

        data_type = "validation" if self.validation else "train"
        if data_type in nn_steps_from_step:
            nn_steps_from_step = {int(k): v for k, v in nn_steps_from_step[data_type].items()}  # maps str -> int
        return nn_steps_from_step
