"""Custom environments for TAO-WM."""

import gymnasium as gym
from gymnasium.envs.registration import register

# Register custom environments
register(
    id="play-table-v0",
    entry_point="taowm.environment:PlayTableEnv",
    max_episode_steps=200,
)

register(
    id="goal-conditioned-v0",
    entry_point="taowm.environment:GoalConditionedEnv",
    max_episode_steps=200,
)
# Optional: Import the environment classes to make them available
from taowm.environment.play_table_env import PlayTableEnv
from taowm.environment.goal_conditioned_env import GoalConditionedEnv

__all__ = ["PlayTableEnv", "GoalConditionedEnv"]
