import gymnasium as gym


def make_env(env_cfg):
    sel_env_spec = None
    all_envs = gym.registry.values()
    for env_spec in all_envs:
        if env_cfg.name in env_spec.id:
            sel_env_spec = env_spec
            break
    if sel_env_spec is None:
        return None
    if "taowm" in sel_env_spec.entry_point:
        env = gym.make(sel_env_spec.id, **env_cfg)
        env._max_episode_steps = env_cfg.max_episode_steps
    else:
        env = gym.make(sel_env_spec.id)
    return env
