from .cartpole import CartPoleVecEnv
from .pendulum import PendulumVecEnv


def make(env_id: str, num_envs: int = 1, **kwargs):
    eid = env_id.lower().replace("-", "")
    if eid.startswith("cartpole"):
        max_steps = 500 if "v1" in env_id.lower() else 200
        return CartPoleVecEnv(num_envs=num_envs, max_episode_steps=max_steps, **kwargs)
    if eid.startswith("pendulum"):
        return PendulumVecEnv(num_envs=num_envs, max_episode_steps=200, **kwargs)
    raise ValueError(f"Unknown env: {env_id}")
