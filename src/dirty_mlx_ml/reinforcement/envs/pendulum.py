import math

import mlx.core as mx

from ..spaces import Box


def _angle_normalize(x):
    return ((x + math.pi) % (2 * math.pi)) - math.pi


class PendulumVecEnv:
    """Vectorized Pendulum-v1, pure MLX (Gymnasium physics)."""

    def __init__(self, num_envs: int = 1, max_episode_steps: int = 200, g: float = 10.0, seed: int | None = None):
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        high = mx.array([1.0, 1.0, self.max_speed], dtype=mx.float32)
        self.observation_space = Box(low=-high, high=high, shape=(3,), dtype="float32")
        self.action_space = Box(
            low=mx.array([-self.max_torque], dtype=mx.float32),
            high=mx.array([self.max_torque], dtype=mx.float32),
            shape=(1,),
            dtype="float32",
        )
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self._th = mx.zeros((num_envs,), dtype=mx.float32)
        self._thdot = mx.zeros((num_envs,), dtype=mx.float32)
        self._steps = mx.zeros((num_envs,), dtype=mx.int32)
        self._prev_done = mx.zeros((num_envs,), dtype=mx.bool_)
        if seed is not None:
            mx.random.seed(seed)

    def _obs(self):
        return mx.stack([mx.cos(self._th), mx.sin(self._th), self._thdot], axis=1).astype(mx.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            mx.random.seed(seed)
        self._th = mx.random.uniform(-math.pi, math.pi, (self.num_envs,)).astype(mx.float32)
        self._thdot = mx.random.uniform(-1.0, 1.0, (self.num_envs,)).astype(mx.float32)
        self._steps = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._prev_done = mx.zeros((self.num_envs,), dtype=mx.bool_)
        return self._obs(), {}

    def step(self, action):
        u = mx.clip(mx.reshape(action, (self.num_envs, -1))[:, 0], -self.max_torque, self.max_torque)
        th, thdot = self._th, self._thdot
        costs = _angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        newthdot = thdot + (3 * self.g / (2 * self.l) * mx.sin(th) + 3.0 / (self.m * self.l**2) * u) * self.dt
        newthdot = mx.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        reward = -costs
        steps = self._steps + 1
        truncated = steps >= self.max_episode_steps
        terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)

        pd = self._prev_done
        new_th = mx.random.uniform(-math.pi, math.pi, (self.num_envs,)).astype(mx.float32)
        new_thdot = mx.random.uniform(-1.0, 1.0, (self.num_envs,)).astype(mx.float32)
        th = mx.where(pd, new_th, newth)
        thdot = mx.where(pd, new_thdot, newthdot)
        steps = mx.where(pd, mx.zeros_like(steps), steps)
        reward = mx.where(pd, mx.zeros_like(reward), reward)
        truncated = mx.where(pd, mx.zeros_like(truncated), truncated)

        done = terminated | truncated
        self._th, self._thdot, self._steps, self._prev_done = th, thdot, steps, done
        obs = mx.stack([mx.cos(th), mx.sin(th), thdot], axis=1).astype(mx.float32)
        return obs, reward.astype(mx.float32), done, {"timeouts": truncated.astype(mx.float32)}

    def close(self):
        pass
