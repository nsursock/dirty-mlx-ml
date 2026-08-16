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
        self._state = mx.zeros((num_envs, 2), dtype=mx.float32)  # [th, thdot]
        self._steps = mx.zeros((num_envs,), dtype=mx.int32)
        self._prev_done = mx.zeros((num_envs,), dtype=mx.bool_)
        self._key = mx.random.key(seed if seed is not None else 0)
        self._step_fn = None

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._key = mx.random.key(seed)
        k, self._key = mx.random.split(self._key)
        k_th, k_thdot = mx.random.split(k)
        th = mx.random.uniform(-math.pi, math.pi, (self.num_envs,), key=k_th).astype(mx.float32)
        thdot = mx.random.uniform(-1.0, 1.0, (self.num_envs,), key=k_thdot).astype(mx.float32)
        self._state = mx.stack([th, thdot], axis=1)
        self._steps = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._prev_done = mx.zeros((self.num_envs,), dtype=mx.bool_)
        obs = mx.stack([mx.cos(th), mx.sin(th), thdot], axis=1).astype(mx.float32)
        return obs, {}

    def _build_step(self):
        num_envs = self.num_envs
        max_steps = self.max_episode_steps
        max_speed = self.max_speed
        max_torque = self.max_torque
        dt = self.dt
        g = self.g
        m = self.m
        l = self.l

        def step(state, steps, prev_done, key, action):
            u = mx.clip(mx.reshape(action, (num_envs, -1))[:, 0], -max_torque, max_torque)

            k, kr_th, kr_thdot = mx.random.split(key, 3)
            reset_th = mx.random.uniform(-math.pi, math.pi, (num_envs,), key=kr_th).astype(mx.float32)
            reset_thdot = mx.random.uniform(-1.0, 1.0, (num_envs,), key=kr_thdot).astype(mx.float32)
            th0 = mx.where(prev_done, reset_th, state[:, 0])
            thdot0 = mx.where(prev_done, reset_thdot, state[:, 1])
            steps_prev = mx.where(prev_done, mx.zeros_like(steps), steps)

            costs = _angle_normalize(th0) ** 2 + 0.1 * thdot0**2 + 0.001 * (u**2)
            newthdot = thdot0 + (3 * g / (2 * l) * mx.sin(th0) + 3.0 / (m * l**2) * u) * dt
            newthdot = mx.clip(newthdot, -max_speed, max_speed)
            newth = th0 + newthdot * dt
            reward = -costs
            steps2 = steps_prev + 1
            truncated = steps2 >= max_steps
            terminated = mx.zeros((num_envs,), dtype=mx.bool_)

            done = terminated | truncated
            state2 = mx.stack([newth, newthdot], axis=1)
            obs = mx.stack([mx.cos(newth), mx.sin(newth), newthdot], axis=1).astype(mx.float32)
            return state2, steps2, done, k, obs, reward, done, truncated

        self._step_fn = mx.compile(step)

    def step(self, action):
        if self._step_fn is None:
            self._build_step()
        state, steps, prev_done, key, obs, reward, done, truncated = self._step_fn(
            self._state, self._steps, self._prev_done, self._key, action
        )
        self._state = state
        self._steps = steps
        self._prev_done = prev_done
        self._key = key
        return obs, reward.astype(mx.float32), done, {"timeouts": truncated.astype(mx.float32)}

    def close(self):
        pass
