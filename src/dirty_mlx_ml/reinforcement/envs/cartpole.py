import math

import mlx.core as mx

from ..spaces import Box, Discrete


class CartPoleVecEnv:
    """Vectorized CartPole-v1, pure MLX (Gymnasium physics)."""

    def __init__(self, num_envs: int = 1, max_episode_steps: int = 500, seed: int | None = None):
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        high = mx.array([self.x_threshold * 2, 1e8, self.theta_threshold * 2, 1e8], dtype=mx.float32)
        self.observation_space = Box(low=-high, high=high, shape=(4,), dtype="float32")
        self.action_space = Discrete(2)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self._state = mx.zeros((num_envs, 4), dtype=mx.float32)
        self._steps = mx.zeros((num_envs,), dtype=mx.int32)
        self._prev_done = mx.zeros((num_envs,), dtype=mx.bool_)
        if seed is not None:
            mx.random.seed(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            mx.random.seed(seed)
        self._state = mx.random.uniform(-0.05, 0.05, (self.num_envs, 4)).astype(mx.float32)
        self._steps = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._prev_done = mx.zeros((self.num_envs,), dtype=mx.bool_)
        return self._state, {}

    def step(self, action):
        action = mx.reshape(action.astype(mx.float32), (self.num_envs,))
        x, x_dot, theta, theta_dot = self._state[:, 0], self._state[:, 1], self._state[:, 2], self._state[:, 3]
        force = mx.where(action > 0.5, self.force_mag, -self.force_mag)
        costheta, sintheta = mx.cos(theta), mx.sin(theta)
        temp = (force + self.polemass_length * (theta_dot * theta_dot) * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * (costheta * costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        state = mx.stack([x, x_dot, theta, theta_dot], axis=1)

        terminated = (x < -self.x_threshold) | (x > self.x_threshold) | (theta < -self.theta_threshold) | (
            theta > self.theta_threshold
        )
        steps = self._steps + 1
        truncated = steps >= self.max_episode_steps
        reward = mx.ones((self.num_envs,), dtype=mx.float32)

        # always-mask autoreset (no host sync)
        pd = self._prev_done
        new_s = mx.random.uniform(-0.05, 0.05, (self.num_envs, 4)).astype(mx.float32)
        mask = pd.astype(mx.float32)[:, None]
        state = state * (1.0 - mask) + new_s * mask
        steps = mx.where(pd, mx.zeros_like(steps), steps)
        reward = mx.where(pd, mx.zeros_like(reward), reward)
        terminated = mx.where(pd, mx.zeros_like(terminated), terminated)
        truncated = mx.where(pd, mx.zeros_like(truncated), truncated)

        done = terminated | truncated
        self._state, self._steps, self._prev_done = state, steps, done
        return state, reward, done, {}

    def close(self):
        pass
