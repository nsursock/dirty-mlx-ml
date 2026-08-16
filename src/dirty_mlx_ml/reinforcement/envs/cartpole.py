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
        self._key = mx.random.key(seed if seed is not None else 0)
        self._step_fn = None

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._key = mx.random.key(seed)
        k, self._key = mx.random.split(self._key)
        self._state = mx.random.uniform(-0.05, 0.05, (self.num_envs, 4), key=k).astype(mx.float32)
        self._steps = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._prev_done = mx.zeros((self.num_envs,), dtype=mx.bool_)
        return self._state, {}

    def _build_step(self):
        num_envs = self.num_envs
        max_steps = self.max_episode_steps
        force_mag = self.force_mag
        polemass_length = self.polemass_length
        total_mass = self.total_mass
        length = self.length
        masspole = self.masspole
        gravity = self.gravity
        tau = self.tau
        x_threshold = self.x_threshold
        theta_threshold = self.theta_threshold

        def step(state, steps, prev_done, key, action):
            action = mx.reshape(action.astype(mx.float32), (num_envs,))
            x = state[:, 0]
            x_dot = state[:, 1]
            theta = state[:, 2]
            theta_dot = state[:, 3]
            force = mx.where(action > 0.5, force_mag, -force_mag)
            costheta = mx.cos(theta)
            sintheta = mx.sin(theta)
            temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
            thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)
            )
            xacc = temp - polemass_length * thetaacc * costheta / total_mass
            x = x + tau * x_dot
            x_dot = x_dot + tau * xacc
            theta = theta + tau * theta_dot
            theta_dot = theta_dot + tau * thetaacc
            state2 = mx.stack([x, x_dot, theta, theta_dot], axis=1)

            terminated = (x < -x_threshold) | (x > x_threshold) | (theta < -theta_threshold) | (
                theta > theta_threshold
            )
            steps2 = steps + 1
            truncated = steps2 >= max_steps
            reward = mx.ones((num_envs,), dtype=mx.float32)

            k, kreset = mx.random.split(key)
            new_s = mx.random.uniform(-0.05, 0.05, (num_envs, 4), key=kreset).astype(mx.float32)
            mask = prev_done.astype(mx.float32)[:, None]
            state2 = state2 * (1.0 - mask) + new_s * mask
            steps2 = mx.where(prev_done, mx.zeros_like(steps2), steps2)
            reward = mx.where(prev_done, mx.zeros_like(reward), reward)
            terminated = mx.where(prev_done, mx.zeros_like(terminated), terminated)
            truncated = mx.where(prev_done, mx.zeros_like(truncated), truncated)
            done = terminated | truncated
            return state2, steps2, done, k, state2, reward, done, truncated

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
        return obs, reward, done, {}

    def close(self):
        pass
