"""Tests for the VecNormalize wrapper (issue #1) and SAC high-env stability (issue #3)."""

import math

import mlx.core as mx

from dirty_mlx_ml.reinforcement import PPO, SAC, VecNormalize, make
from dirty_mlx_ml.reinforcement.spaces import Box


class _SynthVecEnv:
    """Minimal continuous vec env with a reward-scale knob.

    A large ``rew_scale`` makes the reward magnitude large, which drives the
    SAC critic loss to huge values (the high-env NaN failure mode in issue #3)
    unless rewards are normalized.
    """

    def __init__(self, num_envs, obs_dim=8, seed=0, rew_scale=1.0):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.rew_scale = rew_scale
        high = mx.full((obs_dim,), 10.0)
        self.observation_space = Box(low=-high, high=high, shape=(obs_dim,), dtype="float32")
        self.action_space = Box(low=mx.array([-1.0]), high=mx.array([1.0]), shape=(1,), dtype="float32")
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.max_episode_steps = 200
        self._key = mx.random.key(seed)
        self._state = None
        self._steps = None
        self._prev_done = None
        self._step_fn = None

    def reset(self, seed=None):
        if seed is not None:
            self._key = mx.random.key(seed)
        k, self._key = mx.random.split(self._key)
        self._state = mx.random.uniform(-1.0, 1.0, (self.num_envs, self.obs_dim), key=k)
        self._steps = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._prev_done = mx.zeros((self.num_envs,), dtype=mx.bool_)
        return self._state, {}

    def _build_step(self):
        n = self.num_envs
        d = self.obs_dim
        rs = self.rew_scale

        def step(state, steps, prev_done, key, action):
            k, self_key = mx.random.split(key)
            a = mx.reshape(action, (n, 1))
            signal = mx.sum(state, axis=1)
            reward = rs * (a[:, 0] * signal) - 0.01 * (a[:, 0] ** 2)
            state2 = state + 0.1 * a + 0.1 * mx.random.normal((n, d), key=k)
            state2 = mx.clip(state2, -10.0, 10.0)
            steps2 = steps + 1
            truncated = steps2 >= 200
            terminated = mx.zeros((n,), dtype=mx.bool_)
            done = terminated | truncated
            return state2, steps2, done, self_key, state2, reward, done, truncated

        self._step_fn = mx.compile(step)

    def step(self, action):
        if self._step_fn is None:
            self._build_step()
        state, steps, prev_done, key, obs, rew, done, trunc = self._step_fn(
            self._state, self._steps, self._prev_done, self._key, action
        )
        self._state = state
        self._steps = steps
        self._prev_done = prev_done
        self._key = key
        return obs, rew.astype(mx.float32), done, {"timeouts": trunc.astype(mx.float32)}

    def close(self):
        pass


def _last_critic_loss(log_dir):
    import os

    rows = open(os.path.join(log_dir, "sac_progress.csv")).read().splitlines()
    hdr = rows[0].split(",")
    ci = hdr.index("train/critic_loss")
    return [float(r.split(",")[ci]) for r in rows[1:]]


def test_vec_normalize_running_stats():
    env = make("Pendulum-v1", num_envs=4, seed=0)
    vn = VecNormalize(env, norm_obs=True, norm_reward=True)
    obs, _ = vn.reset()
    assert obs.shape == (4, 3)
    for _ in range(50):
        action = mx.random.uniform(-2.0, 2.0, (4, 1))
        obs, reward, done, _ = vn.step(action)
    # stats advanced
    assert float(vn.norm_state["obs_count"]) > 0
    # normalized obs are finite and clipped
    assert bool(mx.all(mx.isfinite(obs)))
    assert bool(mx.all(mx.abs(obs) <= 10.0 + 1e-4))
    # normalized rewards are finite
    assert bool(mx.all(mx.isfinite(reward)))


def test_vec_normalize_unnormalize_reward():
    env = make("Pendulum-v1", num_envs=2, seed=0)
    vn = VecNormalize(env, norm_obs=False, norm_reward=True)
    obs, _ = vn.reset()
    for _ in range(30):
        action = mx.random.uniform(-2.0, 2.0, (2, 1))
        obs, reward, done, _ = vn.step(action)
    raw = vn.unnormalize_reward(reward)
    assert raw.shape == reward.shape
    assert bool(mx.all(mx.isfinite(raw)))


def test_sac_vec_normalize_stays_finite_at_scale(tmp_path):
    # issue #3: high env count + large reward scale -> finite with VecNormalize
    env = VecNormalize(_SynthVecEnv(num_envs=256, obs_dim=8, seed=0, rew_scale=30.0))
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=200,
        buffer_size=100_000,
        batch_size=128,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        seed=0,
        log_dir=str(tmp_path / "vn"),
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=2048, log_interval=1)
    losses = _last_critic_loss(str(tmp_path / "vn"))
    assert all(math.isfinite(x) for x in losses)
    assert max(losses) < 100.0


def test_sac_without_norm_has_large_critic_loss(tmp_path):
    # Same env without normalization: critic loss explodes (the NaN precursor).
    env = _SynthVecEnv(num_envs=256, obs_dim=8, seed=0, rew_scale=30.0)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=200,
        buffer_size=100_000,
        batch_size=128,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        seed=0,
        log_dir=str(tmp_path / "raw"),
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=2048, log_interval=1)
    losses = _last_critic_loss(str(tmp_path / "raw"))
    assert max(losses) > 100.0


def test_sac_max_grad_norm_and_max_ent_coef(tmp_path):
    env = make("Pendulum-v1", num_envs=1, seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=50,
        buffer_size=5000,
        batch_size=64,
        seed=0,
        max_grad_norm=1.0,
        max_ent_coef=10.0,
        log_dir=str(tmp_path / "clip"),
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(total_timesteps=200, log_interval=4)
    assert model.num_timesteps >= 200


def test_ppo_vec_normalize_cartpole():
    env = VecNormalize(make("CartPole-v1", num_envs=4, seed=0))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        seed=0,
        policy_kwargs={"net_arch": [32, 32]},
    )
    model.learn(total_timesteps=1024, log_interval=1)
    assert model.num_timesteps >= 1024
