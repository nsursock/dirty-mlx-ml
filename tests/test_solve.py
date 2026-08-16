"""Solve validation: CartPole (PPO) and Pendulum (SAC) with multi-env speed."""
import time
import os

import mlx.core as mx
import pytest

from dirty_mlx_ml.reinforcement import PPO, SAC
from dirty_mlx_ml.reinforcement.envs import make

LOG_DIR = "logs"


def _eval_mean(model, env_id, n_eps=20, seed=123, max_steps=500):
    env = make(env_id, num_envs=1, seed=seed)
    total = 0.0
    for e in range(n_eps):
        obs, _ = env.reset(seed=seed + e)
        done = False
        steps = 0
        ep = 0.0
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            if action.ndim > 1 and action.shape[-1] == 1 and env.action_space.__class__.__name__ == "Discrete":
                action = action.reshape(-1)
            elif env.action_space.__class__.__name__ == "Discrete":
                action = action.reshape(-1)
            obs, rew, done, _ = env.step(action)
            ep += float(rew.reshape(-1)[0].item())
            done = bool(done.reshape(-1)[0].item())
            steps += 1
        total += ep
    return total / n_eps


@pytest.mark.slow
def test_ppo_solves_cartpole():
    os.makedirs(LOG_DIR, exist_ok=True)
    env = make("CartPole-v1", num_envs=8, seed=0)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=128,
        n_epochs=20,
        learning_rate=1e-3,
        seed=0,
        log_dir=os.path.join(LOG_DIR, "ppo"),
    )
    t0 = time.time()
    model.learn(total_timesteps=160_000, log_interval=5)
    elapsed = time.time() - t0
    fps = 160_000 / max(elapsed, 1e-9)
    mean = _eval_mean(model, "CartPole-v1", n_eps=20, max_steps=500)
    assert mean >= 475.0, f"CartPole not solved: mean={mean:.1f} fps={fps:.0f}"
    assert fps > 2000, f"too slow: fps={fps:.0f}"


@pytest.mark.slow
def test_sac_solves_pendulum():
    os.makedirs(LOG_DIR, exist_ok=True)
    env = make("Pendulum-v1", num_envs=16, seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=500,
        buffer_size=50_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        seed=0,
        log_dir=os.path.join(LOG_DIR, "sac"),
        policy_kwargs={"net_arch": [256, 256]},
    )
    t0 = time.time()
    model.learn(total_timesteps=130_000, log_interval=200)
    elapsed = time.time() - t0
    fps = 130_000 / max(elapsed, 1e-9)
    mean = _eval_mean(model, "Pendulum-v1", n_eps=10, max_steps=200)
    assert mean >= -200.0, f"Pendulum not solved: mean={mean:.1f} fps={fps:.0f}"
    assert fps > 1500, f"too slow: fps={fps:.0f}"
