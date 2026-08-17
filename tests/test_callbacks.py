"""Tests for SAC fixed ent_coef (issue #4) and callbacks (eval / early stopping)."""

import csv
import os

import mlx.core as mx

from dirty_mlx_ml.reinforcement import EvalCallback, SAC, StopTrainingOnRewardThreshold, make


def test_sac_fixed_ent_coef(tmp_path):
    # issue #4: passing a float ent_coef must not crash
    env = make("Pendulum-v1", num_envs=1, seed=0)
    log_dir = str(tmp_path / "sac")
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=50,
        buffer_size=10_000,
        batch_size=64,
        train_freq=1,
        gradient_steps=1,
        ent_coef=0.1,
        seed=0,
        log_dir=log_dir,
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(total_timesteps=200, log_interval=4)
    csv_path = os.path.join(log_dir, "sac_progress.csv")
    assert os.path.isfile(csv_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    ent_coefs = [float(r["train/ent_coef"]) for r in rows if r.get("train/ent_coef")]
    assert ent_coefs, "expected ent_coef to be logged"
    assert all(abs(c - 0.1) < 1e-3 for c in ent_coefs)


def test_sac_auto_ent_coef_still_works(tmp_path):
    env = make("Pendulum-v1", num_envs=1, seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=50,
        buffer_size=5000,
        batch_size=64,
        seed=0,
        ent_coef="auto",
        log_dir=str(tmp_path / "auto"),
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(total_timesteps=200, log_interval=4)
    assert model.num_timesteps >= 200


def test_eval_callback_and_early_stopping(tmp_path):
    env = make("Pendulum-v1", num_envs=1, seed=0)
    eval_env = make("Pendulum-v1", num_envs=1, seed=1)
    stop = StopTrainingOnRewardThreshold(reward_threshold=-1e6)
    eval_cb = EvalCallback(eval_env, callback_on_new_best=stop, eval_freq=100, n_eval_episodes=1)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=50,
        buffer_size=5000,
        batch_size=64,
        seed=0,
        log_dir=str(tmp_path / "cb"),
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(total_timesteps=500, log_interval=1, callback=eval_cb)
    # eval fired at least once and stopped early (threshold is trivially met)
    assert eval_cb.n_calls >= 1
    assert mx.isfinite(mx.array(eval_cb.last_mean_reward)).item()
    assert model.num_timesteps < 500
