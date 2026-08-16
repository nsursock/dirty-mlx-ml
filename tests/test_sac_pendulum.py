import csv
import os

import mlx.core as mx

from dirty_mlx_ml.reinforcement import SAC
from dirty_mlx_ml.reinforcement.envs import make

SAC_TIME = {"time/fps", "time/iterations", "time/time_elapsed", "time/total_timesteps"}
SAC_ROLLOUT = {"rollout/ep_rew_mean", "rollout/ep_len_mean", "rollout/success_rate"}
SAC_TRAIN = {
    "train/loss/policy",
    "train/critic_loss",
    "train/loss/q1",
    "train/loss/q2",
    "train/loss/alpha",
    "train/policy/alpha",
    "train/value/q_mean",
    "train/policy/log_pi_mean",
    "train/ent_coef_loss",
    "train/learning_rate",
    "train/loss",
    "train/n_updates",
}


def test_sac_pendulum_learn_and_csv(tmp_path):
    env = make("Pendulum-v1", num_envs=1, seed=0)
    log_dir = str(tmp_path / "sac_run")
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=50,
        buffer_size=10_000,
        batch_size=64,
        train_freq=1,
        gradient_steps=1,
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
    keys = set(rows[-1].keys())
    missing = (SAC_TIME | SAC_ROLLOUT | SAC_TRAIN) - keys
    assert not missing, missing
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1, 1)


def test_sac_runs_episode():
    env = make("Pendulum-v1", num_envs=1, seed=1)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=20,
        buffer_size=5000,
        batch_size=32,
        seed=1,
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(total_timesteps=100, log_interval=10)
    obs, _ = env.reset()
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _ = env.step(action)
        if bool(done.item()):
            obs, _ = env.reset()
    assert True
