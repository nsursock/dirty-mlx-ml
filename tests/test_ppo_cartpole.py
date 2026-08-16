import csv
import os

import mlx.core as mx

from dirty_mlx_ml.reinforcement import PPO
from dirty_mlx_ml.reinforcement.envs import make

PPO_TIME = {"time/fps", "time/iterations", "time/time_elapsed", "time/total_timesteps"}
PPO_ROLLOUT = {"rollout/ep_rew_mean", "rollout/ep_len_mean", "rollout/success_rate"}
PPO_TRAIN = {
    "train/approx_kl",
    "train/clip_fraction",
    "train/clip_range",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/learning_rate",
    "train/loss",
    "train/n_updates",
    "train/explained_variance",
    "train/std",
}


def test_ppo_cartpole_learn_and_csv(tmp_path):
    env = make("CartPole-v1", num_envs=1, seed=0)
    log_dir = str(tmp_path / "ppo_run")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        n_epochs=2,
        learning_rate=3e-4,
        seed=0,
        log_dir=log_dir,
    )
    model.learn(total_timesteps=512, log_interval=1)
    csv_path = os.path.join(log_dir, "ppo_progress.csv")
    assert os.path.isfile(csv_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    keys = set(rows[-1].keys())
    missing = (PPO_TIME | PPO_ROLLOUT | PPO_TRAIN) - keys
    assert not missing, missing
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape[0] == 1


def test_ppo_improves_or_runs():
    env = make("CartPole-v1", num_envs=1, seed=42)
    model = PPO("MlpPolicy", env, n_steps=128, batch_size=64, n_epochs=4, seed=42, log_dir=None)
    model.learn(total_timesteps=1024, log_interval=1)
    obs, _ = env.reset()
    total = 0.0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _ = env.step(action.reshape(-1))
        total += float(rew.item()) if rew.size == 1 else float(mx.sum(rew).item())
        if bool(done.item() if done.size == 1 else done[0].item()):
            break
    assert total > 0
