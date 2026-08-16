# dirty-mlx-ml

**Dirty** ML on Apple Silicon — a play on [PureJaxRL](https://github.com/luchris429/purejaxrl).

Reinforcement learning in [MLX](https://github.com/ml-explore/mlx), with a Stable-Baselines3-shaped API. Built for a trading bot; open-sourced so the rest of the community can train on Mac GPUs without the CUDA tax.

## Features

- **PPO** and **SAC** ports inspired by [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Runs on **Apple Silicon** via MLX (unified memory, Metal)
- Vectorized envs and training paths kept on-device where it matters
- Familiar `learn` / `predict` surface if you already know SB3
- Built-in **CartPole-v1** and **Pendulum-v1** (multi-env)
- CSV logging of rollout / train / time metrics

## Install

Requires **Python ≥ 3.10** and a Mac with Apple Silicon (MLX).

```bash
pip install -e .
# or with dev/test extras:
pip install -e ".[dev]"
```

## Quick start

### PPO on CartPole

```python
from dirty_mlx_ml.reinforcement import PPO
from dirty_mlx_ml.reinforcement.envs import make

env = make("CartPole-v1", num_envs=16, seed=0)
model = PPO(
    "MlpPolicy",
    env,
    n_steps=256,
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    seed=0,
    log_dir="logs/ppo",
)
model.learn(total_timesteps=80_000)

obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

### SAC on Pendulum

```python
from dirty_mlx_ml.reinforcement import SAC
from dirty_mlx_ml.reinforcement.envs import make

env = make("Pendulum-v1", num_envs=8, seed=0)
model = SAC(
    "MlpPolicy",
    env,
    learning_starts=1000,
    buffer_size=100_000,
    batch_size=256,
    seed=0,
    log_dir="logs/sac",
    policy_kwargs={"net_arch": [256, 256]},
)
model.learn(total_timesteps=50_000)

obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

## Algorithms

| Algo | Action space | Notes |
|------|----------------|-------|
| **PPO** | Discrete & continuous | GAE, clipped surrogate, optional value clip / target KL |
| **SAC** | Continuous | Twin Q, auto entropy (`ent_coef="auto"`), Polyak targets |

Hyperparameters largely mirror SB3 defaults (`learning_rate`, `n_steps`, `gae_lambda`, `tau`, `train_freq`, etc.). Pass `policy_kwargs={"net_arch": [...]}` to set MLP widths.

## Environments

```python
from dirty_mlx_ml.reinforcement.envs import make

env = make("CartPole-v1", num_envs=16, seed=0)
env = make("Pendulum-v1", num_envs=8, seed=0)
```

Both are vectorized (`num_envs`) and return MLX arrays. API shape: `reset` → `(obs, info)`, `step(action)` → `(obs, reward, done, info)`.

Bring your own env by matching that interface and `observation_space` / `action_space` (see `spaces.py`).

## Project layout

```
src/dirty_mlx_ml/
  reinforcement/
    algorithms/   # PPO, SAC
    envs/         # CartPole, Pendulum
    buffers.py    # rollout + replay
    nn.py         # actors / critics
    spaces.py
    logger.py     # CSV progress logs
tests/
bench/
```

## Tests

```bash
pytest tests/ -q
# slower solve checks (CartPole / Pendulum thresholds + FPS):
pytest tests/ -q -m slow
```

## Benchmarks

Performance on Apple Silicon (M-series). Run with `python bench/reinforcement/benchmark_scaling.py` and `python bench/reinforcement/benchmark_solve.py`.

### Scaling Performance

**PPO on CartPole-v1** (env FPS / train FPS / memory):

| Num Envs | Env FPS      | Train FPS    | Memory (MB) |
|----------|--------------|--------------|-------------|
| 16       | 133,550      | 29,884       | 94.9        |
| 256      | 6,035,479    | 294,716      | 96.7        |
| 1,024    | 20,838,734   | 502,415      | 98.2        |
| 8,192    | 126,793,381  | 688,175      | 71.2        |

**SAC on Pendulum-v1** (env FPS / train FPS / memory):

| Num Envs | Env FPS      | Train FPS    | Memory (MB) |
|----------|--------------|--------------|-------------|
| 16       | 287,410      | 2,486        | 66.7        |
| 256      | 5,991,027    | 32,234       | 68.5        |
| 1,024    | 25,261,542   | 118,872      | 68.5        |
| 8,192    | 186,075,321  | 676,473      | 67.8        |

### Solve Performance

Time to solve environments with multiple parallel environments.

**PPO on CartPole-v1** (threshold: 450.0):

| Num Envs | Timesteps | Wall Time (s) | Train FPS |
|----------|-----------|---------------|-----------|
| 256      | 294,912   | 15.73         | 18,751    |
| 512      | 196,608   | 10.38         | 18,947    |
| 1,024    | 393,216   | 21.13         | 18,606    |

**SAC on Pendulum-v1** (threshold: -300.0):

| Num Envs | Timesteps | Wall Time (s) | Train FPS |
|----------|-----------|---------------|-----------|
| 256      | 32,768    | 88.77         | 369       |
| 512      | 49,152    | 32.84         | 1,497     |
| 1,024    | 81,920    | 28.18         | 2,907     |

## Why

This started as the RL stack for a personal trading bot on a Mac. PureJaxRL showed how far “keep it on accelerator” can go; this is the dirtier, SB3-flavored cousin for people who live on Apple Silicon and still want PPO/SAC that feel familiar.

## Status

Early and opinionated. APIs may shift. Issues and PRs welcome.

## License

MIT
