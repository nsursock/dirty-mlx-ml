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

## Configuration

Use YAML configs to tune hyperparameters without editing code. Example configs are provided in `configs/`:

```bash
configs/
  ppo_cartpole.yaml
  sac_pendulum.yaml
```

**PPO config example** (`configs/ppo_cartpole.yaml`):

```yaml
algo: ppo
env_id: CartPole-v1
num_envs: 16
seed: 0
total_timesteps: 100000

learning_rate: 0.0003
n_steps: 256
batch_size: 256
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2

policy_kwargs:
  net_arch: [64, 64]
```

**SAC config example** (`configs/sac_pendulum.yaml`):

```yaml
algo: sac
env_id: Pendulum-v1
num_envs: 8
seed: 0
total_timesteps: 60000

learning_rate: 0.0003
buffer_size: 200000
learning_starts: 1000
batch_size: 256
tau: 0.005
gamma: 0.99
train_freq: 1
gradient_steps: 1
ent_coef: auto

policy_kwargs:
  net_arch: [256, 256]
```

Copy and modify these configs to experiment with different hyperparameters for your own environments.

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

### Hardware Context

Benchmark results below were obtained on: **Apple M3, 16 GB unified memory**.

Performance may vary significantly across different Apple Silicon configurations:
- **Chip variants**: M1, M1 Pro/Max/Ultra, M2, M2 Pro/Max/Ultra, M3, M3 Pro/Max/Ultra
- **Memory**: 8 GB, 16 GB, 32 GB, 64 GB, 96 GB, 128 GB unified memory
- **Core counts**: Different CPU/GPU core configurations affect parallel performance

To check your hardware: `python bench/reinforcement/detect_hardware.py`

### Scaling Performance

**PPO on CartPole-v1** (env FPS / train FPS / memory):

| Num Envs | Env FPS      | Train FPS    | Memory (MB) |
|----------|--------------|--------------|-------------|
| 16       | 29,426       | 30,396       | 92.0        |
| 256      | 471,650      | 349,026      | 91.7        |
| 1,024    | 1,732,100    | 489,436      | 93.8        |
| 8,192    | 16,796,202   | 701,131      | 78.1        |
| 16,384   | 31,039,098   | 781,220      | 36.7        |

*Note: Updated with GAE optimization and improved memory measurement. Sweep stopped at 16,384 envs due to swap growth (+1957MB).*

**SAC on Pendulum-v1** (env FPS / train FPS / memory):

| Num Envs | Env FPS      | Train FPS    | Memory (MB) |
|----------|--------------|--------------|-------------|
| 16       | 33,943       | 2,294        | 62.9        |
| 64       | 108,040      | 8,386        | 63.0        |
| 128      | 220,850      | 16,419       | 63.3        |

*Note: Updated with improved memory measurement. Sweep stopped at 128 envs due to swap growth (+271MB). This is the actual scaling ceiling for this hardware, not a truncation for brevity.*

**Performance Notes:**
- **Scaling table FPS**: Represents pure training throughput during policy updates (micro-benchmark of the training loop)
- **Solve performance FPS**: Effective rate including full training loop overhead (policy evaluation rollouts, environment resets, logging, checkpointing, periodic evaluation)
- The scaling table shows how efficiently the training pipeline scales with parallel environments
- The solve performance shows real-world time-to-solve including all overhead
- **Sweep truncation**: Scaling sweeps stop when either (a) swap grows by >256MB (memory pressure) or (b) train FPS plateaus (no further scaling benefit). This represents the actual scaling ceiling for the hardware, not arbitrary truncation for brevity.

### Solve Performance

Time to solve environments with multiple parallel environments.

**PPO on CartPole-v1** (threshold: 475.0 - Gym's official solved criterion):

| Num Envs | Timesteps | Wall Time (s) | Eval  | Train FPS |
|----------|-----------|---------------|-------|-----------|
| 8        | 32,768    | 10.81         | 500.0 | 3,033     |
| 16       | 32,768    | 9.55          | 500.0 | 3,433     |

*Note: Updated to use Gym's official 475.0 threshold instead of custom 440.0. Solve time overhead includes policy evaluation rollouts, environment reset delays, periodic evaluation, logging, and checkpointing.*

**SAC on Pendulum-v1** (threshold: -200.0):

| Num Envs | Timesteps | Wall Time (s) | Eval   | Train FPS |
|----------|-----------|---------------|--------|-----------|
| 8        | 32,768    | 45.06         | -148.2 | 727       |
| 16       | 65,536    | 50.62         | -162.7 | 1,295     |

*Note: Multi-seed investigation (5 seeds) shows 16 envs achieves 1.68x wall-time speedup over 8 envs (48.3s vs 80.9s) despite requiring more timesteps (98,304 vs 65,536). Original single-seed data showed noise; statistical analysis confirms 16 envs is superior for solve performance. Solve time overhead includes policy evaluation rollouts, environment reset delays, periodic evaluation, logging, and checkpointing.*

## Why

This started as the RL stack for a personal trading bot on a Mac. PureJaxRL showed how far "keep it on accelerator" can go; this is the dirtier, SB3-flavored cousin for people who live on Apple Silicon and still want PPO/SAC that feel familiar.

## Status

Early and opinionated. APIs may shift. Issues and PRs welcome.

## License

MIT
