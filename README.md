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

Primary metric is **time-to-competence (TTS)**: wall-clock until a competent policy, not raw env FPS.

```bash
python bench/reinforcement/benchmark_solve.py --warm --seeds 5
python bench/reinforcement/benchmark_scaling.py   # systems microbench (secondary)
python bench/reinforcement/benchmark_baselines.py # cross-framework baselines (optional deps)
python bench/reinforcement/detect_hardware.py
```

### Hardware

Results below: **Apple M3, 16 GB unified memory**. Numbers vary a lot across M-series chips and memory configs.

### Time-to-Competence (primary)

Train until threshold; stop on first **probe + confirm**. Median ± std over seeds. `--warm` excludes one JIT train cycle from the timer.

Protocol: vectorized probe every few thousand steps → one confirmation eval → stop. Thresholds: CartPole **475.0** (Gym official), Pendulum **-200.0**.

**PPO on CartPole-v1** (5 seeds, warm):

| n_envs | Success | STS (samples)   | TTS (s)       | Eval        | Train FPS     |
|-------:|--------:|----------------:|--------------:|------------:|--------------:|
| 8      | 100%    | 16,384 ± 4,670  | **0.89 ± 0.27** | 494.8 ± 8.0 | 18,138 ± 314  |
| 16     | 100%    | 28,672 ± 10,199 | **1.04 ± 0.36** | 495.0 ± 7.6 | 27,317 ± 570  |

**SAC on Pendulum-v1** (5 seeds, warm; `lr=2e-3`, `tau=0.02`, `learning_starts=256`):

| n_envs | Success | STS (samples)    | TTS (s)         | Eval   |
|-------:|--------:|-----------------:|----------------:|-------:|
| 8      | 100%    | 12,288 ± 10,443  | **2.01 ± 1.83** | ≥ −200 |
| 16     | 100%    | 20,480 ± 6,212   | **1.78 ± 0.55** | ≥ −200 |

STS = samples-to-solve, TTS = wall time-to-solve. Compiling the rollout loop (single `mx.compile` unrolled graph) plus RNG key-threading roughly doubled train FPS — Pendulum TTS ~4–7s → ~2s, CartPole ~1.4–1.8s → ~0.9–1.0s.

### Cross-framework baselines

Same tasks against reference implementations on the same machine, reporting normalized **samples/sec** and **time-to-threshold**. These are optional and *not* in `requirements.txt`/`pyproject.toml`; install into a separate venv:

```bash
/opt/homebrew/bin/python3.14 -m venv .baselines-venv
.baselines-venv/bin/pip install -e .
.baselines-venv/bin/pip install mlx mlx-metal psutil numpy \
    stable-baselines3 gymnasium torch jax flax optax distrax gymnax chex
.baselines-venv/bin/python bench/reinforcement/benchmark_baselines.py --seeds 3
```

Configs: **MLX** uses this repo's tuned HPs (`lr=1e-3` PPO / `lr=2e-3`,`tau=0.02` SAC); **SB3** uses its defaults; **purejaxrl** uses its standard PPO config. MLX and purejaxrl run 16 parallel envs; SB3 runs a single env (its typical CPU setup).

**PPO on CartPole-v1** (threshold 475.0, 3 seeds, median):

| Backend   | Success | STS (samples) | TTS (s) | Eval | samples/sec |
|-----------|--------:|--------------:|--------:|-----:|------------:|
| **MLX**   | 100%    | 32,768        | 1.37    | 483  | 26,767      |
| SB3 CPU   | 100%    | 20,480        | 4.76    | 496  | 4,522       |
| SB3 MPS   | 100%    | 22,528        | 51.65   | 495  | 436         |
| purejaxrl | 100%    | 82,304        | 0.48*   | 500  | 171,342     |

**SAC on Pendulum-v1** (threshold −200.0, 3 seeds, median):

| Backend | Success | STS (samples) | TTS (s) | Eval  | samples/sec |
|---------|--------:|--------------:|--------:|------:|------------:|
| **MLX** | 100%    | 20,480        | 1.53    | −171  | 13,367      |
| SB3 CPU | 100%    | 4,096         | 12.59   | −148  | 325         |
| SB3 MPS | 100%    | 4,096         | 66.92   | −148  | 61          |

Notes:

- **SB3 MPS is slower than SB3 CPU.** PyTorch MPS has high per-op overhead on the small `MlpPolicy` tensors these tasks use (a [known SB3 issue](https://github.com/DLR-RM/stable-baselines3/issues/1245)); it is not representative of MPS on larger models.
- **purejaxrl** is PPO-only (no SAC). Its TTS is a *proxy* derived from training returns at 171k samples/sec (`STS / throughput`), not a held-out eval, so the `*` value is approximate. It is the most throughput-efficient but needs ~2–4x more samples to solve CartPole.
- SB3 SAC is more sample-efficient than MLX SAC (solves in ~4k vs ~20k samples) but ~3.5x slower in wall-clock on a single CPU env.

### Scaling crossover vs purejaxrl

Raw training throughput (samples/sec) as `num_envs` grows, where MLX's GPU overtakes JAX-CPU:

| num_envs | MLX samples/sec | purejaxrl (JAX-CPU) | MLX/JAX |
|---------:|----------------:|--------------------:|--------:|
| 16       | 57,562          | 120,008             | 0.48x   |
| 64       | 151,683         | 176,446             | 0.86x   |
| 256      | 213,440         | 199,323             | **1.07x** |
| 1,024    | 256,186         | 183,688             | **1.39x** |
| 4,096    | 252,259         | 182,357             | **1.38x** |

Run with `python bench/reinforcement/benchmark_baselines.py --sweep`. purejaxrl's whole-program `jit` + `lax.scan` wins on tiny 16-env loops (zero per-step dispatch); MLX overtakes once the batch is GPU-sized (≥256 envs), where Metal's parallelism amortizes the fixed per-kernel launch cost. JAX-CPU saturates near ~180–200k samples/sec, while MLX keeps scaling with batch width.

### Scaling (secondary — systems microbench)

Pure rollout / train throughput (not time-to-competence). `n_envs` **doubles** each row. Sweeps stop on swap growth (>256 MB) or train-FPS plateau. Train FPS warms the compiled path and flushes the lazy graph before timing.

**PPO on CartPole-v1** (stop: train FPS plateau at 8,192):

| Num Envs | Env FPS     | Train FPS  | Memory (MB) |
|---------:|------------:|-----------:|------------:|
| 16       | 28,555      | 73,037     | 116.1       |
| 32       | 63,514      | 136,813    | 120.7       |
| 64       | 92,648      | 239,574    | 123.4       |
| 128      | 197,922     | 418,386    | 125.5       |
| 256      | 408,417     | 468,214    | 128.0       |
| 512      | 1,033,604   | 838,817    | 131.4       |
| 1,024    | 1,896,555   | 1,152,859  | 133.3       |
| 2,048    | 4,861,035   | 1,264,533  | 114.3       |
| 4,096    | 9,459,681   | 1,225,382  | 115.4       |
| 8,192    | 18,291,440  | 1,281,270  | 114.4       |

**SAC on Pendulum-v1** (stop: train FPS plateau at 32,768):

| Num Envs | Env FPS     | Train FPS  | Memory (MB) |
|---------:|------------:|-----------:|------------:|
| 16       | 14,500      | 2,308      | 279.4       |
| 32       | 87,887      | 21,707     | 279.8       |
| 64       | 167,978     | 54,850     | 279.9       |
| 128      | 249,020     | 68,000     | 280.1       |
| 256      | 668,014     | 177,545    | 281.0       |
| 512      | 1,294,874   | 397,801    | 281.0       |
| 1,024    | 2,147,972   | 570,349    | 281.1       |
| 2,048    | 4,892,445   | 1,171,364  | 281.1       |
| 4,096    | 9,869,518   | 1,875,848  | 281.0       |
| 8,192    | 18,060,309  | 2,299,776  | 281.1       |
| 16,384   | 28,192,317  | 2,374,360  | 275.2       |
| 32,768   | 34,121,231  | 1,675,245  | 250.0       |

Scaling FPS = training-loop microbench. TTS table = full early-stop solve path (what you actually care about).


## Why

This started as the RL stack for a personal trading bot on a Mac. PureJaxRL showed how far "keep it on accelerator" can go; this is the dirtier, SB3-flavored cousin for people who live on Apple Silicon and still want PPO/SAC that feel familiar.

## Status

Early and opinionated. APIs may shift. Issues and PRs welcome.

## License

MIT
