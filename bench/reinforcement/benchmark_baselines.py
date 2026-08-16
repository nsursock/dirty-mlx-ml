"""Cross-framework RL baselines on the same Mac.

Compares this repo's MLX PPO/SAC against reference implementations on identical
tasks, reporting normalized metrics:

  * ``samples/sec``  -- effective training throughput (samples-to-solve / wall)
  * ``TTS``          -- wall-clock time-to-threshold (early-stopped on eval)
  * ``STS``          -- samples-to-solve (total env steps consumed)

Backends
--------
  * ``mlx``        -- this repo (Apple Silicon / Metal)
  * ``sb3-cpu``    -- Stable-Baselines3, ``device="cpu"``
  * ``sb3-mps``    -- Stable-Baselines3, ``device="mps"`` (the PyTorch MPS baseline)
  * ``purejaxrl``  -- vendored JAX PPO (CartPole only; purejaxrl has no SAC)

Tasks
-----
  * ``cartpole-ppo``  -- CartPole-v1, threshold 475.0 (Gym official)
  * ``pendulum-sac``  -- Pendulum-v1, threshold -200.0

These baselines are optional and deliberately NOT in ``requirements.txt`` or
``pyproject.toml``. Install them into a separate venv (Python >= 3.11):

    /opt/homebrew/bin/python3.14 -m venv .baselines-venv
    .baselines-venv/bin/pip install -e .
    .baselines-venv/bin/pip install mlx mlx-metal psutil numpy \\
        stable-baselines3 gymnasium torch jax flax optax distrax gymnax chex

Run from the repo root:

    .baselines-venv/bin/python bench/reinforcement/benchmark_baselines.py --seeds 3

Missing backends are skipped with a notice.
"""

import argparse
import os
import sys
import time

import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)  # for benchmark_solve
sys.path.insert(0, os.path.join(BENCH_DIR, "baselines"))  # vendored purejaxrl

from dirty_mlx_ml.reinforcement import PPO, SAC  # noqa: E402
from dirty_mlx_ml.reinforcement.envs import make  # noqa: E402
from benchmark_solve import _make_model, _train_cycle, eval_mean  # noqa: E402

CARTPOLE_THRESHOLD = 475.0
PENDULUM_THRESHOLD = -200.0


def _result(solved, samples, wall, eval_mean_value):
    return {
        "solved": solved,
        "samples": int(samples),
        "wall_s": wall,
        "eval_mean": eval_mean_value,
        "samples_per_sec": samples / max(wall, 1e-9),
    }


# --------------------------------------------------------------------------- #
# MLX backend (this repo)
# --------------------------------------------------------------------------- #
def mlx_run(algo, env_id, threshold, max_timesteps, eval_every, n_eval_eps, max_ep_steps, num_envs, seed):
    env = make(env_id, num_envs=num_envs, seed=seed)
    model = _make_model(algo, env, seed)
    model._last_obs, _ = env.reset(seed=seed)

    t0 = time.time()
    solved_at = None
    last_eval = float("-inf")
    next_check = eval_every
    while model.num_timesteps < max_timesteps:
        _train_cycle(algo, model)
        if model.num_timesteps >= next_check:
            last_eval = eval_mean(model, env_id, n_eval_eps, max_ep_steps, seed=seed + 99)
            if last_eval >= threshold:
                solved_at = model.num_timesteps
                break
            next_check = model.num_timesteps + eval_every
    wall = time.time() - t0
    if solved_at is None:
        last_eval = eval_mean(model, env_id, n_eval_eps, max_ep_steps, seed=seed + 99)
    env.close()
    samples = solved_at if solved_at is not None else model.num_timesteps
    return _result(solved_at is not None, samples, wall, last_eval)


# --------------------------------------------------------------------------- #
# Stable-Baselines3 backend (CPU or MPS)
# --------------------------------------------------------------------------- #
def sb3_run(algo, env_id, threshold, max_timesteps, eval_every, n_eval_eps, max_ep_steps, device, seed):
    from stable_baselines3 import PPO as SB3PPO, SAC as SB3SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

    train_env = make_vec_env(env_id, n_envs=1, seed=seed)
    eval_env = make_vec_env(env_id, n_envs=n_eval_eps, seed=seed + 999)

    stop = StopTrainingOnRewardThreshold(reward_threshold=threshold, verbose=0)
    callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=eval_every,
        n_eval_episodes=n_eval_eps,
        deterministic=True,
        render=False,
        verbose=0,
        callback_after_eval=stop,
    )

    if algo == "PPO":
        model = SB3PPO(
            "MlpPolicy", train_env, learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, seed=seed, device=device, verbose=0,
        )
    else:
        model = SB3SAC(
            "MlpPolicy", train_env, learning_rate=3e-4, learning_starts=100,
            buffer_size=1_000_000, batch_size=256, seed=seed, device=device, verbose=0,
        )

    t0 = time.time()
    model.learn(total_timesteps=max_timesteps, callback=callback)
    wall = time.time() - t0

    last_eval = callback.last_mean_reward if callback.last_mean_reward is not None else float("-inf")
    solved = last_eval >= threshold
    samples = model.num_timesteps
    return _result(solved, samples, wall, last_eval)


# --------------------------------------------------------------------------- #
# purejaxrl backend (JAX, CartPole PPO only)
# --------------------------------------------------------------------------- #
def purejaxrl_run(threshold, max_timesteps, num_envs, seed):
    import jax
    import purejaxrl_ppo as pjr

    config = {
        "LR": 3e-4, "NUM_ENVS": num_envs, "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": max_timesteps,
        "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99, "GAE_LAMBDA": 0.95, "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh", "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": True, "DEBUG": False,
    }
    train_jit = jax.jit(pjr.make_train(config))
    rng = jax.random.PRNGKey(seed)
    t0 = time.time()
    out = train_jit(rng)
    rets = np.asarray(out["metrics"]["returned_episode_returns"])
    rets.flags.writeable = False
    np.asarray(rets).sum()  # force a device->host sync
    wall = time.time() - t0

    ts = np.asarray(out["metrics"]["timestep"])
    throughput = max_timesteps / max(wall, 1e-9)

    # TTS proxy: first per-env timestep at which a returned episode crossed the
    # threshold. Global samples = per-env timestep * num_envs (no held-out eval).
    solved_ts = np.where(rets >= threshold, ts, np.inf)
    first = float(solved_ts.min())
    if np.isfinite(first):
        solved = True
        samples = int(first * num_envs)
        tts = samples / throughput
    else:
        solved = False
        samples = int(max_timesteps)
        tts = wall

    return _result(solved, samples, tts, float(rets.max()))


# --------------------------------------------------------------------------- #
# Runner registry + aggregation
# --------------------------------------------------------------------------- #
def aggregate(seed_results):
    if not seed_results:
        return {}
    solved = sum(1 for r in seed_results if r["solved"])
    def med(xs):
        return float(np.median(xs)) if xs else 0.0

    return {
        "solve_rate": solved / len(seed_results),
        "samples_median": med([r["samples"] for r in seed_results]),
        "wall_s_median": med([r["wall_s"] for r in seed_results]),
        "eval_mean_median": med([r["eval_mean"] for r in seed_results]),
        "samples_per_sec_median": med([r["samples_per_sec"] for r in seed_results]),
    }


TASKS = {
    "cartpole-ppo": {
        "env_id": "CartPole-v1", "algo": "PPO", "threshold": CARTPOLE_THRESHOLD,
        "max_timesteps": 100_000, "eval_every": 2_048, "n_eval_eps": 5, "max_ep_steps": 500,
        "num_envs": 16, "purejaxrl_max_timesteps": 500_000,
    },
    "pendulum-sac": {
        "env_id": "Pendulum-v1", "algo": "SAC", "threshold": PENDULUM_THRESHOLD,
        "max_timesteps": 100_000, "eval_every": 4_096, "n_eval_eps": 5, "max_ep_steps": 200,
        "num_envs": 16,
    },
}


def run_task(task_id, backend, device, seed):
    task = TASKS[task_id]
    if backend == "mlx":
        return mlx_run(
            task["algo"], task["env_id"], task["threshold"], task["max_timesteps"],
            task["eval_every"], task["n_eval_eps"], task["max_ep_steps"],
            task["num_envs"], seed,
        )
    if backend in ("sb3-cpu", "sb3-mps"):
        dev = "cpu" if backend == "sb3-cpu" else "mps"
        return sb3_run(
            task["algo"], task["env_id"], task["threshold"], task["max_timesteps"],
            task["eval_every"], task["n_eval_eps"], task["max_ep_steps"], dev, seed,
        )
    if backend == "purejaxrl":
        if task_id != "cartpole-ppo":
            raise ValueError("purejaxrl only supports cartpole-ppo")
        return purejaxrl_run(task["threshold"], task["purejaxrl_max_timesteps"], task["num_envs"], seed)
    raise ValueError(f"unknown backend: {backend}")


def _throughput_mlx(num_envs, n_cycles=5):
    env = make("CartPole-v1", num_envs=num_envs, seed=0)
    model = PPO(
        "MlpPolicy", env, learning_rate=1e-3, n_steps=256, batch_size=128,
        n_epochs=20, seed=0, policy_kwargs={"net_arch": [64, 64]},
    )
    model._last_obs, _ = env.reset(seed=0)
    model.collect_rollouts()
    model.train()
    t0 = time.time()
    for _ in range(n_cycles):
        model.collect_rollouts()
        model.train()
    wall = time.time() - t0
    samples = model.n_steps * num_envs * n_cycles
    env.close()
    return samples / max(wall, 1e-9)


def _throughput_pjr(num_envs, num_updates=8):
    import jax
    import purejaxrl_ppo as pjr

    num_steps = 256
    total = num_steps * num_envs * num_updates
    config = {
        "LR": 3e-4, "NUM_ENVS": num_envs, "NUM_STEPS": num_steps,
        "TOTAL_TIMESTEPS": total, "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99, "GAE_LAMBDA": 0.95, "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh", "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": True, "DEBUG": False,
    }
    train_jit = jax.jit(pjr.make_train(config))
    rng = jax.random.PRNGKey(0)
    train_jit(rng)  # warmup / compile
    rng = jax.random.PRNGKey(0)
    t0 = time.time()
    out = train_jit(rng)
    np.asarray(out["metrics"]["returned_episode_returns"]).sum()  # host sync
    wall = time.time() - t0
    return total / max(wall, 1e-9)


def run_sweep(env_list):
    print("\n" + "=" * 96)
    print("CartPole PPO — training throughput vs num_envs (MLX vs purejaxrl/JAX-CPU)")
    print("=" * 96)
    print(f"{'num_envs':>10} {'MLX samples/sec':>18} {'purejaxrl samples/sec':>22} {'MLX/JAX':>9}")
    print("-" * 96)
    for n in env_list:
        mlx_sps = pjr_sps = None
        try:
            mlx_sps = _throughput_mlx(n)
        except Exception as e:
            print(f"{n:>10}  MLX ERROR: {type(e).__name__}: {e}")
        try:
            pjr_sps = _throughput_pjr(n)
        except Exception as e:
            print(f"{n:>10}  purejaxrl ERROR: {type(e).__name__}: {e}")
        if mlx_sps and pjr_sps:
            ratio = mlx_sps / pjr_sps
            print(
                f"{n:>10} {mlx_sps:>18,.0f} {pjr_sps:>22,.0f} {ratio:>9.2f}x",
                flush=True,
            )
        else:
            print(f"{n:>10} {'—':>18} {'—':>22} {'—':>9}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Cross-framework RL baselines")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument(
        "--backends", nargs="+", default=["mlx", "sb3-cpu", "sb3-mps", "purejaxrl"],
        choices=["mlx", "sb3-cpu", "sb3-mps", "purejaxrl"],
    )
    parser.add_argument("--sweep", action="store_true", help="Run num_envs throughput sweep")
    parser.add_argument("--sweep-envs", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args.sweep_envs)
        return

    rows = []
    for task_id in args.tasks:
        task = TASKS[task_id]
        print(f"\n{'='*90}\n{task_id}  (threshold={task['threshold']}, max_steps={task['max_timesteps']})")
        for backend in args.backends:
            if backend == "purejaxrl" and task_id != "cartpole-ppo":
                continue
            print(f"\n  backend={backend} ...", flush=True)
            seed_results = []
            for i in range(args.seeds):
                seed = i * 1000
                try:
                    r = run_task(task_id, backend, None, seed)
                except ImportError as e:
                    print(f"    SKIP: missing dependency ({e})")
                    seed_results = []
                    break
                except Exception as e:
                    print(f"    seed={seed} ERROR: {type(e).__name__}: {e}")
                    continue
                status = "SOLVED" if r["solved"] else "FAIL"
                print(
                    f"    seed={seed} {status}  STS={r['samples']}  TTS={r['wall_s']:.2f}s  "
                    f"eval={r['eval_mean']:.1f}  {r['samples_per_sec']:.0f} samples/sec",
                    flush=True,
                )
                seed_results.append(r)
            if not seed_results:
                continue
            agg = aggregate(seed_results)
            agg["task"] = task_id
            agg["backend"] = backend
            rows.append(agg)

    print(f"\n{'='*90}\nBASELINE SUMMARY (median over seeds)\n{'='*90}")
    hdr = f"{'task':>14} {'backend':>10} {'success':>8} {'STS':>12} {'TTS_s':>9} {'eval':>10} {'samples/sec':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['task']:>14} {r['backend']:>10} {r['solve_rate']:>7.0%} "
            f"{r['samples_median']:>12.0f} {r['wall_s_median']:>9.2f} "
            f"{r['eval_mean_median']:>10.1f} {r['samples_per_sec_median']:>12.0f}"
        )


if __name__ == "__main__":
    main()
