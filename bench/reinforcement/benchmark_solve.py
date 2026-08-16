"""Timesteps-to-solve benchmark: double num_envs until 16k (or plateau/swap)."""
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dirty_mlx_ml.reinforcement import PPO, SAC
from dirty_mlx_ml.reinforcement.envs import make

MAX_NUM_ENVS = 16_384
START_NUM_ENVS = 1
# quick slice: set via --envs or default list below
DEFAULT_ENVS = [8, 16]


def get_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def swap_used_mb() -> float:
    try:
        return psutil.swap_memory().used / (1024 * 1024)
    except Exception:
        return 0.0


def is_swapping(baseline: float, growth_mb: float = 256.0) -> bool:
    return swap_used_mb() - baseline > growth_mb


def aggregate_seed_results(seed_results: List[Dict]) -> Dict:
    """Aggregate results across multiple seeds using median ± std."""
    if not seed_results:
        return {}
    
    # Extract arrays for each metric
    timesteps = [r["timesteps"] for r in seed_results if r["timesteps"] is not None]
    wall_s = [r["wall_s"] for r in seed_results]
    eval_means = [r["eval_mean"] for r in seed_results]
    train_fps = [r["train_fps"] for r in seed_results]
    memory_mb = [r["memory_mb"] for r in seed_results]
    solved_count = sum(1 for r in seed_results if r["solved"])
    
    # Compute statistics
    def compute_stats(values):
        if not values:
            return {"median": 0.0, "std": 0.0}
        return {
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    
    ts_stats = compute_stats(timesteps)
    wall_stats = compute_stats(wall_s)
    eval_stats = compute_stats(eval_means)
    fps_stats = compute_stats(train_fps)
    mem_stats = compute_stats(memory_mb)
    
    return {
        "num_envs": seed_results[0]["num_envs"],
        "solve_rate": solved_count / len(seed_results),
        "timesteps_median": ts_stats["median"],
        "timesteps_std": ts_stats["std"],
        "wall_s_median": wall_stats["median"],
        "wall_s_std": wall_stats["std"],
        "eval_mean_median": eval_stats["median"],
        "eval_mean_std": eval_stats["std"],
        "train_fps_median": fps_stats["median"],
        "train_fps_std": fps_stats["std"],
        "memory_mb_median": mem_stats["median"],
        "memory_mb_std": mem_stats["std"],
        # Keep original fields for compatibility
        "solved": solved_count > 0,  # Consider solved if at least one seed solved
        "timesteps": ts_stats["median"],  # Use median as representative
        "wall_s": wall_stats["median"],
        "eval_mean": eval_stats["median"],
        "train_fps": fps_stats["median"],
        "memory_mb": mem_stats["median"],
    }


def eval_mean(model, env_id: str, n_eps: int, max_steps: int, seed: int = 123) -> float:
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
    env.close()
    return total / n_eps


def train_until_solved(
    algo_name: str,
    env_id: str,
    num_envs: int,
    threshold: float,
    max_timesteps: int,
    eval_every: int,
    n_eval_eps: int,
    max_ep_steps: int,
    seed: int = 0,
    consecutive_evals: int = 3,  # Number of consecutive evaluations above threshold to count as solved
) -> Dict:
    env = make(env_id, num_envs=num_envs, seed=seed)
    if algo_name == "PPO":
        # Use exact hyperparameters from test_solve.py
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=256,
            batch_size=128,
            n_epochs=20,
            learning_rate=1e-3,
            seed=seed,
            log_dir=None,
            verbose=0,
        )
    else:
        # Use exact hyperparameters from test_solve.py
        model = SAC(
            "MlpPolicy",
            env,
            learning_starts=1000,
            buffer_size=100_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            seed=seed,
            log_dir=None,
            verbose=0,
            policy_kwargs={"net_arch": [256, 256]},
        )




    model._last_obs, _ = env.reset(seed=seed)
    t0 = time.time()
    solved_at: Optional[int] = None
    last_eval = float("-inf") if threshold > 0 else float("inf")
    next_check = eval_every
    consecutive_above_threshold = 0  # Track consecutive evaluations above threshold

    while model.num_timesteps < max_timesteps:
        if algo_name == "PPO":
            model.collect_rollouts()
            model.train()
        else:
            for _ in range(model.train_freq):
                obs = model._last_obs
                action = model._sample_action(obs, random=model.num_timesteps < model.learning_starts)
                new_obs, rewards, dones, infos = model.env.step(action)
                model.num_timesteps += model.n_envs
                model._update_ep_stats(rewards, dones)
                
                # Handle terminated vs truncated
                if isinstance(dones, tuple) and len(dones) == 2:
                    terminated, truncated = dones
                else:
                    terminated = dones
                    truncated = mx.zeros((model.n_envs,))
                
                if isinstance(infos, dict) and "timeouts" in infos:
                    timeouts = infos["timeouts"]
                    truncated = mx.maximum(truncated, timeouts)
                
                model.replay.add(obs, new_obs, action, rewards, terminated.astype(mx.float32), truncated.astype(mx.float32))
                model._last_obs = new_obs
            if model.num_timesteps >= model.learning_starts:
                gs = model.gradient_steps if model.gradient_steps > 0 else model.train_freq
                model.train(gs, model.batch_size)

        if model.num_timesteps >= next_check:
            last_eval = eval_mean(model, env_id, n_eval_eps, max_ep_steps, seed=seed + 99)
            better = last_eval >= threshold if threshold > 0 else last_eval >= threshold
            # CartPole threshold positive; Pendulum threshold negative (higher is better both)
            if last_eval >= threshold:
                consecutive_above_threshold += 1
                if consecutive_above_threshold >= consecutive_evals:
                    solved_at = model.num_timesteps
                    break
            else:
                consecutive_above_threshold = 0  # Reset counter if below threshold
            next_check = model.num_timesteps + eval_every

    wall = time.time() - t0
    mem = get_memory_mb()
    if solved_at is None:
        last_eval = eval_mean(model, env_id, n_eval_eps, max_ep_steps, seed=seed + 99)
        if last_eval >= threshold:
            solved_at = model.num_timesteps

    env.close()
    return {
        "num_envs": num_envs,
        "solved": solved_at is not None,
        "timesteps": solved_at if solved_at is not None else model.num_timesteps,
        "wall_s": wall,
        "eval_mean": last_eval,
        "memory_mb": mem,
        "train_fps": model.num_timesteps / max(wall, 1e-9),
    }


def run_solve_bench(
    algo_name: str,
    env_id: str,
    threshold: float,
    max_timesteps: int,
    eval_every: int,
    n_eval_eps: int,
    max_ep_steps: int,
    env_list: Optional[List[int]] = None,
    start_num_envs: int = START_NUM_ENVS,
    max_num_envs: int = MAX_NUM_ENVS,
    n_seeds: int = 5,  # Number of seeds for statistical significance
    consecutive_evals: int = 3,  # Consecutive evaluations above threshold
) -> List[Dict]:
    results = []
    baseline_swap = swap_used_mb()
    ts_hist: List[float] = []

    if env_list is None:
        env_list = []
        n = start_num_envs
        while n <= max_num_envs:
            env_list.append(n)
            n *= 2

    print(f"\n{'='*72}\n{algo_name} solve bench on {env_id} (threshold={threshold})")
    print(f"n_envs list: {env_list}\n{'='*72}")

    for num_envs in env_list:
        print(f"\n{algo_name} | n_envs={num_envs} (n_seeds={n_seeds}) ...", flush=True)
        
        # Run multiple seeds for statistical significance
        seed_results = []
        for seed_idx in range(n_seeds):
            seed = seed_idx * 1000  # Use different seeds for each run
            print(f"  Seed {seed_idx + 1}/{n_seeds} (seed={seed})...", flush=True)
            
            r = train_until_solved(
                algo_name,
                env_id,
                num_envs,
                threshold=threshold,
                max_timesteps=max_timesteps,
                eval_every=eval_every,
                n_eval_eps=n_eval_eps,
                max_ep_steps=max_ep_steps,
                seed=seed,
                consecutive_evals=consecutive_evals,
            )
            seed_results.append(r)
            status = "SOLVED" if r["solved"] else "FAIL"
            print(
                f"    {status}  timesteps={r['timesteps']}  wall={r['wall_s']:.2f}s  "
                f"eval={r['eval_mean']:.1f}  fps={r['train_fps']:.0f}  mem={r['memory_mb']:.0f}MB",
                flush=True,
            )
        
        # Compute statistics across seeds
        aggregated = aggregate_seed_results(seed_results)
        results.append(aggregated)
        ts_hist.append(float(aggregated["timesteps_median"]))
        
        # Print aggregated results
        solve_rate = aggregated["solve_rate"]
        print(
            f"  AGGREGATED: solve_rate={solve_rate:.1%} | "
            f"timesteps={aggregated['timesteps_median']:.0f} ± {aggregated['timesteps_std']:.0f} | "
            f"wall={aggregated['wall_s_median']:.2f} ± {aggregated['wall_s_std']:.2f}s | "
            f"eval={aggregated['eval_mean_median']:.1f} ± {aggregated['eval_mean_std']:.1f} | "
            f"fps={aggregated['train_fps_median']:.0f} ± {aggregated['train_fps_std']:.0f} | "
            f"mem={aggregated['memory_mb_median']:.0f} ± {aggregated['memory_mb_std']:.0f}MB",
            flush=True,
        )

        if is_swapping(baseline_swap):
            print(f"  STOP: swap grew at {num_envs} envs")
            break

    return results



def print_table(algo: str, env: str, rows: List[Dict]):
    if not rows:
        return
    base_ts = rows[0]["timesteps"] or 1
    base_wall = rows[0]["wall_s"] or 1e-9
    print(f"\n{algo} on {env}")
    print("-" * 120)
    print(
        f"{'n_envs':>8} {'solve_rate':>10} {'timesteps':>20} {'wall_s':>15} "
        f"{'eval':>12} {'train_fps':>15} {'ts_speedup':>11} {'wall_speedup':>12}"
    )
    print("-" * 120)
    for r in rows:
        ts_sp = base_ts / max(r["timesteps"], 1)
        w_sp = base_wall / max(r["wall_s"], 1e-9)
        
        # Format timesteps as median ± std
        if "timesteps_std" in r and r["timesteps_std"] > 0:
            ts_str = f"{r['timesteps_median']:.0f} ± {r['timesteps_std']:.0f}"
        else:
            ts_str = f"{r['timesteps']:.0f}"
        
        # Format wall time as median ± std
        if "wall_s_std" in r and r["wall_s_std"] > 0:
            wall_str = f"{r['wall_s_median']:.2f} ± {r['wall_s_std']:.2f}"
        else:
            wall_str = f"{r['wall_s']:.2f}"
        
        # Format eval as median ± std
        if "eval_mean_std" in r and r["eval_mean_std"] > 0:
            eval_str = f"{r['eval_mean_median']:.1f} ± {r['eval_mean_std']:.1f}"
        else:
            eval_str = f"{r['eval_mean']:.1f}"
        
        # Format FPS as median ± std
        if "train_fps_std" in r and r["train_fps_std"] > 0:
            fps_str = f"{r['train_fps_median']:.0f} ± {r['train_fps_std']:.0f}"
        else:
            fps_str = f"{r['train_fps']:.0f}"
        
        # Format solve rate as percentage
        if "solve_rate" in r:
            solve_str = f"{r['solve_rate']:.0%}"
        else:
            solve_str = "Y" if r['solved'] else "N"
        
        print(
            f"{r['num_envs']:>8} {solve_str:>10} {ts_str:>20} "
            f"{wall_str:>15} {eval_str:>12} {fps_str:>15} "
            f"{ts_sp:>10.2f}x {w_sp:>11.2f}x"
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-seed solve benchmark")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds for statistical significance (default: 5)")
    parser.add_argument("--envs", type=int, nargs="+", default=None, help="List of num_envs to benchmark")
    parser.add_argument("--consecutive-evals", type=int, default=3, 
                       help="Number of consecutive evals above threshold to count as solved (default: 3)")
    args = parser.parse_args()
    
    env_list = args.envs if args.envs else DEFAULT_ENVS
    all_rows = []

    ppo = run_solve_bench(
        "PPO",
        "CartPole-v1",
        threshold=440.0,  # From test_solve.py
        max_timesteps=160_000,  # From test_solve.py
        eval_every=32_768,
        n_eval_eps=20,  # From test_solve.py
        max_ep_steps=500,  # From test_solve.py
        env_list=env_list,
        n_seeds=args.seeds,
        consecutive_evals=args.consecutive_evals,
    )
    all_rows.append(("PPO", "CartPole-v1", ppo))

    sac = run_solve_bench(
        "SAC",
        "Pendulum-v1",
        threshold=-200.0,  # From test_solve.py
        max_timesteps=100_000,  # From test_solve.py
        eval_every=16_384,
        n_eval_eps=10,  # From test_solve.py
        max_ep_steps=200,  # From test_solve.py
        env_list=env_list,
        n_seeds=args.seeds,
        consecutive_evals=args.consecutive_evals,
    )
    all_rows.append(("SAC", "Pendulum-v1", sac))

    print("\n" + "=" * 120)
    print(f"SOLVE BENCHMARK RESULTS (n_seeds={args.seeds}, consecutive_evals={args.consecutive_evals})")
    print("Results show median ± std across seeds. Speedup vs first n_envs row.")
    print("=" * 120)
    for algo, env, rows in all_rows:
        print_table(algo, env, rows)


if __name__ == "__main__":
    main()

