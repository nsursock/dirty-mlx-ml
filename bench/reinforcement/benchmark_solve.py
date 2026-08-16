"""Time-to-competence benchmark: multi-seed median TTS with cheap vectorized eval."""
import os
import sys
import time
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dirty_mlx_ml.reinforcement import PPO, SAC
from dirty_mlx_ml.reinforcement.envs import make

MAX_NUM_ENVS = 16_384
START_NUM_ENVS = 1
DEFAULT_ENVS = [8, 16]


def get_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_memory_mb_median(samples: int = 5, delay: float = 0.1) -> float:
    memory_samples = []
    for _ in range(samples):
        memory_samples.append(get_memory_mb())
        if delay > 0:
            time.sleep(delay)
    memory_samples.sort()
    return memory_samples[len(memory_samples) // 2]


def swap_used_mb() -> float:
    try:
        return psutil.swap_memory().used / (1024 * 1024)
    except Exception:
        return 0.0


def is_swapping(baseline: float, growth_mb: float = 256.0) -> bool:
    return swap_used_mb() - baseline > growth_mb


def aggregate_seed_results(seed_results: List[Dict]) -> Dict:
    if not seed_results:
        return {}

    timesteps = [r["timesteps"] for r in seed_results if r["timesteps"] is not None]
    wall_s = [r["wall_s"] for r in seed_results]
    eval_means = [r["eval_mean"] for r in seed_results]
    train_fps = [r["train_fps"] for r in seed_results]
    memory_mb = [r["memory_mb"] for r in seed_results]
    solved_count = sum(1 for r in seed_results if r["solved"])

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
        "solved": solved_count > 0,
        "timesteps": ts_stats["median"],
        "wall_s": wall_stats["median"],
        "eval_mean": eval_stats["median"],
        "train_fps": fps_stats["median"],
        "memory_mb": mem_stats["median"],
    }


def _policy_action(model, obs, deterministic: bool = True):
    """Batched action without host sync (no mx.eval)."""
    if hasattr(model, "policy"):
        action, _, _ = model.policy.get_action(obs, deterministic=deterministic)
        if model.discrete:
            return action
        return mx.clip(action, model.env.action_space.low, model.env.action_space.high)
    a_tanh, _ = model.actor.sample(obs, deterministic=deterministic)
    return model._scale_action(a_tanh)


def eval_mean(model, env_id: str, n_eps: int, max_steps: int, seed: int = 123) -> float:
    """Vectorized first-episode mean return; one host sync at the end."""
    n_eps = max(int(n_eps), 1)
    env = make(env_id, num_envs=n_eps, seed=seed)
    obs, _ = env.reset(seed=seed)
    discrete = env.action_space.__class__.__name__ == "Discrete"
    ep_ret = mx.zeros((n_eps,), dtype=mx.float32)
    alive = mx.ones((n_eps,), dtype=mx.float32)

    for _ in range(max_steps):
        action = _policy_action(model, obs, deterministic=True)
        if discrete:
            action = mx.reshape(action, (-1,))
        obs, rew, done, _ = env.step(action)
        d = done.astype(mx.float32)
        ep_ret = ep_ret + rew * alive
        alive = alive * (1.0 - d)

    mx.eval(ep_ret, alive)
    env.close()
    return float(mx.mean(ep_ret).item())


def _make_model(algo_name: str, env, seed: int):
    if algo_name == "PPO":
        return PPO(
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
    return SAC(
        "MlpPolicy",
        env,
        learning_starts=500,
        buffer_size=50_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        seed=seed,
        log_dir=None,
        verbose=0,
        policy_kwargs={"net_arch": [256, 256]},
    )


def _train_cycle(algo_name: str, model):
    if algo_name == "PPO":
        model.collect_rollouts()
        model.train()
        return
    for _ in range(model.train_freq):
        obs = model._last_obs
        action = model._sample_action(obs, random=model.num_timesteps < model.learning_starts)
        new_obs, rewards, dones, infos = model.env.step(action)
        model.num_timesteps += model.n_envs
        model._update_ep_stats(rewards, dones)
        if isinstance(dones, tuple) and len(dones) == 2:
            terminated, truncated = dones
        else:
            terminated = dones
            truncated = mx.zeros((model.n_envs,))
        if isinstance(infos, dict) and "timeouts" in infos:
            truncated = mx.maximum(truncated, infos["timeouts"])
        model.replay.add(
            obs,
            new_obs,
            action,
            rewards,
            terminated.astype(mx.float32),
            truncated.astype(mx.float32),
        )
        model._last_obs = new_obs
    if model.num_timesteps >= model.learning_starts:
        gs = model.gradient_steps if model.gradient_steps > 0 else model.train_freq
        model.train(gs, model.batch_size)


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
    consecutive_evals: int = 1,
    n_confirm_eps: int = 10,
    warm: bool = False,
) -> Dict:
    """
    Time-to-competence: train until probe+confirm hit threshold (early stop).

    Protocol:
      - every eval_every steps: cheap vectorized probe (n_eval_eps)
      - if probe >= threshold: one confirmation eval (n_confirm_eps)
      - if confirm >= threshold: stop (counts as consecutive_evals=1 pass)
      - optional warm: one train cycle before the timer (JIT outside TTS)
    """
    env = make(env_id, num_envs=num_envs, seed=seed)
    model = _make_model(algo_name, env, seed)
    model._last_obs, _ = env.reset(seed=seed)

    if warm:
        _train_cycle(algo_name, model)
        # reset progress so warm cycle is not counted in TTS samples
        model.num_timesteps = 0
        model._n_updates = 0
        model._last_obs, _ = env.reset(seed=seed)
        if algo_name == "PPO":
            model._last_episode_starts = mx.ones((model.n_envs,))
            model.buffer.reset()
        else:
            # keep network weights warm; clear replay so learning_starts still applies
            model.replay.pos = 0
            model.replay.full = False

    t0 = time.time()
    solved_at: Optional[int] = None
    last_eval = float("-inf")
    next_check = eval_every
    consecutive_above = 0

    while model.num_timesteps < max_timesteps:
        _train_cycle(algo_name, model)

        if model.num_timesteps >= next_check:
            probe = eval_mean(model, env_id, n_eval_eps, max_ep_steps, seed=seed + 99)
            last_eval = probe
            if probe >= threshold:
                confirm = eval_mean(
                    model, env_id, max(n_confirm_eps, n_eval_eps), max_ep_steps, seed=seed + 199
                )
                last_eval = confirm
                if confirm >= threshold:
                    consecutive_above += 1
                    if consecutive_above >= consecutive_evals:
                        solved_at = model.num_timesteps
                        break
                else:
                    consecutive_above = 0
            else:
                consecutive_above = 0
            next_check = model.num_timesteps + eval_every

    wall = time.time() - t0
    mem = get_memory_mb_median(samples=3, delay=0.05)
    if solved_at is None:
        last_eval = eval_mean(
            model, env_id, max(n_confirm_eps, n_eval_eps), max_ep_steps, seed=seed + 99
        )
        if last_eval >= threshold:
            solved_at = model.num_timesteps

    env.close()
    ts = solved_at if solved_at is not None else model.num_timesteps
    return {
        "num_envs": num_envs,
        "solved": solved_at is not None,
        "timesteps": ts,
        "wall_s": wall,
        "eval_mean": last_eval,
        "memory_mb": mem,
        "train_fps": ts / max(wall, 1e-9),
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
    n_seeds: int = 5,
    consecutive_evals: int = 1,
    n_confirm_eps: int = 10,
    warm: bool = False,
) -> List[Dict]:
    results = []
    baseline_swap = swap_used_mb()

    if env_list is None:
        env_list = []
        n = start_num_envs
        while n <= max_num_envs:
            env_list.append(n)
            n *= 2

    print(f"\n{'='*72}\n{algo_name} time-to-competence on {env_id} (threshold={threshold})")
    print(
        f"n_envs={env_list}  n_seeds={n_seeds}  eval_every={eval_every}  "
        f"probe_eps={n_eval_eps}  confirm_eps={n_confirm_eps}  warm={warm}"
    )
    print(f"{'='*72}")

    for num_envs in env_list:
        print(f"\n{algo_name} | n_envs={num_envs} (n_seeds={n_seeds}) ...", flush=True)
        seed_results = []
        for seed_idx in range(n_seeds):
            seed = seed_idx * 1000
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
                n_confirm_eps=n_confirm_eps,
                warm=warm,
            )
            seed_results.append(r)
            status = "SOLVED" if r["solved"] else "FAIL"
            print(
                f"    {status}  STS={r['timesteps']}  TTS={r['wall_s']:.2f}s  "
                f"eval={r['eval_mean']:.1f}  fps={r['train_fps']:.0f}  mem={r['memory_mb']:.0f}MB",
                flush=True,
            )

        aggregated = aggregate_seed_results(seed_results)
        results.append(aggregated)
        print(
            f"  AGGREGATED: success={aggregated['solve_rate']:.0%} | "
            f"STS={aggregated['timesteps_median']:.0f} ± {aggregated['timesteps_std']:.0f} | "
            f"TTS={aggregated['wall_s_median']:.2f} ± {aggregated['wall_s_std']:.2f}s | "
            f"eval={aggregated['eval_mean_median']:.1f} ± {aggregated['eval_mean_std']:.1f} | "
            f"fps={aggregated['train_fps_median']:.0f} ± {aggregated['train_fps_std']:.0f}",
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
    print(f"\n{algo} on {env} — Time-to-Competence (median ± std)")
    print("-" * 120)
    print(
        f"{'n_envs':>8} {'success':>8} {'STS':>20} {'TTS_s':>15} "
        f"{'eval':>12} {'train_fps':>15} {'ts_speedup':>11} {'wall_speedup':>12}"
    )
    print("-" * 120)
    for r in rows:
        ts_sp = base_ts / max(r["timesteps"], 1)
        w_sp = base_wall / max(r["wall_s"], 1e-9)
        if "timesteps_std" in r and r["timesteps_std"] > 0:
            ts_str = f"{r['timesteps_median']:.0f} ± {r['timesteps_std']:.0f}"
        else:
            ts_str = f"{r['timesteps']:.0f}"
        if "wall_s_std" in r and r["wall_s_std"] > 0:
            wall_str = f"{r['wall_s_median']:.2f} ± {r['wall_s_std']:.2f}"
        else:
            wall_str = f"{r['wall_s']:.2f}"
        if "eval_mean_std" in r and r["eval_mean_std"] > 0:
            eval_str = f"{r['eval_mean_median']:.1f} ± {r['eval_mean_std']:.1f}"
        else:
            eval_str = f"{r['eval_mean']:.1f}"
        if "train_fps_std" in r and r["train_fps_std"] > 0:
            fps_str = f"{r['train_fps_median']:.0f} ± {r['train_fps_std']:.0f}"
        else:
            fps_str = f"{r['train_fps']:.0f}"
        solve_str = f"{r['solve_rate']:.0%}" if "solve_rate" in r else ("Y" if r["solved"] else "N")
        print(
            f"{r['num_envs']:>8} {solve_str:>8} {ts_str:>20} "
            f"{wall_str:>15} {eval_str:>12} {fps_str:>15} "
            f"{ts_sp:>10.2f}x {w_sp:>11.2f}x"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-seed time-to-competence benchmark")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds for median±std (default: 5)")
    parser.add_argument("--envs", type=int, nargs="+", default=None, help="num_envs list")
    parser.add_argument(
        "--consecutive-evals",
        type=int,
        default=1,
        help="Confirm passes required after probe (default: 1)",
    )
    parser.add_argument(
        "--warm",
        action="store_true",
        help="JIT-warmup one train cycle outside the TTS timer",
    )
    parser.add_argument("--algo", choices=["ppo", "sac", "both"], default="both")
    args = parser.parse_args()

    env_list = args.envs if args.envs else DEFAULT_ENVS
    all_rows = []

    if args.algo in ("ppo", "both"):
        ppo = run_solve_bench(
            "PPO",
            "CartPole-v1",
            threshold=475.0,
            max_timesteps=160_000,
            eval_every=4_096,
            n_eval_eps=5,
            max_ep_steps=500,
            env_list=env_list,
            n_seeds=args.seeds,
            consecutive_evals=args.consecutive_evals,
            n_confirm_eps=10,
            warm=args.warm,
        )
        all_rows.append(("PPO", "CartPole-v1", ppo))

    if args.algo in ("sac", "both"):
        sac = run_solve_bench(
            "SAC",
            "Pendulum-v1",
            threshold=-200.0,
            max_timesteps=130_000,
            eval_every=8_192,
            n_eval_eps=5,
            max_ep_steps=200,
            env_list=env_list,
            n_seeds=args.seeds,
            consecutive_evals=args.consecutive_evals,
            n_confirm_eps=10,
            warm=args.warm,
        )
        all_rows.append(("SAC", "Pendulum-v1", sac))

    print("\n" + "=" * 120)
    print(
        f"TIME-TO-COMPETENCE (n_seeds={args.seeds}, consecutive_evals={args.consecutive_evals}, "
        f"warm={args.warm})"
    )
    print("STS = samples-to-solve, TTS = wall-clock time-to-solve. Median ± std across seeds.")
    print("=" * 120)
    for algo, env, rows in all_rows:
        print_table(algo, env, rows)


if __name__ == "__main__":
    main()
