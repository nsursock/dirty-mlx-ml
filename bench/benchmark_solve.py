"""Timesteps-to-solve benchmark: double num_envs until 16k (or plateau/swap)."""
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dirty_mlx_ml.reinforcement import PPO, SAC
from dirty_mlx_ml.reinforcement.envs import make

MAX_NUM_ENVS = 16_384
START_NUM_ENVS = 1
# quick slice: set via --envs or default list below
DEFAULT_ENVS = [256, 512, 1024]


def get_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def swap_used_mb() -> float:
    try:
        return psutil.swap_memory().used / (1024 * 1024)
    except Exception:
        return 0.0


def is_swapping(baseline: float, growth_mb: float = 256.0) -> bool:
    return swap_used_mb() - baseline > growth_mb


def eval_mean(model, env_id: str, n_eps: int, max_steps: int, seed: int = 123) -> float:
    env = make(env_id, num_envs=1, seed=seed)
    discrete = hasattr(env.action_space, "n")
    total = 0.0
    for e in range(n_eps):
        obs, _ = env.reset(seed=seed + e)
        done = False
        steps = 0
        ep = 0.0
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            if discrete:
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
) -> Dict:
    env = make(env_id, num_envs=num_envs, seed=seed)
    if algo_name == "PPO":
        # min 32 steps for GAE horizon; cap rollout size for speed
        n_steps = max(32, min(256, 8192 // num_envs))
        bs = max(1, min(64, 2048 // num_envs))
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=bs,
            n_epochs=10,
            learning_rate=3e-4,
            seed=seed,
            log_dir=None,
            verbose=0,
        )
    else:
        # 1 env-step adds n_envs transitions → match with n_envs grad steps
        model = SAC(
            "MlpPolicy",
            env,
            learning_starts=min(10_000, max(1000, num_envs * 2)),
            buffer_size=300_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=max(1, min(num_envs, 64)),
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
                timeouts = (
                    infos.get("timeouts", mx.zeros((model.n_envs,)))
                    if isinstance(infos, dict)
                    else mx.zeros((model.n_envs,))
                )
                model.replay.add(obs, new_obs, action, rewards, dones.astype(mx.float32), timeouts)
                model._last_obs = new_obs
            if model.num_timesteps >= model.learning_starts:
                gs = model.gradient_steps if model.gradient_steps > 0 else model.train_freq
                model.train(gs, model.batch_size)

        if model.num_timesteps >= next_check:
            last_eval = eval_mean(model, env_id, n_eval_eps, max_ep_steps, seed=seed + 99)
            better = last_eval >= threshold if threshold > 0 else last_eval >= threshold
            # CartPole threshold positive; Pendulum threshold negative (higher is better both)
            if last_eval >= threshold:
                solved_at = model.num_timesteps
                break
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
        print(f"\n{algo_name} | n_envs={num_envs} ...", flush=True)
        r = train_until_solved(
            algo_name,
            env_id,
            num_envs,
            threshold=threshold,
            max_timesteps=max_timesteps,
            eval_every=eval_every,
            n_eval_eps=n_eval_eps,
            max_ep_steps=max_ep_steps,
        )
        results.append(r)
        ts_hist.append(float(r["timesteps"]))
        status = "SOLVED" if r["solved"] else "FAIL"
        print(
            f"  {status}  timesteps={r['timesteps']}  wall={r['wall_s']:.2f}s  "
            f"eval={r['eval_mean']:.1f}  fps={r['train_fps']:.0f}  mem={r['memory_mb']:.0f}MB",
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
    print("-" * 100)
    print(
        f"{'n_envs':>8} {'solved':>7} {'timesteps':>12} {'wall_s':>10} "
        f"{'eval':>10} {'train_fps':>12} {'ts_speedup':>11} {'wall_speedup':>12}"
    )
    print("-" * 100)
    for r in rows:
        ts_sp = base_ts / max(r["timesteps"], 1)
        w_sp = base_wall / max(r["wall_s"], 1e-9)
        print(
            f"{r['num_envs']:>8} {'Y' if r['solved'] else 'N':>7} {r['timesteps']:>12} "
            f"{r['wall_s']:>10.2f} {r['eval_mean']:>10.1f} {r['train_fps']:>12.0f} "
            f"{ts_sp:>10.2f}x {w_sp:>11.2f}x"
        )


def main():
    env_list = DEFAULT_ENVS  # 256, 512, 1024
    all_rows = []

    ppo = run_solve_bench(
        "PPO",
        "CartPole-v1",
        threshold=450.0,
        max_timesteps=400_000,
        eval_every=32_768,
        n_eval_eps=5,
        max_ep_steps=500,
        env_list=env_list,
    )
    all_rows.append(("PPO", "CartPole-v1", ppo))

    sac = run_solve_bench(
        "SAC",
        "Pendulum-v1",
        threshold=-300.0,
        max_timesteps=200_000,
        eval_every=16_384,
        n_eval_eps=5,
        max_ep_steps=200,
        env_list=env_list,
    )
    all_rows.append(("SAC", "Pendulum-v1", sac))

    print("\n" + "=" * 100)
    print("SOLVE BENCHMARK RESULTS (speedup vs first n_envs row)")
    print("=" * 100)
    for algo, env, rows in all_rows:
        print_table(algo, env, rows)


if __name__ == "__main__":
    main()

