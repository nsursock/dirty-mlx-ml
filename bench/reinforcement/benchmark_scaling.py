import time
import psutil
import os
from typing import Dict, List
import mlx.core as mx

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dirty_mlx_ml.reinforcement import PPO, SAC
from dirty_mlx_ml.reinforcement.envs import make



def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def swap_used_mb() -> float:
    try:
        return psutil.swap_memory().used / (1024 * 1024)
    except Exception:
        return 0.0


def is_swapping(baseline_swap_mb: float, growth_mb: float = 256.0) -> bool:
    """True if swap grew by growth_mb since baseline (real pressure)."""
    return swap_used_mb() - baseline_swap_mb > growth_mb




def measure_env_fps(env, num_steps: int = 1000) -> float:
    obs, _ = env.reset()
    start = time.time()
    for _ in range(num_steps):
        if hasattr(env.action_space, 'n'):
            action = mx.random.randint(0, env.action_space.n, (env.num_envs,))
        else:
            action = mx.random.uniform(
                env.action_space.low[0], 
                env.action_space.high[0], 
                (env.num_envs, env.action_space.shape[0])
            )
        obs, reward, _, _ = env.step(action)
        mx.eval(obs, reward)  # Force actual computation instead of lazy graph building
    elapsed = time.time() - start
    total_steps = num_steps * env.num_envs
    return total_steps / elapsed


def measure_train_fps(algo, target_timesteps: int = 8192) -> float:
    """Wall-clock train FPS for a fixed number of env timesteps (fair across n_envs)."""
    if algo._last_obs is None:
        algo._last_obs, _ = algo.env.reset()
    start_timesteps = algo.num_timesteps
    start = time.time()
    while algo.num_timesteps - start_timesteps < target_timesteps:
        if hasattr(algo, "collect_rollouts"):
            algo.collect_rollouts()
            algo.train()
        else:
            for _ in range(algo.train_freq):
                obs = algo._last_obs
                action = algo._sample_action(obs, random=algo.num_timesteps < algo.learning_starts)
                new_obs, rewards, dones, infos = algo.env.step(action)
                algo.num_timesteps += algo.n_envs
                algo._update_ep_stats(rewards, dones)
                timeouts = (
                    infos.get("timeouts", mx.zeros((algo.n_envs,)))
                    if isinstance(infos, dict)
                    else mx.zeros((algo.n_envs,))
                )
                algo.replay.add(obs, new_obs, action, rewards, dones.astype(mx.float32), timeouts)
                algo._last_obs = new_obs
            if algo.num_timesteps >= algo.learning_starts:
                algo.train(algo.gradient_steps, algo.batch_size)
    elapsed = max(time.time() - start, 1e-9)
    total_steps = algo.num_timesteps - start_timesteps
    return total_steps / elapsed



def detect_plateau(fps_history: List[float], window: int = 3, threshold: float = 0.05) -> bool:
    if len(fps_history) < window + 1:
        return False
    recent = fps_history[-window:]
    prev = fps_history[-window-1:-1]
    avg_recent = sum(recent) / len(recent)
    avg_prev = sum(prev) / len(prev)
    return abs(avg_recent - avg_prev) / avg_prev < threshold


def benchmark_algorithm(
    algo_class,
    algo_name: str,
    env_id: str,
    start_num_envs: int = 1,
    max_num_envs: int = 1_000_000,
    warmup_steps: int = 100,
    measure_steps: int = 500,
) -> Dict:
    results = {
        'algorithm': algo_name,
        'env': env_id,
        'num_envs': [],
        'env_fps': [],
        'train_fps': [],
        'memory_mb': [],
    }
    
    num_envs = start_num_envs
    env_fps_history = []
    train_fps_history = []
    baseline_swap = swap_used_mb()

    while num_envs <= max_num_envs:
        print(f"\nBenchmarking {algo_name} on {env_id} with {num_envs} envs...")
        
        env = make(env_id, num_envs=num_envs)
        
        if algo_name == 'PPO':
            algo = PPO(
                policy='MlpPolicy',
                env=env,
                learning_rate=3e-4,
                n_steps=128,
                batch_size=64,
                n_epochs=4,
                verbose=0,
            )
        else:
            algo = SAC(
                policy='MlpPolicy',
                env=env,
                learning_rate=3e-4,
                batch_size=64,
                learning_starts=50,
                train_freq=1,
                gradient_steps=1,
                verbose=0,
            )
        
        mx.eval(algo.policy.parameters() if algo_name == 'PPO' else algo.actor.parameters())
        
        # Warmup
        for _ in range(warmup_steps):
            if hasattr(env.action_space, 'n'):
                action = mx.random.randint(0, env.action_space.n, (num_envs,))
            else:
                action = mx.random.uniform(
                    env.action_space.low[0], 
                    env.action_space.high[0], 
                    (num_envs, env.action_space.shape[0])
                )
            env.step(action)
        
        # Measure env FPS
        env_fps = measure_env_fps(env, measure_steps)
        env_fps_history.append(env_fps)

        if algo._last_obs is None:
            algo._last_obs, _ = env.reset()

        # one PPO rollout ≈ n_steps*n_envs; cap measure so huge n_envs still finish
        train_fps = measure_train_fps(algo, target_timesteps=max(8192, min(num_envs * 128, 262_144)))



        train_fps_history.append(train_fps)
        
        # Measure memory
        memory_mb = get_memory_mb()
        
        results['num_envs'].append(num_envs)
        results['env_fps'].append(env_fps)
        results['train_fps'].append(train_fps)
        results['memory_mb'].append(memory_mb)
        
        print(f"  Env FPS: {env_fps:.2f}")
        print(f"  Train FPS: {train_fps:.2f}")
        print(f"  Memory: {memory_mb:.2f} MB")

        env.close()

        if is_swapping(baseline_swap):
            print(f"  STOP: swap grew +{swap_used_mb() - baseline_swap:.0f}MB at {num_envs} envs")
            break
        # plateau: train FPS no longer improving (primary signal)
        if detect_plateau(train_fps_history):
            print(f"  Train FPS plateaued at {num_envs} envs")
            break


        num_envs *= 2

    
    return results


def run_all_benchmarks():
    all_results = []
    
    # PPO on CartPole
    result = benchmark_algorithm(
        PPO,
        'PPO',
        'CartPole-v1',
        start_num_envs=16,
        max_num_envs=1_000_000,
        warmup_steps=50,
        measure_steps=200,
    )
    all_results.append(result)

    # SAC on Pendulum
    result = benchmark_algorithm(
        SAC,
        'SAC',
        'Pendulum-v1',
        start_num_envs=16,
        max_num_envs=1_000_000,

        warmup_steps=50,
        measure_steps=200,
    )
    all_results.append(result)
    
    return all_results


def print_results(results: List[Dict]):
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    for r in results:
        print(f"\n{r['algorithm']} on {r['env']}")
        print("-" * 60)
        print(f"{'Num Envs':<12} {'Env FPS':<12} {'Train FPS':<12} {'Memory (MB)':<12}")
        print("-" * 60)
        for i in range(len(r['num_envs'])):
            print(f"{r['num_envs'][i]:<12} {r['env_fps'][i]:<12.2f} {r['train_fps'][i]:<12.2f} {r['memory_mb'][i]:<12.2f}")
        
        # Calculate scaling efficiency
        if len(r['num_envs']) > 1:
            print("\nScaling Efficiency (FPS / Num Envs):")
            print(f"{'Num Envs':<12} {'Env Eff':<12} {'Train Eff':<12}")
            print("-" * 60)
            for i in range(len(r['num_envs'])):
                env_eff = r['env_fps'][i] / r['num_envs'][i]
                train_eff = r['train_fps'][i] / r['num_envs'][i]
                print(f"{r['num_envs'][i]:<12} {env_eff:<12.2f} {train_eff:<12.2f}")


if __name__ == '__main__':
    results = run_all_benchmarks()
    print_results(results)
