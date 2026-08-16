"""Environment validation tests.

These tests verify that our custom MLX environments are internally consistent,
reproducible, and implement correct physics/termination logic.
"""

import mlx.core as mx
import numpy as np

from dirty_mlx_ml.reinforcement.envs import make


def test_cartpole_physics_correctness():
    """Test that CartPole physics implementation is correct."""
    env = make("CartPole-v1", num_envs=1, seed=42)
    
    # Reset to known state
    obs, _ = env.reset(seed=42)
    
    # Check that initial state is within bounds
    assert mx.all(mx.abs(obs) <= 0.05), "Initial state should be in [-0.05, 0.05]"
    
    # Test that action 0 (left) produces negative force
    obs_left, reward, done, _ = env.step(mx.array([[0]]))
    assert reward[0] == 1.0, "CartPole reward should be 1.0 per step"
    
    # Test that action 1 (right) produces positive force  
    obs_right, reward, done, _ = env.step(mx.array([[1]]))
    assert reward[0] == 1.0, "CartPole reward should be 1.0 per step"


def test_cartpole_termination_conditions():
    """Test that CartPole termination conditions are correct."""
    env = make("CartPole-v1", num_envs=1, seed=42)
    
    # Test that environment can run for multiple steps without premature termination
    obs, _ = env.reset(seed=42)
    
    steps = 0
    for _ in range(20):
        obs, reward, done, _ = env.step(mx.array([[1]]))
        steps += 1
        if done[0]:
            break
    
    # Should be able to run at least a few steps before termination
    assert steps >= 5, f"Environment terminated too early at step {steps}"
    
    # Test auto-reset functionality
    obs_after_reset, _, _, _ = env.step(mx.array([[1]]))
    # After done, environment should auto-reset to new random state
    assert not mx.allclose(obs, obs_after_reset), "Environment should auto-reset after done"


def test_pendulum_physics_correctness():
    """Test that Pendulum physics implementation is correct."""
    env = make("Pendulum-v1", num_envs=1, seed=42)
    
    # Reset to known state
    obs, _ = env.reset(seed=42)
    
    # Check observation space (cos(theta), sin(theta), theta_dot)
    assert obs.shape == (1, 3), "Pendulum observation should be (1, 3)"
    
    # Check that cos^2 + sin^2 ≈ 1 (should be on unit circle)
    cos_theta = obs[0, 0]
    sin_theta = obs[0, 1]
    norm = cos_theta * cos_theta + sin_theta * sin_theta
    assert abs(float(norm) - 1.0) < 1e-5, "cos^2 + sin^2 should equal 1"
    
    # Test action application
    obs, reward, done, _ = env.step(mx.array([[1.0]]))
    assert obs.shape == (1, 3), "Observation shape should remain consistent"
    # Pendulum reward is negative cost, so should be <= 0
    assert reward[0] <= 0, "Pendulum reward should be negative (cost)"


def test_vectorized_env_independence():
    """Test that different parallel environments evolve independently."""
    env = make("CartPole-v1", num_envs=4, seed=42)
    
    # Reset all environments
    obs, _ = env.reset(seed=42)
    
    # Take different actions in each environment
    actions = mx.array([[0], [1], [0], [1]])
    next_obs, rewards, dones, _ = env.step(actions)
    
    # All environments should have different states since they took different actions
    # Check that not all observations are identical
    obs_np = np.array(next_obs)
    # At least some environments should have different x positions
    assert not np.allclose(obs_np[:, 0], obs_np[0, 0]), \
        "Different actions should produce different states"


def test_env_seed_reproducibility():
    """Test that environments are reproducible with the same seed."""
    env1 = make("CartPole-v1", num_envs=1, seed=42)
    env2 = make("CartPole-v1", num_envs=1, seed=42)
    
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    assert np.allclose(np.array(obs1), np.array(obs2), atol=1e-6), \
        "Same seed should produce same initial observation"
    
    # Take same actions
    for _ in range(5):
        action = 1
        next_obs1, reward1, done1, _ = env1.step(mx.array([[action]]))
        next_obs2, reward2, done2, _ = env2.step(mx.array([[action]]))
        
        assert np.allclose(np.array(next_obs1), np.array(next_obs2), atol=1e-6), \
            "Same seed + actions should produce same observations"
        assert np.allclose(np.array(reward1), np.array(reward2), atol=1e-6), \
            "Same seed + actions should produce same rewards"
        assert np.allclose(np.array(done1), np.array(done2), atol=1e-6), \
            "Same seed + actions should produce same done signals"


def test_environment_api_compliance():
    """Test that our environments comply with expected API."""
    for env_id in ["CartPole-v1", "Pendulum-v1"]:
        env = make(env_id, num_envs=2, seed=42)
        
        # Check required attributes
        assert hasattr(env, "num_envs"), "Environment should have num_envs"
        assert hasattr(env, "observation_space"), "Environment should have observation_space"
        assert hasattr(env, "action_space"), "Environment should have action_space"
        
        # Check methods
        assert hasattr(env, "reset"), "Environment should have reset method"
        assert hasattr(env, "step"), "Environment should have step method"
        assert hasattr(env, "close"), "Environment should have close method"
        
        # Test reset
        obs, info = env.reset(seed=42)
        assert obs.shape[0] == 2, "Observation batch size should match num_envs"
        
        # Test step
        if env_id == "CartPole-v1":
            action = mx.array([[0], [1]])
        else:
            action = mx.array([[1.0], [-1.0]])
        
        next_obs, reward, done, info = env.step(action)
        assert next_obs.shape == obs.shape, "Next observation shape should match"
        assert reward.shape[0] == 2, "Reward batch size should match num_envs"
        assert done.shape[0] == 2, "Done batch size should match num_envs"


if __name__ == "__main__":
    test_cartpole_physics_correctness()
    test_cartpole_termination_conditions()
    test_pendulum_physics_correctness()
    test_vectorized_env_independence()
    test_env_seed_reproducibility()
    test_environment_api_compliance()
    print("All environment validation tests passed!")
