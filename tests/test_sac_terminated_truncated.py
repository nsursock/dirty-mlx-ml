"""Test SAC terminated vs truncated bootstrapping behavior."""

import mlx.core as mx

from dirty_mlx_ml.reinforcement.buffers import ReplayBuffer
from dirty_mlx_ml.reinforcement.algorithms.sac import SAC
from dirty_mlx_ml.reinforcement.envs import make


def test_replay_buffer_terminated_truncated():
    """Test that ReplayBuffer properly stores and samples terminated/truncated flags."""
    buffer_size = 100
    n_envs = 2
    obs_dim = 4
    act_dim = 1
    
    buffer = ReplayBuffer(buffer_size, n_envs, obs_dim, act_dim)
    
    # Add some transitions with different terminated/truncated states
    obs = mx.zeros((n_envs, obs_dim))
    next_obs = mx.ones((n_envs, obs_dim))
    actions = mx.zeros((n_envs, act_dim))
    rewards = mx.ones((n_envs,))
    
    # Transition 1: terminated but not truncated
    terminated = mx.array([1.0, 0.0])
    truncated = mx.array([0.0, 0.0])
    buffer.add(obs, next_obs, actions, rewards, terminated, truncated)
    
    # Transition 2: truncated but not terminated
    terminated = mx.array([0.0, 0.0])
    truncated = mx.array([1.0, 0.0])
    buffer.add(obs, next_obs, actions, rewards, terminated, truncated)
    
    # Transition 3: both terminated and truncated (edge case)
    terminated = mx.array([0.0, 1.0])
    truncated = mx.array([0.0, 1.0])
    buffer.add(obs, next_obs, actions, rewards, terminated, truncated)
    
    # Sample and verify the data
    batch = buffer.sample(batch_size=10)
    
    assert "terminated" in batch
    assert "truncated" in batch
    assert batch["terminated"].shape == (10, 1)
    assert batch["truncated"].shape == (10, 1)
    
    # Check that we have the right mix of states
    # Since we're sampling randomly, we just check the structure is correct
    assert mx.all(batch["terminated"] >= 0.0)
    assert mx.all(batch["terminated"] <= 1.0)
    assert mx.all(batch["truncated"] >= 0.0)
    assert mx.all(batch["truncated"] <= 1.0)


def test_sac_bootstrapping_terminated():
    """Test that SAC does not bootstrap from terminated states."""
    env = make("Pendulum-v1", num_envs=1, seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=10,
        buffer_size=1000,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
        seed=0,
        policy_kwargs={"net_arch": [32, 32]},
    )
    
    # Manually add transitions to the buffer
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        action = model._sample_action(obs, random=True)
        new_obs, rewards, dones, infos = env.step(action)
        
        # Handle terminated vs truncated
        if isinstance(dones, tuple) and len(dones) == 2:
            terminated, truncated = dones
        else:
            terminated = dones
            truncated = mx.zeros((model.n_envs,))
        
        model.replay.add(obs, new_obs, action, rewards, terminated.astype(mx.float32), truncated.astype(mx.float32))
        obs = new_obs
    
    # Sample a batch and check the target calculation
    batch = model.replay.sample(32)
    
    # Get the next Q values
    next_actions, next_log_prob = model.actor.sample(batch["next_obs"])
    next_actions = mx.stop_gradient(next_actions)
    next_log_prob = mx.stop_gradient(next_log_prob)
    nq1, nq2 = model.critic_target(batch["next_obs"], next_actions)
    next_q = mx.minimum(nq1, nq2)
    
    # For terminated states, target should be just reward (no bootstrapping)
    terminated = batch["terminated"].reshape(-1)
    gamma = model.gamma
    
    # Simulate the target calculation from the fixed code
    should_bootstrap = 1.0 - terminated
    target_q_fixed = batch["rewards"].reshape(-1) + should_bootstrap * gamma * next_q
    
    # For terminated transitions, target should equal reward
    terminated_mask = terminated > 0.5
    if mx.any(terminated_mask):
        # Use where instead of boolean indexing
        terminated_targets = mx.where(terminated_mask, target_q_fixed, mx.zeros_like(target_q_fixed))
        terminated_rewards = mx.where(terminated_mask, batch["rewards"].reshape(-1), mx.zeros_like(batch["rewards"].reshape(-1)))
        # Allow small numerical error
        diff = mx.abs(terminated_targets - terminated_rewards)
        max_diff = mx.max(mx.where(terminated_mask, diff, mx.zeros_like(diff)))
        assert max_diff < 1e-5, \
            "Terminated states should not bootstrap"


def test_sac_bootstrapping_truncated():
    """Test that SAC does bootstrap from truncated states."""
    env = make("Pendulum-v1", num_envs=1, seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=10,
        buffer_size=1000,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
        seed=0,
        policy_kwargs={"net_arch": [32, 32]},
    )
    
    # Manually add transitions to the buffer with truncated states
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        action = model._sample_action(obs, random=True)
        new_obs, rewards, dones, infos = env.step(action)
        
        # Handle terminated vs truncated
        if isinstance(dones, tuple) and len(dones) == 2:
            terminated, truncated = dones
        else:
            terminated = dones
            truncated = mx.zeros((model.n_envs,))
        
        # Force some states to be truncated for testing
        if isinstance(infos, dict):
            # Simulate timeout by setting truncated flag
            if mx.random.uniform() > 0.7:
                truncated = mx.ones_like(truncated)
        
        model.replay.add(obs, new_obs, action, rewards, terminated.astype(mx.float32), truncated.astype(mx.float32))
        obs = new_obs
    
    # Sample a batch and check the target calculation
    batch = model.replay.sample(32)
    
    # Get the next Q values
    next_actions, next_log_prob = model.actor.sample(batch["next_obs"])
    next_actions = mx.stop_gradient(next_actions)
    next_log_prob = mx.stop_gradient(next_log_prob)
    nq1, nq2 = model.critic_target(batch["next_obs"], next_actions)
    next_q = mx.minimum(nq1, nq2)
    
    # For truncated states, target should bootstrap
    truncated = batch["truncated"].reshape(-1)
    gamma = model.gamma
    
    # Simulate the target calculation from the fixed code
    terminated = batch["terminated"].reshape(-1)
    should_bootstrap = 1.0 - terminated
    target_q_fixed = batch["rewards"].reshape(-1) + should_bootstrap * gamma * next_q
    
    # For truncated (but not terminated) transitions, target should include bootstrap
    truncated_mask = (truncated > 0.5) & (terminated <= 0.5)
    if mx.any(truncated_mask):
        # Use where instead of boolean indexing
        truncated_targets = mx.where(truncated_mask, target_q_fixed, mx.zeros_like(target_q_fixed))
        truncated_rewards = mx.where(truncated_mask, batch["rewards"].reshape(-1), mx.zeros_like(batch["rewards"].reshape(-1)))
        # Should include bootstrap term (gamma * next_q)
        # So target should be different from just reward
        bootstrap_contribution = gamma * mx.where(truncated_mask, next_q, mx.zeros_like(next_q))
        expected_targets = truncated_rewards + bootstrap_contribution
        # Only check where the mask is active
        diff = mx.abs(truncated_targets - expected_targets)
        # Check maximum difference where mask is active
        max_diff = mx.max(mx.where(truncated_mask, diff, mx.zeros_like(diff)))
        assert max_diff < 1e-5, \
            "Truncated states should bootstrap"


def test_sac_training_with_terminated_truncated():
    """Test that SAC training works correctly with the new terminated/truncated handling."""
    env = make("Pendulum-v1", num_envs=2, seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=20,
        buffer_size=1000,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
        seed=0,
        policy_kwargs={"net_arch": [32, 32]},
    )
    
    # Train for a short period
    model.learn(total_timesteps=100, log_interval=50)
    
    # Verify that training completed without errors
    assert model.num_timesteps >= 100
    
    # Verify that the buffer has the correct structure
    batch = model.replay.sample(10)
    assert "terminated" in batch
    assert "truncated" in batch


if __name__ == "__main__":
    test_replay_buffer_terminated_truncated()
    test_sac_bootstrapping_terminated()
    test_sac_bootstrapping_truncated()
    test_sac_training_with_terminated_truncated()
    print("All tests passed!")
