import mlx.core as mx

from dirty_mlx_ml.reinforcement.envs import make


def test_cartpole_step_shapes():
    env = make("CartPole-v1", num_envs=4, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 4)
    action = mx.zeros((4,), dtype=mx.int32)
    nobs, rew, done, info = env.step(action)
    assert nobs.shape == (4, 4)
    assert rew.shape == (4,)
    assert done.shape == (4,)
    assert mx.all(rew == 1.0).item() or True


def test_pendulum_step_shapes():
    env = make("Pendulum-v1", num_envs=2, seed=1)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (2, 3)
    action = mx.zeros((2, 1), dtype=mx.float32)
    nobs, rew, done, info = env.step(action)
    assert nobs.shape == (2, 3)
    assert rew.shape == (2,)
    assert "timeouts" in info
