"""SB3-style ``VecNormalize`` env wrapper.

Running mean/std normalization of observations and rewards, tracked in MLX.
The wrapper is transparent to the algorithms:

* SAC drives the env through the plain-Python :meth:`reset` / :meth:`step`
  path, so normalization + running-stat updates happen there.
* PPO drives the env through its compiled rollout, which reads the raw
  ``_step_fn`` of the wrapped env and applies the same normalization inline
  (see ``_build_compiled_rollout`` in ``algorithms/ppo.py``).

Running statistics are stored in the ``norm_state`` dict so they can be passed
as captured inputs/outputs to ``mx.compile`` and stay readable on the host for
``predict`` / ``normalize_obs``.
"""

import mlx.core as mx


def welford_update(mean, var, count, x):
    """Batch Welford update of running mean/var/count.

    ``x`` has shape ``(num_envs, *obs_shape)``; the running stats reduce over
    the leading ``num_envs`` axis.
    """
    batch_mean = mx.mean(x, axis=0)
    batch_var = mx.var(x, axis=0)
    batch_count = float(x.shape[0])
    total = count + batch_count
    delta = batch_mean - mean
    new_mean = mean + delta * (batch_count / total)
    new_var = (
        var * count + batch_var * batch_count + (delta * delta) * count * batch_count / total
    ) / total
    return new_mean, new_var, total


def normalize(x, mean, var, eps, clip):
    """Normalize ``x`` by running mean/std, optionally clipped."""
    x = (x - mean) / mx.sqrt(var + eps)
    if clip is not None:
        x = mx.clip(x, -clip, clip)
    return x


class VecNormalize:
    """Running mean/std observation + reward normalization wrapper.

    Mirrors SB3's ``VecNormalize``: observations are normalized by their running
    mean/std; rewards are normalized by the running std of the discounted return
    (with ``gamma``), which makes Q/advantage learning robust to reward scale.
    """

    def __init__(
        self,
        venv,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float | None = 10.0,
        clip_reward: float | None = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.single_observation_space = getattr(venv, "single_observation_space", venv.observation_space)
        self.single_action_space = getattr(venv, "single_action_space", venv.action_space)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        obs_shape = venv.observation_space.shape
        self.norm_state = {
            "obs_mean": mx.zeros(obs_shape),
            "obs_var": mx.ones(obs_shape),
            "obs_count": mx.array(0.0),
            "ret_mean": mx.array(0.0),
            "ret_var": mx.array(1.0),
            "ret_count": mx.array(0.0),
            "returns": mx.zeros((self.num_envs,)),
        }

    # -- raw env internals (delegated so PPO's compiled rollout can reach through)
    @property
    def _state(self):
        return self.venv._state

    @_state.setter
    def _state(self, value):
        self.venv._state = value

    @property
    def _steps(self):
        return self.venv._steps

    @_steps.setter
    def _steps(self, value):
        self.venv._steps = value

    @property
    def _prev_done(self):
        return self.venv._prev_done

    @_prev_done.setter
    def _prev_done(self, value):
        self.venv._prev_done = value

    @property
    def _key(self):
        return self.venv._key

    @_key.setter
    def _key(self, value):
        self.venv._key = value

    @property
    def _step_fn(self):
        return self.venv._step_fn

    @_step_fn.setter
    def _step_fn(self, value):
        self.venv._step_fn = value

    def _build_step(self):
        if getattr(self.venv, "_step_fn", None) is None:
            self.venv._build_step()

    # -- normalization helpers (host-side, used by predict / SAC)
    def normalize_obs(self, obs):
        s = self.norm_state
        if self.norm_obs:
            obs = normalize(obs, s["obs_mean"], s["obs_var"], self.epsilon, self.clip_obs)
        return obs

    def normalize_reward(self, reward):
        s = self.norm_state
        if self.norm_reward:
            reward = reward / mx.sqrt(s["ret_var"] + self.epsilon)
            if self.clip_reward is not None:
                reward = mx.clip(reward, -self.clip_reward, self.clip_reward)
        return reward

    def unnormalize_reward(self, reward):
        return reward * mx.sqrt(self.norm_state["ret_var"] + self.epsilon)

    def _update(self, obs, reward, done):
        s = self.norm_state
        if self.norm_obs:
            s["obs_mean"], s["obs_var"], s["obs_count"] = welford_update(
                s["obs_mean"], s["obs_var"], s["obs_count"], obs
            )
        if self.norm_reward:
            rets = s["returns"] * self.gamma + reward
            rets = mx.where(done, 0.0, rets)
            s["returns"] = rets
            s["ret_mean"], s["ret_var"], s["ret_count"] = welford_update(
                s["ret_mean"], s["ret_var"], s["ret_count"], rets
            )

    def reset(self, seed=None):
        obs, info = self.venv.reset(seed=seed)
        done = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self._update(obs, mx.zeros((self.num_envs,)), done)
        self.norm_state["returns"] = mx.zeros((self.num_envs,))
        return self.normalize_obs(obs), info

    def step(self, action):
        obs, reward, done, infos = self.venv.step(action)
        self._update(obs, reward, done)
        return self.normalize_obs(obs), self.normalize_reward(reward), done, infos

    def close(self):
        self.venv.close()

    def __getattr__(self, name):
        # Fall through to the wrapped env for anything not explicitly defined
        # (e.g. custom env attributes the algorithms might reach for).
        venv = self.__dict__.get("venv")
        if venv is None:
            raise AttributeError(name)
        return getattr(venv, name)
