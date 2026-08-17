"""SB3-style training callbacks.

A minimal callback system mirroring stable-baselines3's ``BaseCallback``:
``EvalCallback`` periodically evaluates the policy and ``StopTrainingOnRewardThreshold``
halts training once a mean-reward threshold is reached. Callbacks are passed to
``PPO.learn`` / ``SAC.learn`` via the ``callback`` keyword argument.
"""

import math

from .utils import to_float


class BaseCallback:
    """Base class for callbacks passed into ``learn()``.

    Subclasses override ``on_step`` (and optionally the lifecycle hooks).
    ``on_step`` returns ``False`` to stop training early.
    """

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.model = None
        self.n_calls = 0

    def init_callback(self, model):
        self.model = model

    def on_training_start(self):
        pass

    def on_step(self) -> bool:
        self.n_calls += 1
        return True

    def on_training_end(self):
        pass


class EvalCallback(BaseCallback):
    """Periodically evaluate the policy and record ``eval/mean_reward``.

    Args:
        eval_env: Vectorized env used for evaluation (typically ``num_envs=1``).
        callback_on_new_best: Optional child callback invoked after each
            evaluation (e.g. :class:`StopTrainingOnRewardThreshold`).
        eval_freq: Evaluate every ``eval_freq`` timesteps.
        n_eval_episodes: Number of episodes per evaluation.
        deterministic: Use deterministic actions during evaluation.
    """

    def __init__(
        self,
        eval_env,
        callback_on_new_best=None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.callback_on_new_best = callback_on_new_best
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_mean_reward = -math.inf
        self.last_mean_reward = -math.inf
        self._last_eval_timestep = 0

    def _evaluate(self) -> float:
        env = self.eval_env
        obs, _ = env.reset()
        rewards = []
        ep_reward = 0.0
        n_episodes = 0
        max_steps = getattr(env, "max_episode_steps", 500)
        cap = max_steps * self.n_eval_episodes + 100
        steps = 0
        while n_episodes < self.n_eval_episodes and steps < cap:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done, _ = env.step(action)
            ep_reward += to_float(reward)
            steps += 1
            if to_float(done) > 0.5:
                rewards.append(ep_reward)
                ep_reward = 0.0
                n_episodes += 1
                obs, _ = env.reset()
        if rewards:
            return sum(rewards) / len(rewards)
        return ep_reward / max(n_episodes, 1)

    def on_step(self) -> bool:
        if self.model.num_timesteps - self._last_eval_timestep >= self.eval_freq:
            self._last_eval_timestep = self.model.num_timesteps
            mean_reward = self._evaluate()
            self.last_mean_reward = mean_reward
            self.best_mean_reward = max(self.best_mean_reward, mean_reward)
            self.model.logger.record("eval/mean_reward", mean_reward)
            self.model.logger.record("eval/best_mean_reward", self.best_mean_reward)
            if self.verbose:
                print(f"Eval num_timesteps={self.model.num_timesteps}, mean_reward={mean_reward:.2f}")
            if self.callback_on_new_best is not None:
                self.callback_on_new_best.init_callback(self.model)
                self.callback_on_new_best.last_mean_reward = mean_reward
                return self.callback_on_new_best.on_step()
        self.n_calls += 1
        return True


class StopTrainingOnRewardThreshold(BaseCallback):
    """Stop training once the (eval) mean reward reaches ``reward_threshold``."""

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.last_mean_reward = -math.inf

    def on_step(self) -> bool:
        self.n_calls += 1
        if self.last_mean_reward >= self.reward_threshold:
            if self.verbose:
                print(
                    f"Stopping training: mean reward {self.last_mean_reward:.2f} "
                    f">= threshold {self.reward_threshold:.2f}"
                )
            return False
        return True
