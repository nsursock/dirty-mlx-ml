import time
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..buffers import RolloutBuffer
from ..logger import CSVLogger
from ..nn import ActorCriticContinuous, ActorCriticDiscrete
from ..spaces import Discrete
from ..utils import explained_variance, to_float


class PPO:
    def __init__(
        self,
        policy: str,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf=None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl=None,
        stats_window_size: int = 100,
        seed: int | None = None,
        verbose: int = 0,
        log_dir: str | None = None,
        policy_kwargs: dict | None = None,
        **kwargs,
    ):
        self.env = env
        self.n_envs = env.num_envs
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        # scale batch with n_envs → constant #minibatches; more envs = wider GPU mats, fewer updates
        self.batch_size = min(batch_size * self.n_envs, n_steps * self.n_envs)
        self.n_epochs = n_epochs

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.verbose = verbose
        self.num_timesteps = 0
        self._n_updates = 0
        pk = policy_kwargs or {}
        hidden = tuple(pk.get("net_arch", [64, 64]))
        if seed is not None:
            mx.random.seed(seed)

        obs_dim = env.observation_space.shape[0]
        self.discrete = isinstance(env.action_space, Discrete)
        if self.discrete:
            self.act_dim = 1
            self.policy = ActorCriticDiscrete(obs_dim, env.action_space.n, hidden)
        else:
            self.act_dim = env.action_space.shape[0]
            self.policy = ActorCriticContinuous(obs_dim, self.act_dim, hidden)
        mx.eval(self.policy.parameters())

        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.buffer = RolloutBuffer(n_steps, self.n_envs, obs_dim, self.act_dim, gamma, gae_lambda)
        self.logger = CSVLogger(log_dir, name="ppo")
        self.ep_info_buffer = deque(maxlen=stats_window_size)
        self.ep_success_buffer = deque(maxlen=stats_window_size)

        self._ep_rew = mx.zeros((self.n_envs,))
        self._ep_len = mx.zeros((self.n_envs,))
        self._roll_rew_sum = mx.array(0.0)
        self._roll_len_sum = mx.array(0.0)
        self._roll_ep_count = mx.array(0.0)
        self._last_obs = None
        self._last_episode_starts = mx.ones((self.n_envs,))
        self.start_time = None
        self._num_timesteps_at_start = 0
        self._compiled_mb_step = None

    def _build_compiled_mb_step(self):
        policy = self.policy
        optimizer = self.optimizer
        clip_range = self.clip_range
        ent_coef, vf_coef = self.ent_coef, self.vf_coef
        normalize = self.normalize_advantage
        discrete = self.discrete
        max_grad_norm = self.max_grad_norm
        clip_range_vf = self.clip_range_vf
        use_vf_clip = clip_range_vf is not None
        vf_clip = 0.0 if clip_range_vf is None else float(clip_range_vf)
        do_clip_grad = max_grad_norm is not None
        grad_clip = 0.0 if max_grad_norm is None else float(max_grad_norm)

        def mb_step(obs, actions, old_values, old_log_prob, advantages, returns):
            def loss_fn(model):
                if discrete:
                    values, log_prob, entropy = model.evaluate(
                        obs, mx.reshape(actions, (-1,)).astype(mx.int32)
                    )
                else:
                    values, log_prob, entropy = model.evaluate(obs, actions)
                adv = advantages
                if normalize:
                    adv = (adv - mx.mean(adv)) / (mx.std(adv) + 1e-8)
                ratio = mx.exp(log_prob - old_log_prob)
                p1 = adv * ratio
                p2 = adv * mx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
                policy_loss = -mx.mean(mx.minimum(p1, p2))
                if use_vf_clip:
                    values_pred = old_values + mx.clip(values - old_values, -vf_clip, vf_clip)
                else:
                    values_pred = values
                value_loss = mx.mean((returns - values_pred) ** 2)
                entropy_loss = -mx.mean(entropy)
                loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
                clip_frac = mx.mean((mx.abs(ratio - 1.0) > clip_range).astype(mx.float32))
                approx_kl = mx.mean((ratio - 1.0) - (log_prob - old_log_prob))
                return loss, policy_loss, value_loss, entropy_loss, clip_frac, approx_kl

            (loss, pg, vl, en, cf, kl), grads = nn.value_and_grad(policy, loss_fn)(policy)
            if do_clip_grad:
                grads, _ = optim.clip_grad_norm(grads, grad_clip)
            optimizer.update(policy, grads)
            return loss, pg, vl, en, cf, kl

        return mx.compile(
            mb_step,
            inputs=[policy.state, optimizer.state],
            outputs=[policy.state, optimizer.state],
        )

    def _get_compiled_mb_step(self):
        if self._compiled_mb_step is None:
            self._compiled_mb_step = self._build_compiled_mb_step()
        return self._compiled_mb_step

    def _update_ep_stats(self, rewards, dones):
        # pure MLX — no host sync on hot path
        self._ep_rew = self._ep_rew + rewards
        self._ep_len = self._ep_len + 1.0
        d = dones.astype(mx.float32)
        finished = d > 0.5
        # running sums for mean at dump time
        n_fin = mx.sum(d)
        self._roll_rew_sum = self._roll_rew_sum + mx.sum(mx.where(finished, self._ep_rew, 0.0))
        self._roll_len_sum = self._roll_len_sum + mx.sum(mx.where(finished, self._ep_len, 0.0))
        self._roll_ep_count = self._roll_ep_count + n_fin
        mask = 1.0 - d
        self._ep_rew = self._ep_rew * mask
        self._ep_len = self._ep_len * mask


    def collect_rollouts(self):
        self.buffer.reset()
        if self._last_obs is None:
            self._last_obs, _ = self.env.reset()
            self._last_episode_starts = mx.ones((self.n_envs,))

        for _ in range(self.n_steps):
            obs = self._last_obs
            action, value, log_prob = self.policy.get_action(obs)
            if self.discrete:
                env_action = action
                store_action = mx.reshape(action.astype(mx.float32), (self.n_envs, 1))
            else:
                env_action = mx.clip(action, self.env.action_space.low, self.env.action_space.high)
                store_action = action

            new_obs, rewards, dones, infos = self.env.step(env_action)
            self.num_timesteps += self.n_envs
            self._update_ep_stats(rewards, dones)

            if isinstance(infos, dict) and "timeouts" in infos:
                timeouts = infos["timeouts"]
                last_v = self.policy.forward(new_obs)[1]
                rewards = rewards + self.gamma * last_v * timeouts

            self.buffer.add(obs, store_action, rewards, self._last_episode_starts, value, log_prob)
            self._last_obs = new_obs
            self._last_episode_starts = dones.astype(mx.float32)

        last_values = self.policy.forward(self._last_obs)[1]
        self.buffer.compute_returns_and_advantage(last_values, self._last_episode_starts)
        mx.eval(
            self.buffer.obs,
            self.buffer.actions,
            self.buffer.advantages,
            self.buffer.returns,
            self.buffer.values,
            self.buffer.log_probs,
        )

    def train(self):
        pg_s = vl_s = en_s = cf_s = kl_s = mx.array(0.0)
        n_mb = 0
        last_loss = mx.array(0.0)
        continue_training = True
        mb_step = self._get_compiled_mb_step()
        target_kl = self.target_kl

        for _epoch in range(self.n_epochs):
            for batch in self.buffer.get(self.batch_size):
                loss, pg, vl, en, cf, kl = mb_step(
                    batch["obs"],
                    batch["actions"],
                    batch["old_values"],
                    batch["old_log_prob"],
                    batch["advantages"],
                    batch["returns"],
                )
                last_loss = loss
                pg_s = pg_s + pg
                vl_s = vl_s + vl
                en_s = en_s + en
                cf_s = cf_s + cf
                kl_s = kl_s + kl
                n_mb += 1
                if target_kl is not None and to_float(kl) > 1.5 * target_kl:
                    continue_training = False
                    break
            self._n_updates += 1
            if not continue_training:
                break

        inv = 1.0 / max(n_mb, 1)
        mx.eval(
            self.policy.state,
            self.optimizer.state,
            pg_s,
            vl_s,
            en_s,
            cf_s,
            kl_s,
            last_loss,
        )
        ev = explained_variance(self.buffer.values, self.buffer.returns)
        self.logger.record("train/approx_kl", to_float(kl_s) * inv)
        self.logger.record("train/clip_fraction", to_float(cf_s) * inv)
        self.logger.record("train/clip_range", self.clip_range)
        self.logger.record("train/policy_gradient_loss", to_float(pg_s) * inv)
        self.logger.record("train/value_loss", to_float(vl_s) * inv)
        self.logger.record("train/entropy_loss", to_float(en_s) * inv)
        self.logger.record("train/learning_rate", self.learning_rate)
        self.logger.record("train/loss", to_float(last_loss))
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/explained_variance", ev)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", float(mx.mean(mx.exp(self.policy.log_std)).item()))
        else:
            self.logger.record("train/std", 0.0)


    def dump_logs(self, iteration: int = 0):
        elapsed = max(time.time() - self.start_time, 1e-9)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / elapsed)
        self.logger.record("time/fps", fps)
        self.logger.record("time/iterations", iteration)
        self.logger.record("time/time_elapsed", elapsed)
        self.logger.record("time/total_timesteps", self.num_timesteps)
        mx.eval(self._roll_rew_sum, self._roll_len_sum, self._roll_ep_count)
        n_ep = max(to_float(self._roll_ep_count), 1.0)
        self.logger.record("rollout/ep_rew_mean", to_float(self._roll_rew_sum) / n_ep)
        self.logger.record("rollout/ep_len_mean", to_float(self._roll_len_sum) / n_ep)
        self.logger.record("rollout/success_rate", 0.0)
        # decay window (approx stats_window)
        self._roll_rew_sum = self._roll_rew_sum * 0.0
        self._roll_len_sum = self._roll_len_sum * 0.0
        self._roll_ep_count = self._roll_ep_count * 0.0
        self.logger.dump(step=self.num_timesteps)



    def learn(self, total_timesteps: int, log_interval: int = 1, progress_bar: bool = False, **kwargs):
        self.start_time = time.time()
        self._num_timesteps_at_start = self.num_timesteps
        self._last_obs, _ = self.env.reset()
        self._last_episode_starts = mx.ones((self.n_envs,))
        iteration = 0
        while self.num_timesteps < total_timesteps:
            self.collect_rollouts()
            iteration += 1
            self.train()
            if log_interval and iteration % log_interval == 0:
                self.dump_logs(iteration)
        self.logger.close()
        return self

    def predict(self, observation, deterministic: bool = False):
        if not isinstance(observation, mx.array):
            observation = mx.array(observation, dtype=mx.float32)
        if observation.ndim == 1:
            observation = observation[None]
        action, _, _ = self.policy.get_action(observation, deterministic=deterministic)
        mx.eval(action)
        if self.discrete:
            return action, None
        return mx.clip(action, self.env.action_space.low, self.env.action_space.high), None
