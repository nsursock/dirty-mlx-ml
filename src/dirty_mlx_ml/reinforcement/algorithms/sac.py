import math
import time
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from ..buffers import ReplayBuffer
from ..logger import CSVLogger
from ..nn import SACActor, TwinQ
from ..rollout_logging import (
    COMPLETED_ONLY,
    ONGOING,
    normalize_rollout_log_mode,
    should_skip_completed_only_dump,
)
from ..utils import polyak_update, to_float
from ..vec_normalize import VecNormalize


class LogAlpha(nn.Module):
    def __init__(self, init: float = 1.0):
        super().__init__()
        self.log_alpha = mx.array(math.log(init), dtype=mx.float32)

    def __call__(self):
        return self.log_alpha


class SAC:
    def __init__(
        self,
        policy: str,
        env,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
        stats_window_size: int = 100,
        seed: int | None = None,
        verbose: int = 0,
        log_dir: str | None = None,
        policy_kwargs: dict | None = None,
        max_grad_norm: float | None = None,
        max_ent_coef: float | None = None,
        rollout_log_mode: str = COMPLETED_ONLY,
        **kwargs,
    ):
        self.env = env
        self.n_envs = env.num_envs
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq if isinstance(train_freq, int) else train_freq[0]
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.verbose = verbose
        self.num_timesteps = 0
        self._n_updates = 0
        self.max_grad_norm = max_grad_norm
        self.max_ent_coef = max_ent_coef
        self.rollout_log_mode = normalize_rollout_log_mode(rollout_log_mode)
        pk = policy_kwargs or {}
        hidden = tuple(pk.get("net_arch", [256, 256]))
        if seed is not None:
            mx.random.seed(seed)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.act_dim = act_dim
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self._act_scale = (self.action_high - self.action_low) / 2.0
        self._act_bias = (self.action_high + self.action_low) / 2.0

        self.actor = SACActor(obs_dim, act_dim, hidden)
        self.critic = TwinQ(obs_dim, act_dim, hidden)
        self.critic_target = TwinQ(obs_dim, act_dim, hidden)
        self.critic_target.update(tree_map(lambda x: mx.array(x), self.critic.parameters()))
        mx.eval(self.actor.parameters(), self.critic.parameters(), self.critic_target.parameters())

        self.actor_opt = optim.Adam(learning_rate=learning_rate)
        self.critic_opt = optim.Adam(learning_rate=learning_rate)

        self.target_entropy = -float(act_dim) if target_entropy == "auto" else float(target_entropy)
        self.alpha_mod = None
        self.ent_coef_fixed = None
        self.ent_opt = None
        if isinstance(ent_coef, str) and ent_coef.startswith("auto"):
            init = float(ent_coef.split("_")[1]) if "_" in ent_coef else 1.0
            self.alpha_mod = LogAlpha(init)
            self.ent_opt = optim.Adam(learning_rate=learning_rate)
            mx.eval(self.alpha_mod.parameters())
        else:
            self.ent_coef_fixed = float(ent_coef)

        self.replay = ReplayBuffer(buffer_size, self.n_envs, obs_dim, act_dim)
        self.logger = CSVLogger(log_dir, name="sac")
        self.ep_info_buffer = deque(maxlen=stats_window_size)
        self.ep_success_buffer = deque(maxlen=stats_window_size)

        self._ep_rew = mx.zeros((self.n_envs,))
        self._ep_len = mx.zeros((self.n_envs,))
        self._roll_rew_sum = mx.array(0.0)
        self._roll_len_sum = mx.array(0.0)
        self._roll_ep_count = mx.array(0.0)
        self._step_rew_sum = mx.array(0.0)
        self._step_count = mx.array(0.0)
        self._last_obs = None
        self.start_time = None
        self._num_timesteps_at_start = 0
        self._compiled_step = None
        self._compiled_auto_ent = None
        self._rng_key = mx.random.key(seed if seed is not None else 0)
        self._train_sums = None
        self._train_count = 0

    def _scale_action(self, action_tanh):
        return action_tanh * self._act_scale + self._act_bias

    def _update_ep_stats(self, rewards, dones):
        self._ep_rew = self._ep_rew + rewards
        self._ep_len = self._ep_len + 1.0
        self._step_rew_sum = self._step_rew_sum + mx.sum(rewards)
        self._step_count = self._step_count + float(rewards.shape[0])

        if isinstance(dones, tuple) and len(dones) == 2:
            terminated, truncated = dones
            d = mx.maximum(terminated, truncated)
        else:
            d = dones

        d = d.astype(mx.float32)
        finished = d > 0.5
        self._roll_rew_sum = self._roll_rew_sum + mx.sum(mx.where(finished, self._ep_rew, 0.0))
        self._roll_len_sum = self._roll_len_sum + mx.sum(mx.where(finished, self._ep_len, 0.0))
        self._roll_ep_count = self._roll_ep_count + mx.sum(d)
        mask = 1.0 - d
        self._ep_rew = self._ep_rew * mask
        self._ep_len = self._ep_len * mask

    def _sample_action(self, obs, random=False):
        if random:
            a = mx.random.uniform(-1.0, 1.0, (self.n_envs, self.act_dim))
            return self._scale_action(a)
        a_tanh, _ = self.actor.sample(obs, deterministic=False)
        return self._scale_action(a_tanh)

    def _build_compiled_step(self):
        actor = self.actor
        critic = self.critic
        critic_target = self.critic_target
        actor_opt = self.actor_opt
        critic_opt = self.critic_opt
        alpha_mod = self.alpha_mod
        ent_opt = self.ent_opt
        gamma = self.gamma
        tau = self.tau
        target_entropy = self.target_entropy
        auto_ent = alpha_mod is not None
        ent_coef_fixed = self.ent_coef_fixed
        do_clip_grad = self.max_grad_norm is not None
        grad_clip = 0.0 if self.max_grad_norm is None else float(self.max_grad_norm)
        do_clip_ent = self.max_ent_coef is not None
        ent_clip = 0.0 if self.max_ent_coef is None else float(self.max_ent_coef)

        def step(key, obs, next_obs, actions, rewards, terminated, tau_step):
            k1, k2, k3, key = mx.random.split(key, 4)
            if auto_ent:
                _, log_prob_alpha = actor.sample(obs, key=k1)
                log_prob_alpha = mx.stop_gradient(log_prob_alpha)

                def ent_loss_fn(mod):
                    return (-mod.log_alpha * (log_prob_alpha + target_entropy)).mean()

                e_loss, e_grads = nn.value_and_grad(alpha_mod, ent_loss_fn)(alpha_mod)
                if do_clip_grad:
                    e_grads, _ = optim.clip_grad_norm(e_grads, grad_clip)
                ent_opt.update(alpha_mod, e_grads)
                ent_coef = mx.exp(mx.stop_gradient(alpha_mod.log_alpha))
            else:
                e_loss = 0.0
                ent_coef = mx.array(ent_coef_fixed, dtype=mx.float32)

            if do_clip_ent:
                ent_coef = mx.clip(ent_coef, 0.0, ent_clip)

            next_actions, next_log_prob = actor.sample(next_obs, key=k2)
            next_actions = mx.stop_gradient(next_actions)
            next_log_prob = mx.stop_gradient(next_log_prob)
            nq1, nq2 = critic_target(next_obs, next_actions)
            next_q = mx.minimum(nq1, nq2) - ent_coef * next_log_prob

            term = terminated.reshape(-1)
            should_bootstrap = 1.0 - term
            target_q = rewards.reshape(-1) + should_bootstrap * gamma * next_q
            target_q = mx.stop_gradient(target_q)

            def critic_loss_fn(model):
                q1, q2 = model(obs, actions)
                l1 = mx.mean((q1 - target_q) ** 2)
                l2 = mx.mean((q2 - target_q) ** 2)
                return 0.5 * (l1 + l2), l1, l2, mx.mean(mx.minimum(q1, q2))

            (c_loss, l1, l2, q_mean), c_grads = nn.value_and_grad(critic, critic_loss_fn)(critic)
            if do_clip_grad:
                c_grads, _ = optim.clip_grad_norm(c_grads, grad_clip)
            critic_opt.update(critic, c_grads)

            def actor_loss_fn(model):
                actions_pi, log_prob = model.sample(obs, key=k3)
                q1, q2 = critic(obs, actions_pi)
                loss = mx.mean(ent_coef * log_prob - mx.minimum(q1, q2))
                return loss, mx.mean(log_prob)

            (a_loss, log_pi_actor), a_grads = nn.value_and_grad(actor, actor_loss_fn)(actor)
            if do_clip_grad:
                a_grads, _ = optim.clip_grad_norm(a_grads, grad_clip)
            actor_opt.update(actor, a_grads)

            if auto_ent:
                log_pi_stat = mx.mean(log_prob_alpha)
            else:
                log_pi_stat = mx.stop_gradient(log_pi_actor)

            critic_target.update(
                polyak_update(critic.parameters(), critic_target.parameters(), tau_step)
            )

            return c_loss, a_loss, e_loss, ent_coef, l1, l2, q_mean, log_pi_stat, key

        inputs = [actor.state, critic.state, critic_target.state, actor_opt.state, critic_opt.state]
        outputs = [actor.state, critic.state, critic_target.state, actor_opt.state, critic_opt.state]
        if auto_ent:
            inputs.extend([alpha_mod.state, ent_opt.state])
            outputs.extend([alpha_mod.state, ent_opt.state])

        return mx.compile(step, inputs=inputs, outputs=outputs)

    def _get_compiled_step(self):
        auto_ent = self.alpha_mod is not None
        if self._compiled_step is None or self._compiled_auto_ent != auto_ent:
            self._compiled_step = self._build_compiled_step()
            self._compiled_auto_ent = auto_ent
        return self._compiled_step

    def train(self, gradient_steps: int, batch_size: int):
        actor_s = mx.array(0.0)
        critic_s = mx.array(0.0)
        ent_s = mx.array(0.0)
        ent_loss_s = mx.array(0.0)
        q1_s = mx.array(0.0)
        q2_s = mx.array(0.0)
        q_mean_s = mx.array(0.0)
        log_pi_s = mx.array(0.0)
        last_loss = mx.array(0.0)
        step_fn = self._get_compiled_step()

        for gs in range(gradient_steps):
            batch = self.replay.sample(batch_size)
            # tau_step=0 skips polyak when outside target_update_interval
            tau_step = self.tau if (gs % self.target_update_interval == 0) else 0.0
            c_loss, a_loss, e_loss, ent_coef, l1, l2, q_mean, log_pi, key = step_fn(
                self._rng_key,
                batch["obs"],
                batch["next_obs"],
                batch["actions"],
                batch["rewards"],
                batch["terminated"],
                tau_step,
            )
            self._rng_key = key
            critic_s = critic_s + c_loss
            actor_s = actor_s + a_loss
            ent_s = ent_s + ent_coef
            ent_loss_s = ent_loss_s + e_loss
            q1_s = q1_s + l1
            q2_s = q2_s + l2
            q_mean_s = q_mean_s + q_mean
            log_pi_s = log_pi_s + log_pi
            last_loss = a_loss

        self._n_updates += gradient_steps
        # Accumulate metrics lazily; materialize + log only at dump time so the
        # per-step hot path stays host-sync free.
        if self._train_sums is None:
            self._train_sums = {
                "actor_loss": mx.array(0.0),
                "critic_loss": mx.array(0.0),
                "ent_coef": mx.array(0.0),
                "ent_coef_loss": mx.array(0.0),
                "q1": mx.array(0.0),
                "q2": mx.array(0.0),
                "q_mean": mx.array(0.0),
                "log_pi": mx.array(0.0),
                "loss": mx.array(0.0),
            }
        s = self._train_sums
        s["actor_loss"] = s["actor_loss"] + actor_s
        s["critic_loss"] = s["critic_loss"] + critic_s
        s["ent_coef"] = s["ent_coef"] + ent_s
        s["ent_coef_loss"] = s["ent_coef_loss"] + ent_loss_s
        s["q1"] = s["q1"] + q1_s
        s["q2"] = s["q2"] + q2_s
        s["q_mean"] = s["q_mean"] + q_mean_s
        s["log_pi"] = s["log_pi"] + log_pi_s
        s["loss"] = s["loss"] + last_loss
        self._train_count += 1

    def _flush_train_metrics(self):
        if not self._train_sums or self._train_count == 0:
            return
        s = self._train_sums
        mx.eval(
            self.actor.state,
            self.critic.state,
            self.critic_target.state,
            self.actor_opt.state,
            self.critic_opt.state,
            *s.values(),
        )
        if self.alpha_mod is not None:
            mx.eval(self.alpha_mod.state, self.ent_opt.state)

        n = max(self._train_count, 1)
        self.logger.record("train/loss/policy", to_float(s["actor_loss"]) / n)
        self.logger.record("train/actor_loss", to_float(s["actor_loss"]) / n)
        self.logger.record("train/critic_loss", to_float(s["critic_loss"]) / n)
        self.logger.record("train/loss/q1", to_float(s["q1"]) / n)
        self.logger.record("train/loss/q2", to_float(s["q2"]) / n)
        self.logger.record("train/loss/alpha", to_float(s["ent_coef_loss"]) / n)
        self.logger.record("train/policy/alpha", to_float(s["ent_coef"]) / n)
        self.logger.record("train/ent_coef", to_float(s["ent_coef"]) / n)
        self.logger.record("train/value/q_mean", to_float(s["q_mean"]) / n)
        self.logger.record("train/policy/log_pi_mean", to_float(s["log_pi"]) / n)
        self.logger.record("train/ent_coef_loss", to_float(s["ent_coef_loss"]) / n)
        self.logger.record("train/learning_rate", self.learning_rate)
        self.logger.record("train/loss", to_float(s["loss"]) / n)
        self.logger.record("train/n_updates", self._n_updates)
        for k in self._train_sums:
            self._train_sums[k] = mx.array(0.0)
        self._train_count = 0

    def dump_logs(self, iteration: int = 0, force: bool = False):
        """Write one progress CSV row.

        ``completed_only`` (default): skip the row when no episode finished
        since the last dump, unless ``force=True`` (used at training end).
        ``ongoing``: always write; classic ``ep_*`` columns stay finite via
        in-progress episode accumulators when nothing has completed yet.
        """
        if self.start_time is None:
            self.start_time = time.time()
        mx.eval(self._roll_rew_sum, self._roll_len_sum, self._roll_ep_count)
        n_ep = to_float(self._roll_ep_count)
        if self.rollout_log_mode == COMPLETED_ONLY and should_skip_completed_only_dump(n_ep, force):
            return False

        elapsed = max(time.time() - self.start_time, 1e-9)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / elapsed)
        self._flush_train_metrics()
        self.logger.record("time/fps", fps)
        self.logger.record("time/iterations", iteration)
        self.logger.record("time/time_elapsed", elapsed)
        self.logger.record("time/total_timesteps", self.num_timesteps)

        mx.eval(self._ep_rew, self._ep_len, self._step_rew_sum, self._step_count)
        ongoing_rew = to_float(mx.mean(self._ep_rew))
        ongoing_len = to_float(mx.mean(self._ep_len))
        step_n = to_float(self._step_count)
        step_rew = (to_float(self._step_rew_sum) / step_n) if step_n > 0 else 0.0

        if n_ep > 0:
            ep_rew = to_float(self._roll_rew_sum) / n_ep
            ep_len = to_float(self._roll_len_sum) / n_ep
        elif self.rollout_log_mode == ONGOING:
            ep_rew, ep_len = ongoing_rew, ongoing_len
        else:
            ep_rew, ep_len = float("nan"), float("nan")

        self.logger.record("rollout/ep_rew_mean", ep_rew)
        self.logger.record("rollout/ep_len_mean", ep_len)
        if self.rollout_log_mode == ONGOING:
            self.logger.record("rollout/ongoing_ep_rew_mean", ongoing_rew)
            self.logger.record("rollout/ongoing_ep_len_mean", ongoing_len)
            self.logger.record("rollout/step_rew_mean", step_rew)
        self.logger.record("rollout/success_rate", 0.0)

        self._roll_rew_sum = self._roll_rew_sum * 0.0
        self._roll_len_sum = self._roll_len_sum * 0.0
        self._roll_ep_count = self._roll_ep_count * 0.0
        self._step_rew_sum = self._step_rew_sum * 0.0
        self._step_count = self._step_count * 0.0
        self.logger.dump(step=self.num_timesteps)
        return True

    def learn(self, total_timesteps: int, log_interval: int = 4, progress_bar: bool = False, callback=None, **kwargs):
        self.start_time = time.time()
        self._num_timesteps_at_start = self.num_timesteps
        if callback is not None:
            callback.init_callback(self)
            callback.on_training_start()
        self._last_obs, _ = self.env.reset()
        iteration = 0
        steps_per_cycle = self.train_freq
        while self.num_timesteps < total_timesteps:
            for _ in range(steps_per_cycle):
                obs = self._last_obs
                action = self._sample_action(obs, random=self.num_timesteps < self.learning_starts)
                new_obs, rewards, dones, infos = self.env.step(action)
                self.num_timesteps += self.n_envs
                self._update_ep_stats(rewards, dones)

                if isinstance(dones, tuple) and len(dones) == 2:
                    terminated, truncated = dones
                else:
                    terminated = dones
                    truncated = mx.zeros((self.n_envs,))

                if isinstance(infos, dict) and "timeouts" in infos:
                    timeouts = infos["timeouts"]
                    truncated = mx.maximum(truncated, timeouts)

                self.replay.add(
                    obs, new_obs, action, rewards, terminated.astype(mx.float32), truncated.astype(mx.float32)
                )
                self._last_obs = new_obs

            if self.num_timesteps >= self.learning_starts:
                gs = self.gradient_steps if self.gradient_steps > 0 else steps_per_cycle
                self.train(gs, self.batch_size)

            iteration += 1
            if callback is not None and not callback.on_step():
                break
            if log_interval and iteration % log_interval == 0:
                self.dump_logs(iteration)

        self.dump_logs(iteration, force=True)
        if callback is not None:
            callback.on_training_end()
        self.logger.close()
        return self

    def predict(self, observation, deterministic: bool = False):
        if not isinstance(observation, mx.array):
            observation = mx.array(observation, dtype=mx.float32)
        if observation.ndim == 1:
            observation = observation[None]
        if isinstance(self.env, VecNormalize):
            observation = self.env.normalize_obs(observation)
        a_tanh, _ = self.actor.sample(observation, deterministic=deterministic)
        action = self._scale_action(a_tanh)
        mx.eval(action)
        return action, None
