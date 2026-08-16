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
from ..utils import polyak_update, to_float


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
        self._last_obs = None
        self.start_time = None
        self._num_timesteps_at_start = 0

    def _scale_action(self, action_tanh):
        return action_tanh * self._act_scale + self._act_bias

    def _update_ep_stats(self, rewards, dones):
        self._ep_rew = self._ep_rew + rewards
        self._ep_len = self._ep_len + 1.0
        d = dones.astype(mx.float32)
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

    def train(self, gradient_steps: int, batch_size: int):
        actor_s = critic_s = ent_s = ent_loss_s = 0.0
        q1_s = q2_s = q_mean_s = log_pi_s = 0.0
        n_ent = 0
        gamma, tau = self.gamma, self.tau
        target_entropy = self.target_entropy
        last_loss = 0.0

        for gs in range(gradient_steps):
            batch = self.replay.sample(batch_size)

            if self.alpha_mod is not None:
                ent_coef = mx.exp(mx.stop_gradient(self.alpha_mod.log_alpha))
            else:
                ent_coef = mx.array(self.ent_coef_fixed, dtype=mx.float32)

            if self.alpha_mod is not None:
                _, log_prob = self.actor.sample(batch["obs"])
                log_prob = mx.stop_gradient(log_prob)

                def ent_loss_fn(mod):
                    return (-mod.log_alpha * (log_prob + target_entropy)).mean()

                e_loss, e_grads = nn.value_and_grad(self.alpha_mod, ent_loss_fn)(self.alpha_mod)
                self.ent_opt.update(self.alpha_mod, e_grads)
                mx.eval(self.alpha_mod.parameters(), self.ent_opt.state)
                ent_loss_s += to_float(e_loss)
                n_ent += 1
                ent_coef = mx.exp(mx.stop_gradient(self.alpha_mod.log_alpha))

            ent_s += to_float(ent_coef)

            next_actions, next_log_prob = self.actor.sample(batch["next_obs"])
            next_actions = mx.stop_gradient(next_actions)
            next_log_prob = mx.stop_gradient(next_log_prob)
            nq1, nq2 = self.critic_target(batch["next_obs"], next_actions)
            next_q = mx.minimum(nq1, nq2) - ent_coef * next_log_prob
            target_q = batch["rewards"].reshape(-1) + (1.0 - batch["dones"].reshape(-1)) * gamma * next_q
            target_q = mx.stop_gradient(target_q)

            def critic_loss_fn(model):
                q1, q2 = model(batch["obs"], batch["actions"])
                l1 = mx.mean((q1 - target_q) ** 2)
                l2 = mx.mean((q2 - target_q) ** 2)
                return 0.5 * (l1 + l2), (l1, l2, q1, q2)

            # value_and_grad needs scalar; split
            def critic_scalar(model):
                q1, q2 = model(batch["obs"], batch["actions"])
                return 0.5 * (mx.mean((q1 - target_q) ** 2) + mx.mean((q2 - target_q) ** 2))

            c_loss, c_grads = nn.value_and_grad(self.critic, critic_scalar)(self.critic)
            self.critic_opt.update(self.critic, c_grads)
            mx.eval(self.critic.parameters(), self.critic_opt.state)
            with_q1, with_q2 = self.critic(batch["obs"], batch["actions"])
            l1 = mx.mean((with_q1 - target_q) ** 2)
            l2 = mx.mean((with_q2 - target_q) ** 2)
            critic_s += to_float(c_loss)
            q1_s += to_float(l1)
            q2_s += to_float(l2)
            q_mean_s += to_float(mx.mean(mx.minimum(with_q1, with_q2)))

            def actor_loss_fn(model):
                actions_pi, log_prob = model.sample(batch["obs"])
                q1, q2 = self.critic(batch["obs"], actions_pi)
                return mx.mean(ent_coef * log_prob - mx.minimum(q1, q2))

            a_loss, a_grads = nn.value_and_grad(self.actor, actor_loss_fn)(self.actor)
            self.actor_opt.update(self.actor, a_grads)
            mx.eval(self.actor.parameters(), self.actor_opt.state)
            actor_s += to_float(a_loss)
            last_loss = to_float(a_loss)
            _, lp = self.actor.sample(batch["obs"])
            log_pi_s += to_float(mx.mean(lp))

            if gs % self.target_update_interval == 0:
                self.critic_target.update(
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), tau)
                )
                mx.eval(self.critic_target.parameters())

        inv = 1.0 / max(gradient_steps, 1)
        self._n_updates += gradient_steps
        alpha = ent_s * inv
        # SB3-compat + full table fields
        self.logger.record("train/loss/policy", actor_s * inv)
        self.logger.record("train/actor_loss", actor_s * inv)
        self.logger.record("train/critic_loss", critic_s * inv)
        self.logger.record("train/loss/q1", q1_s * inv)
        self.logger.record("train/loss/q2", q2_s * inv)
        self.logger.record("train/loss/alpha", ent_loss_s / max(n_ent, 1) if n_ent else 0.0)
        self.logger.record("train/policy/alpha", alpha)
        self.logger.record("train/ent_coef", alpha)
        self.logger.record("train/value/q_mean", q_mean_s * inv)
        self.logger.record("train/policy/log_pi_mean", log_pi_s * inv)
        self.logger.record("train/ent_coef_loss", ent_loss_s / max(n_ent, 1) if n_ent else 0.0)
        self.logger.record("train/learning_rate", self.learning_rate)
        self.logger.record("train/loss", last_loss)
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/explained_variance", 0.0)
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
        self._roll_rew_sum = self._roll_rew_sum * 0.0
        self._roll_len_sum = self._roll_len_sum * 0.0
        self._roll_ep_count = self._roll_ep_count * 0.0
        self.logger.dump(step=self.num_timesteps)



    def learn(self, total_timesteps: int, log_interval: int = 4, progress_bar: bool = False, **kwargs):
        self.start_time = time.time()
        self._num_timesteps_at_start = self.num_timesteps
        self._last_obs, _ = self.env.reset()
        iteration = 0
        # collect train_freq env-steps per env, then train
        steps_per_cycle = self.train_freq
        while self.num_timesteps < total_timesteps:
            for _ in range(steps_per_cycle):
                obs = self._last_obs
                action = self._sample_action(obs, random=self.num_timesteps < self.learning_starts)
                new_obs, rewards, dones, infos = self.env.step(action)
                self.num_timesteps += self.n_envs
                self._update_ep_stats(rewards, dones)
                timeouts = infos.get("timeouts", mx.zeros((self.n_envs,))) if isinstance(infos, dict) else mx.zeros(
                    (self.n_envs,)
                )
                self.replay.add(obs, new_obs, action, rewards, dones.astype(mx.float32), timeouts)
                self._last_obs = new_obs

            if self.num_timesteps >= self.learning_starts:
                gs = self.gradient_steps if self.gradient_steps > 0 else steps_per_cycle
                self.train(gs, self.batch_size)

            iteration += 1
            if log_interval and iteration % log_interval == 0:
                self.dump_logs(iteration)

        if self.logger._vals:
            self.dump_logs(iteration)
        self.logger.close()
        return self

    def predict(self, observation, deterministic: bool = False):
        if not isinstance(observation, mx.array):
            observation = mx.array(observation, dtype=mx.float32)
        if observation.ndim == 1:
            observation = observation[None]
        a_tanh, _ = self.actor.sample(observation, deterministic=deterministic)
        action = self._scale_action(a_tanh)
        mx.eval(action)
        return action, None
