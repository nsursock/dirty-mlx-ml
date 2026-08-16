import mlx.core as mx


def _gae(rewards, values, episode_starts, last_values, dones, gamma, gae_lambda):
    """Vectorized-enough GAE; reverse scan in MLX arrays (no numpy)."""
    n_steps = int(rewards.shape[0])
    adv = mx.zeros_like(rewards)
    last_gae = mx.zeros((rewards.shape[1],))
    next_values = last_values
    next_non_terminal = 1.0 - dones
    # reverse step by step (still on device; fused via mx.eval at end)
    adv_list = [None] * n_steps
    for step in range(n_steps - 1, -1, -1):
        if step != n_steps - 1:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        adv_list[step] = last_gae
    return mx.stack(adv_list)


class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, act_dim: int, gamma: float, gae_lambda: float):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        s, e = self.n_steps, self.n_envs
        self.obs = mx.zeros((s, e, self.obs_dim))
        self.actions = mx.zeros((s, e, self.act_dim))
        self.rewards = mx.zeros((s, e))
        self.episode_starts = mx.zeros((s, e))
        self.values = mx.zeros((s, e))
        self.log_probs = mx.zeros((s, e))
        self.advantages = mx.zeros((s, e))
        self.returns = mx.zeros((s, e))
        self.pos = 0
        self.full = False
        self._flat = False

    def add(self, obs, actions, rewards, episode_starts, values, log_probs):
        p = self.pos
        self.obs[p] = obs
        self.actions[p] = actions
        self.rewards[p] = rewards
        self.episode_starts[p] = episode_starts
        self.values[p] = values
        self.log_probs[p] = log_probs
        self.pos = p + 1
        if self.pos == self.n_steps:
            self.full = True

    def compute_returns_and_advantage(self, last_values, dones):
        self.advantages = _gae(
            self.rewards, self.values, self.episode_starts, last_values, dones, self.gamma, self.gae_lambda
        )
        self.returns = self.advantages + self.values

    def get(self, batch_size: int):
        assert self.full
        n = self.n_steps * self.n_envs
        if not self._flat:
            self.obs = self.obs.swapaxes(0, 1).reshape(n, self.obs_dim)
            self.actions = self.actions.swapaxes(0, 1).reshape(n, self.act_dim)
            self.values = self.values.swapaxes(0, 1).reshape(n)
            self.log_probs = self.log_probs.swapaxes(0, 1).reshape(n)
            self.advantages = self.advantages.swapaxes(0, 1).reshape(n)
            self.returns = self.returns.swapaxes(0, 1).reshape(n)
            self._flat = True
        idx = mx.random.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            yield {
                "obs": self.obs[b],
                "actions": self.actions[b],
                "old_values": self.values[b],
                "old_log_prob": self.log_probs[b],
                "advantages": self.advantages[b],
                "returns": self.returns[b],
            }


class ReplayBuffer:
    def __init__(self, buffer_size: int, n_envs: int, obs_dim: int, act_dim: int):
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        s, e = self.buffer_size, n_envs
        self.obs = mx.zeros((s, e, obs_dim))
        self.next_obs = mx.zeros((s, e, obs_dim))
        self.actions = mx.zeros((s, e, act_dim))
        self.rewards = mx.zeros((s, e))
        self.dones = mx.zeros((s, e))
        self.timeouts = mx.zeros((s, e))
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, actions, rewards, dones, timeouts=None):
        p = self.pos
        self.obs[p] = obs
        self.next_obs[p] = next_obs
        self.actions[p] = actions
        self.rewards[p] = rewards
        self.dones[p] = dones
        if timeouts is not None:
            self.timeouts[p] = timeouts
        self.pos = p + 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def size(self):
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size: int):
        upper = self.buffer_size if self.full else max(self.pos, 1)
        bi = mx.random.randint(0, upper, (batch_size,))
        ei = mx.random.randint(0, self.n_envs, (batch_size,))
        dones = self.dones[bi, ei] * (1.0 - self.timeouts[bi, ei])
        return {
            "obs": self.obs[bi, ei],
            "next_obs": self.next_obs[bi, ei],
            "actions": self.actions[bi, ei],
            "rewards": self.rewards[bi, ei].reshape(-1, 1),
            "dones": dones.reshape(-1, 1),
        }
