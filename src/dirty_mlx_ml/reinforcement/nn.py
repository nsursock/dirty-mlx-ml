import math

import mlx.core as mx
import mlx.nn as nn


def orthogonal_init(shape, gain=1.0):
    flat_in = shape[0]
    flat_out = math.prod(shape[1:]) if len(shape) > 1 else 1
    a = mx.random.normal((max(flat_in, flat_out), min(flat_in, flat_out)))
    q, r = mx.linalg.qr(a)
    d = mx.sign(mx.diag(r))
    q = q * d
    if flat_in < flat_out:
        q = q.T
    # q shape (flat_in, flat_out) ideally
    if q.shape[0] != flat_in:
        q = q.T
    q = q.reshape(shape) * gain
    return q.astype(mx.float32)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(64, 64), act="tanh"):
        super().__init__()
        self.act_name = act
        dims = [in_dim, *list(hidden), out_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            gain = math.sqrt(2) if i < len(dims) - 2 else 1.0
            try:
                lin.weight = orthogonal_init((dims[i + 1], dims[i]), gain=gain)
            except Exception:
                pass
            lin.bias = mx.zeros((dims[i + 1],))
            self.layers.append(lin)

    def _act(self, x):
        if self.act_name == "relu":
            return nn.relu(x)
        return mx.tanh(x)

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self._act(x)
        return x


class ActorCriticDiscrete(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(64, 64)):
        super().__init__()
        self.pi = MLP(obs_dim, n_actions, hidden, act="tanh")
        self.vf = MLP(obs_dim, 1, hidden, act="tanh")

    def forward(self, obs):
        logits = self.pi(obs)
        value = self.vf(obs).squeeze(-1)
        return logits, value

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        if deterministic:
            action = mx.argmax(logits, axis=-1)
            log_prob = mx.zeros(obs.shape[0])
        else:
            action = mx.random.categorical(logits)
            log_prob = -nn.losses.cross_entropy(logits, action, reduction="none")
        return action, value, log_prob

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        log_prob = -nn.losses.cross_entropy(logits, actions.astype(mx.int32), reduction="none")
        probs = mx.softmax(logits, axis=-1)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8), axis=-1)
        return value, log_prob, entropy


class ActorCriticContinuous(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64), log_std_init=0.0):
        super().__init__()
        self.pi = MLP(obs_dim, act_dim, hidden, act="tanh")
        self.vf = MLP(obs_dim, 1, hidden, act="tanh")
        self.log_std = mx.full((act_dim,), float(log_std_init))

    def forward(self, obs):
        mean = self.pi(obs)
        value = self.vf(obs).squeeze(-1)
        return mean, value

    def get_action(self, obs, deterministic=False):
        mean, value = self.forward(obs)
        std = mx.exp(self.log_std)
        if deterministic:
            action = mean
        else:
            action = mean + mx.random.normal(mean.shape) * std
        log_prob = _gauss_log_prob(action, mean, self.log_std)
        return action, value, log_prob

    def evaluate(self, obs, actions):
        mean, value = self.forward(obs)
        log_prob = _gauss_log_prob(actions, mean, self.log_std)
        entropy = mx.sum(0.5 + 0.5 * math.log(2 * math.pi) + self.log_std, axis=-1)
        if entropy.ndim == 0:
            entropy = mx.broadcast_to(entropy, value.shape)
        return value, log_prob, entropy


def _gauss_log_prob(x, mean, log_std):
    var = mx.exp(2.0 * log_std)
    return mx.sum(-((x - mean) ** 2) / (2.0 * var) - log_std - 0.5 * math.log(2 * math.pi), axis=-1)


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SACActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.trunk = MLP(obs_dim, hidden[-1], hidden[:-1] if len(hidden) > 1 else (), act="relu")
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std_h = nn.Linear(hidden[-1], act_dim)

    def __call__(self, obs):
        h = self.trunk(obs)
        if len(self.trunk.layers) == 1:
            h = nn.relu(h)
        mu = self.mu(h)
        log_std = mx.clip(self.log_std_h(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs, deterministic=False):
        mu, log_std = self(obs)
        std = mx.exp(log_std)
        if deterministic:
            z = mu
        else:
            z = mu + std * mx.random.normal(mu.shape)
        action = mx.tanh(z)
        log_prob = _gauss_log_prob(z, mu, log_std) - mx.sum(mx.log(1.0 - action**2 + 1e-6), axis=-1)
        return action, log_prob


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1, hidden, act="relu")

    def __call__(self, obs, act):
        x = mx.concatenate([obs, act], axis=-1)
        return self.net(x).squeeze(-1)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = QNet(obs_dim, act_dim, hidden)
        self.q2 = QNet(obs_dim, act_dim, hidden)

    def __call__(self, obs, act):
        return self.q1(obs, act), self.q2(obs, act)
