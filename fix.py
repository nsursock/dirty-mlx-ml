import mlx.core as mx
from typing import Any, Tuple, Union

class VecNormalize:
    def __init__(self, obs: Any, rew: Any, obs_key: str = 'obs', rew_key: str = 'reward'):
        self.obs_key = obs_key
        self.rew_key = rew_key
        
        # Determine dimensionality for stats initialization
        def get_dim(arr):
            if hasattr(arr, 'shape'):
                return arr.shape[0] if len(arr.shape) > 1 else 1
            return 1
            
        dim = get_dim(obs)
        
        self.obs_mean = mx.zeros((dim,))
        self.obs_std = mx.ones((dim,))
        self.rew_mean = mx.zeros((dim,))
        self.rew_std = mx.ones((dim,))
        
        self._obs = obs
        self._rew = rew
        self._count = 0

    def update(self, obs: Any, rew: Any, n: int = 1):
        """Update running statistics."""
        # Increment count
        self._count += n
        
        # Accumulate mean
        count_factor = self._count if self._count > 0 else 1
        self.obs_mean = (self.obs_mean * (count_factor - 1) + obs) / count_factor
        self.rew_mean = (self.rew_mean * (count_factor - 1) + rew) / count_factor
        
        # Calculate std from running mean diffs
        obs_diff = obs - self.obs_mean
        rew_diff = rew - self.rew_mean
        
        self.obs_std = ((self.obs_std ** 2) * (count_factor - 1) + (obs_diff ** 2)) ** 0.5 / count_factor
        self.rew_std = ((self.rew_std ** 2) * (count_factor - 1) + (rew_diff ** 2)) ** 0.5 / count_factor

    def __call__(self, batch):
        # Handle dict or batch input
        if isinstance(batch, dict):
            norm_obs = (batch[self.obs_key] - self.obs_mean) / (self.obs_std + 1e-4)
            norm_rew = (batch[self.rew_key] - self.rew_mean) / (self.rew_std + 1e-4)
            return {self.obs_key: norm_obs, self.rew_key: norm_rew}
        else:
            # Fallback for raw arrays
            norm_obs = (batch - self.obs_mean) / (self.obs_std + 1e-4)
            norm_rew = (batch - self.rew_mean) / (self.rew_std + 1e-4)
            return {self.obs_key: norm_obs, self.rew_key: norm_rew}