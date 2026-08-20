"""Rollout progress CSV logging modes for PPO/SAC.

``completed_only`` (default)
    Emit a progress row only when at least one episode finished since the last
    dump. Completed-episode means are always defined on every written row.
    Pass ``force=True`` on the final dump to flush train metrics even if no
    episode completed.

``ongoing``
    Emit a progress row on every dump. Writes always-defined ongoing / step
    metrics, and fills classic ``ep_*_mean`` from completed episodes when
    available, otherwise from the in-progress episode accumulators.
"""

from __future__ import annotations

COMPLETED_ONLY = "completed_only"
ONGOING = "ongoing"
ROLLOUT_LOG_MODES = (COMPLETED_ONLY, ONGOING)


def normalize_rollout_log_mode(mode: str | None) -> str:
    m = COMPLETED_ONLY if mode is None else str(mode).strip().lower()
    if m not in ROLLOUT_LOG_MODES:
        raise ValueError(
            f"unknown rollout_log_mode={mode!r}; want one of {ROLLOUT_LOG_MODES}"
        )
    return m


def should_skip_completed_only_dump(n_ep: float, force: bool) -> bool:
    """Return True when ``completed_only`` mode should skip this CSV row."""
    return (not force) and float(n_ep) <= 0.0
