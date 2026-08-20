"""Rollout progress CSV logging modes for PPO/SAC.

``completed_only`` (default)
    Always emit a progress row (train/time metrics every dump). Populate
    classic ``ep_*`` columns only when ≥1 episode finished since the last
    dump; otherwise leave those cells blank (no NaNs). Pass ``force=True``
    on the final dump to flush remaining train metrics.

``ongoing``
    Always emit a progress row. Writes always-defined ongoing / step
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
    """Deprecated: rows are never skipped. Kept for call-site compatibility."""
    del n_ep, force
    return False
