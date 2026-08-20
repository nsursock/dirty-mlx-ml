"""Tests for rollout progress CSV logging modes."""

from __future__ import annotations

import pytest

from dirty_mlx_ml.reinforcement.rollout_logging import (
    COMPLETED_ONLY,
    ONGOING,
    normalize_rollout_log_mode,
    should_skip_completed_only_dump,
)


def test_normalize_default_and_aliases():
    assert normalize_rollout_log_mode(None) == COMPLETED_ONLY
    assert normalize_rollout_log_mode("completed_only") == COMPLETED_ONLY
    assert normalize_rollout_log_mode("ONGOING") == ONGOING


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError, match="unknown rollout_log_mode"):
        normalize_rollout_log_mode("carry_forward")


def test_completed_only_never_skips_rows():
    # Rows always dump; mode only gates whether ep_* are populated.
    assert should_skip_completed_only_dump(0.0, force=False) is False
    assert should_skip_completed_only_dump(0.0, force=True) is False
    assert should_skip_completed_only_dump(3.0, force=False) is False
