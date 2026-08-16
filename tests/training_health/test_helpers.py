"""Unit tests for the validation helpers on synthetic data (no training)."""
import numpy as np
import pandas as pd
import pytest

from dirty_mlx_ml.reinforcement.validation import CheckSpec, load_csv, run_spec, validate_sac


def _spec(col, kind, **params):
    return CheckSpec("t", col, kind, params=params)


def test_detects_inf():
    df = pd.DataFrame({"train/loss": [1.0, 2.0, np.inf]})
    c = run_spec(df, _spec("train/loss", "finite"))
    assert not c.ok


def test_detects_midrun_nan():
    df = pd.DataFrame({"train/loss": [1.0, np.nan, 3.0]})
    c = run_spec(df, _spec("train/loss", "finite"))
    assert not c.ok


def test_allows_warmup_nan_prefix():
    df = pd.DataFrame({"train/loss": [np.nan, np.nan, 1.0, 2.0]})
    c = run_spec(df, _spec("train/loss", "finite"))
    assert c.ok


def test_allows_rollout_nan_anywhere():
    df = pd.DataFrame({"rollout/ep_rew_mean": [1.0, np.nan, 3.0]})
    c = run_spec(df, _spec("rollout/ep_rew_mean", "finite", allow_nan=True))
    assert c.ok


def test_detects_non_monotonic_timesteps():
    df = pd.DataFrame({"time/total_timesteps": [10.0, 20.0, 15.0]})
    c = run_spec(df, _spec("time/total_timesteps", "monotonic"))
    assert not c.ok


def test_detects_out_of_bounds_clip_fraction():
    df = pd.DataFrame({"train/clip_fraction": [0.1, 0.5, 1.5]})
    c = run_spec(df, _spec("train/clip_fraction", "bounded", lo=0.0, hi=1.0))
    assert not c.ok


def test_detects_alpha_non_positive():
    df = pd.DataFrame({"train/ent_coef": [0.5, 0.0, -0.1]})
    c = run_spec(df, _spec("train/ent_coef", "gt", lo=0.0))
    assert not c.ok


def test_detects_log_prob_singularity():
    df = pd.DataFrame({"train/policy/log_pi_mean": [-0.5, -200.0]})
    c = run_spec(df, _spec("train/policy/log_pi_mean", "bounded", lo=-50.0, hi=20.0))
    assert not c.ok


def test_load_csv_empty_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("a,b\n")
    with pytest.raises(ValueError):
        load_csv(str(p))


def test_q_gap_divergence_detected():
    cols = {c: [0.0, 1.0] for c in ["train/loss/q1", "train/loss/q2"]}
    cols["train/loss/q1"] = [0.0, 5.0]
    cols["train/loss/q2"] = [0.0, 0.0]
    df = pd.DataFrame(cols)
    checks = validate_sac(df, csv_name="syn")
    gap = next(c for c in checks if c.id == "train:q1_q2_gap")
    assert not gap.ok
