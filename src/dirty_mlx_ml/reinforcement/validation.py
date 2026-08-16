"""Health validation for PPO/SAC progress CSVs.

Reusable, dependency-light (pandas + numpy only) so it can be shared between the
pytest suite (``tests/training_health``) and a standalone script
(``bench/reinforcement/validate_csvs.py``).

Design (from the feedback/validation consensus):

  * RL metrics are noisy -> **no** step-to-step monotonic-decrease assertions.
  * Instead: bounded-with-tolerance, trend-over-a-smoothed-window, and
    rolling z-score spike detection.
  * Severity tiers: ``hard`` (definitely broken), ``warn`` (possible
    instability), ``solve`` (task actually solved -- disabled by default).
  * Project-specific guards baked in: tanh-squash log-prob bounds, ``alpha > 0``,
    Q1/Q2 divergence gap.

Usage
-----
Script (validate both CSVs in a folder)::

    python bench/reinforcement/validate_csvs.py logs/sac
    python bench/reinforcement/validate_csvs.py logs/ppo/ppo_progress.csv

Library (pytest or ad-hoc)::

    from dirty_mlx_ml.reinforcement.validation import validate_ppo, validate_sac
    df = pd.read_csv("logs/ppo/ppo_progress.csv")
    for c in validate_ppo(df, csv_name="ppo"):
        print(c)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column schemas (must match CSVLogger + ppo.py / sac.py)
# ---------------------------------------------------------------------------

PPO_COLUMNS = [
    "time/fps",
    "time/iterations",
    "time/time_elapsed",
    "time/total_timesteps",
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "rollout/success_rate",
    "train/approx_kl",
    "train/clip_fraction",
    "train/clip_range",
    "train/entropy_loss",
    "train/explained_variance",
    "train/learning_rate",
    "train/loss",
    "train/n_updates",
    "train/policy_gradient_loss",
    "train/std",
    "train/value_loss",
]

SAC_COLUMNS = [
    "time/fps",
    "time/iterations",
    "time/time_elapsed",
    "time/total_timesteps",
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "rollout/success_rate",
    "train/actor_loss",
    "train/critic_loss",
    "train/ent_coef",
    "train/ent_coef_loss",
    "train/learning_rate",
    "train/loss",
    "train/loss/alpha",
    "train/loss/policy",
    "train/loss/q1",
    "train/loss/q2",
    "train/n_updates",
    "train/policy/alpha",
    "train/policy/log_pi_mean",
    "train/value/q_mean",
]


# ---------------------------------------------------------------------------
# Result / spec types
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One validation result."""

    id: str
    column: str
    kind: str
    severity: str  # "hard" | "warn" | "solve"
    ok: bool
    message: str


@dataclass(frozen=True)
class CheckSpec:
    """Static description of a single check (parametrized into pytest)."""

    id: str
    column: str
    kind: str
    severity: str = "hard"
    params: dict = dc_field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared time/rollout specs (identical for PPO and SAC)
# ---------------------------------------------------------------------------


def _common_specs() -> List[CheckSpec]:
    return [
        CheckSpec("time/fps:finite", "time/fps", "finite"),
        CheckSpec("time/fps:ge0", "time/fps", "ge", params={"lo": 0.0}),
        CheckSpec("time/iterations:ge0", "time/iterations", "ge", params={"lo": 0.0}),
        CheckSpec("time/iterations:monotonic", "time/iterations", "monotonic"),
        CheckSpec("time/time_elapsed:ge0", "time/time_elapsed", "ge", params={"lo": 0.0}),
        CheckSpec("time/time_elapsed:monotonic", "time/time_elapsed", "monotonic"),
        CheckSpec("time/total_timesteps:ge0", "time/total_timesteps", "ge", params={"lo": 0.0}),
        CheckSpec("time/total_timesteps:monotonic", "time/total_timesteps", "monotonic"),
        CheckSpec(
            "rollout/ep_len_mean:finite",
            "rollout/ep_len_mean",
            "finite",
            params={"allow_nan": True},
        ),
        CheckSpec(
            "rollout/ep_len_mean:ge0",
            "rollout/ep_len_mean",
            "ge",
            params={"lo": 0.0, "allow_nan": True},
        ),
        CheckSpec(
            "rollout/ep_rew_mean:finite",
            "rollout/ep_rew_mean",
            "finite",
            params={"allow_nan": True},
        ),
        CheckSpec(
            "rollout/success_rate:bounded",
            "rollout/success_rate",
            "bounded",
            params={"lo": 0.0, "hi": 1.0},
        ),
    ]


# ---------------------------------------------------------------------------
# Per-algorithm specs
# ---------------------------------------------------------------------------

PPO_SPECS: List[CheckSpec] = _common_specs() + [
    CheckSpec("train/approx_kl:finite", "train/approx_kl", "finite"),
    CheckSpec("train/approx_kl:ge0", "train/approx_kl", "ge", params={"lo": -1e-6}),
    CheckSpec(
        "train/approx_kl:bounded",
        "train/approx_kl",
        "bounded",
        params={"lo": -1e-6, "hi": 0.25, "tol_frac": 0.02},
    ),
    CheckSpec(
        "train/approx_kl:p95",
        "train/approx_kl",
        "quantile_le",
        severity="warn",
        params={"q": 0.95, "hi": 0.10},
    ),
    CheckSpec(
        "train/approx_kl:spikes",
        "train/approx_kl",
        "spikes",
        severity="warn",
        params={"window": 20, "z_thresh": 5.0},
    ),
    CheckSpec("train/clip_fraction:finite", "train/clip_fraction", "finite"),
    CheckSpec(
        "train/clip_fraction:bounded",
        "train/clip_fraction",
        "bounded",
        params={"lo": 0.0, "hi": 1.0},
    ),
    CheckSpec("train/clip_range:finite", "train/clip_range", "finite"),
    CheckSpec("train/clip_range:bounded", "train/clip_range", "bounded", params={"lo": 0.0, "hi": 1.0}),
    CheckSpec("train/entropy_loss:finite", "train/entropy_loss", "finite"),
    CheckSpec(
        "train/entropy_loss:not_collapsed",
        "train/entropy_loss",
        "tail_mean_le",
        severity="warn",
        params={"frac": 0.25, "threshold": -1e-3},
    ),
    CheckSpec("train/explained_variance:finite", "train/explained_variance", "finite"),
    CheckSpec(
        "train/explained_variance:le1",
        "train/explained_variance",
        "le",
        severity="warn",
        params={"hi": 1.01},
    ),
    CheckSpec(
        "train/explained_variance:trend",
        "train/explained_variance",
        "trend",
        severity="warn",
        params={"direction": "increasing", "window": 20, "min_slope": 0.0},
    ),
    CheckSpec("train/learning_rate:finite", "train/learning_rate", "finite"),
    CheckSpec(
        "train/learning_rate:bounded",
        "train/learning_rate",
        "bounded",
        params={"lo": 1e-6, "hi": 1.0},
    ),
    CheckSpec("train/loss:finite", "train/loss", "finite"),
    CheckSpec("train/loss:spikes", "train/loss", "spikes", severity="warn", params={"window": 20, "z_thresh": 5.0}),
    CheckSpec("train/n_updates:finite", "train/n_updates", "finite"),
    CheckSpec("train/n_updates:ge0", "train/n_updates", "ge", params={"lo": 0.0}),
    CheckSpec("train/n_updates:monotonic", "train/n_updates", "monotonic"),
    CheckSpec("train/n_updates:integer", "train/n_updates", "integer", severity="warn"),
    CheckSpec("train/policy_gradient_loss:finite", "train/policy_gradient_loss", "finite"),
    CheckSpec(
        "train/policy_gradient_loss:spikes",
        "train/policy_gradient_loss",
        "spikes",
        severity="warn",
        params={"window": 20, "z_thresh": 5.0},
    ),
    CheckSpec("train/std:finite", "train/std", "finite"),
    CheckSpec("train/std:ge0", "train/std", "ge", params={"lo": 0.0}),
    CheckSpec("train/value_loss:finite", "train/value_loss", "finite"),
    CheckSpec("train/value_loss:ge0", "train/value_loss", "ge", params={"lo": 0.0}),
    CheckSpec(
        "train/value_loss:spikes",
        "train/value_loss",
        "spikes",
        severity="warn",
        params={"window": 20, "z_thresh": 5.0},
    ),
]

SAC_SPECS: List[CheckSpec] = _common_specs() + [
    CheckSpec("train/actor_loss:finite", "train/actor_loss", "finite"),
    CheckSpec("train/actor_loss:spikes", "train/actor_loss", "spikes", severity="warn", params={"window": 20, "z_thresh": 5.0}),
    CheckSpec("train/critic_loss:finite", "train/critic_loss", "finite"),
    CheckSpec("train/critic_loss:spikes", "train/critic_loss", "spikes", severity="warn", params={"window": 20, "z_thresh": 5.0}),
    CheckSpec("train/ent_coef:finite", "train/ent_coef", "finite"),
    CheckSpec("train/ent_coef:gt0", "train/ent_coef", "gt", params={"lo": 0.0}),
    CheckSpec("train/ent_coef_loss:finite", "train/ent_coef_loss", "finite"),
    CheckSpec("train/ent_coef_loss:spikes", "train/ent_coef_loss", "spikes", severity="warn", params={"window": 20, "z_thresh": 5.0}),
    CheckSpec("train/learning_rate:finite", "train/learning_rate", "finite"),
    CheckSpec("train/learning_rate:bounded", "train/learning_rate", "bounded", params={"lo": 1e-6, "hi": 1.0}),
    CheckSpec("train/loss:finite", "train/loss", "finite"),
    CheckSpec("train/loss:spikes", "train/loss", "spikes", severity="warn", params={"window": 20, "z_thresh": 5.0}),
    CheckSpec("train/loss/alpha:finite", "train/loss/alpha", "finite"),
    CheckSpec("train/loss/alpha:equals_ent_coef_loss", "train/loss/alpha", "equal_to", params={"other": "train/ent_coef_loss"}),
    CheckSpec("train/loss/policy:finite", "train/loss/policy", "finite"),
    CheckSpec("train/loss/policy:equals_actor_loss", "train/loss/policy", "equal_to", params={"other": "train/actor_loss"}),
    CheckSpec("train/loss/q1:finite", "train/loss/q1", "finite"),
    CheckSpec("train/loss/q1:spikes", "train/loss/q1", "spikes", severity="warn", params={"window": 20, "z_thresh": 5.0}),
    CheckSpec("train/loss/q2:finite", "train/loss/q2", "finite"),
    CheckSpec("train/loss/q2:spikes", "train/loss/q2", "spikes", severity="warn", params={"window": 20, "z_thresh": 5.0}),
    CheckSpec("train/n_updates:finite", "train/n_updates", "finite"),
    CheckSpec("train/n_updates:ge0", "train/n_updates", "ge", params={"lo": 0.0}),
    CheckSpec("train/n_updates:monotonic", "train/n_updates", "monotonic"),
    CheckSpec("train/n_updates:integer", "train/n_updates", "integer", severity="warn"),
    CheckSpec("train/policy/alpha:gt0", "train/policy/alpha", "gt", params={"lo": 0.0}),
    CheckSpec("train/policy/alpha:equals_ent_coef", "train/policy/alpha", "equal_to", params={"other": "train/ent_coef"}),
    CheckSpec("train/policy/log_pi_mean:finite", "train/policy/log_pi_mean", "finite"),
    CheckSpec(
        "train/policy/log_pi_mean:bounded",
        "train/policy/log_pi_mean",
        "bounded",
        params={"lo": -50.0, "hi": 20.0},
    ),
    CheckSpec("train/value/q_mean:finite", "train/value/q_mean", "finite"),
    CheckSpec("train/value/q_mean:spikes", "train/value/q_mean", "spikes", severity="warn", params={"window": 20, "z_thresh": 5.0}),
]

# Optional task-solve specs (env-specific, disabled unless ``solve=True``).
PPO_SOLVE_SPECS: List[CheckSpec] = [
    CheckSpec("rollout/ep_len_mean:solved", "rollout/ep_len_mean", "tail_mean_ge", severity="solve", params={"frac": 0.5, "threshold": 195.0}),
]
SAC_SOLVE_SPECS: List[CheckSpec] = [
    CheckSpec("rollout/ep_rew_mean:solved", "rollout/ep_rew_mean", "tail_mean_ge", severity="solve", params={"frac": 0.5, "threshold": -200.0}),
]


# ---------------------------------------------------------------------------
# Check implementations (all return (ok, message))
# ---------------------------------------------------------------------------


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _check_finite(s, name, p):
    x = _num(s)
    inf_mask = np.isinf(x.to_numpy(dtype=float))
    n_inf = int(inf_mask.sum())
    if n_inf:
        return False, f"{name}: {n_inf} Inf value(s)"
    non_null = x.dropna()
    if non_null.empty:
        if p.get("allow_nan", False):
            return True, f"{name}: no completed episodes logged (all NaN)"
        return False, f"{name}: no finite values to validate"
    n_nan = int(x.isna().sum())
    if n_nan and not p.get("allow_nan", False):
        first = x.first_valid_index()
        if x.loc[first:].isna().any():
            return False, f"{name}: NaN after first valid value (mid-run blowup) — {int(x.loc[first:].isna().sum())} NaN(s)"
    return True, f"{name}: finite across {len(non_null)} row(s)"


def _check_ge(s, name, p):
    x = _num(s).dropna()
    if x.empty:
        return True, f"{name}: no data (skipped)"
    lo = p["lo"]
    bad = int((x < lo).sum())
    ok = bad == 0
    msg = f"{name}: all >= {lo}" if ok else f"{name}: {bad} value(s) < {lo} (min={x.min():.6g})"
    return ok, msg


def _check_gt(s, name, p):
    x = _num(s).dropna()
    if x.empty:
        return True, f"{name}: no data (skipped)"
    lo = p["lo"]
    bad = int((x <= lo).sum())
    ok = bad == 0
    msg = f"{name}: all > {lo}" if ok else f"{name}: {bad} value(s) <= {lo} (min={x.min():.6g})"
    return ok, msg


def _check_le(s, name, p):
    x = _num(s).dropna()
    if x.empty:
        return True, f"{name}: no data (skipped)"
    hi = p["hi"]
    bad = int((x > hi).sum())
    ok = bad == 0
    msg = f"{name}: all <= {hi}" if ok else f"{name}: {bad} value(s) > {hi} (max={x.max():.6g})"
    return ok, msg


def _check_bounded(s, name, p):
    x = _num(s).dropna()
    if x.empty:
        return True, f"{name}: no data (skipped)"
    lo, hi = p["lo"], p["hi"]
    tol = p.get("tol_frac", 0.0)
    frac = float(((x < lo) | (x > hi)).mean())
    ok = frac <= tol
    msg = (
        f"{name}: {frac:.1%} outside [{lo}, {hi}] (tol {tol:.1%})"
        if ok
        else f"{name}: {frac:.1%} outside [{lo}, {hi}] (tol {tol:.1%}) — worst {x[(x < lo) | (x > hi)].abs().max():.6g}"
    )
    return ok, msg


def _check_monotonic(s, name, p):
    x = _num(s).dropna()
    if len(x) < 2:
        return True, f"{name}: too few rows (skipped)"
    d = x.diff().dropna()
    n_bad = int((d < 0).sum())
    ok = n_bad == 0
    msg = f"{name}: monotonic non-decreasing" if ok else f"{name}: {n_bad} decrease(s) — restarted/concatenated log?"
    return ok, msg


def _check_integer(s, name, p):
    x = _num(s).dropna()
    if x.empty:
        return True, f"{name}: no data (skipped)"
    atol = p.get("atol", 1e-3)
    arr = x.to_numpy(dtype=float)
    ok = bool(np.allclose(arr, np.round(arr), atol=atol))
    msg = f"{name}: integral" if ok else f"{name}: non-integer values (e.g. {x[~np.isclose(arr, np.round(arr), atol=atol)].head(3).tolist()})"
    return ok, msg


def _check_spikes(s, name, p):
    x = _num(s).dropna().reset_index(drop=True)
    window = p.get("window", 20)
    z_thresh = p.get("z_thresh", 5.0)
    if len(x) < max(3, window // 2):
        return True, f"{name}: too few rows for spike detection (skipped)"
    roll_mean = x.rolling(window, min_periods=max(3, window // 2)).mean()
    roll_std = x.rolling(window, min_periods=max(3, window // 2)).std().replace(0, np.nan)
    z = (x - roll_mean).abs() / roll_std
    spikes = z[z > z_thresh]
    ok = spikes.empty
    msg = (
        f"{name}: no spikes (|z| > {z_thresh})"
        if ok
        else f"{name}: {len(spikes)} spike(s) at idx {spikes.index.tolist()[:5]}"
    )
    return ok, msg


def _check_trend(s, name, p):
    x = _num(s).dropna()
    window = p.get("window", 20)
    direction = p.get("direction", "increasing")
    min_slope = p.get("min_slope", 0.0)
    sm = x.rolling(window, min_periods=max(5, window // 5)).mean().dropna()
    if len(sm) < 2:
        return True, f"{name}: too few rows for trend (skipped)"
    slope = float(np.polyfit(np.arange(len(sm)), sm.to_numpy(dtype=float), 1)[0])
    if direction == "increasing":
        ok = slope >= min_slope
    elif direction == "decreasing":
        ok = slope <= -min_slope
    else:
        raise ValueError(f"bad direction {direction!r}")
    msg = f"{name}: {direction} trend (slope={slope:.6g})" if ok else f"{name}: expected {direction}, slope={slope:.6g}"
    return ok, msg


def _check_quantile_le(s, name, p):
    x = _num(s).dropna()
    if x.empty:
        return True, f"{name}: no data (skipped)"
    q, hi = p["q"], p["hi"]
    val = float(x.quantile(q))
    ok = val <= hi
    msg = f"{name}: {q:.0%}-quantile {val:.6g} <= {hi}" if ok else f"{name}: {q:.0%}-quantile {val:.6g} > {hi}"
    return ok, msg


def _tail(s, p):
    x = _num(s).dropna()
    frac = p.get("frac", 0.25)
    n = max(1, int(len(x) * frac))
    return x.tail(n)


def _check_tail_mean_ge(s, name, p):
    t = _tail(s, p)
    if t.empty:
        return True, f"{name}: no data (skipped)"
    val = float(t.mean())
    th = p["threshold"]
    ok = val >= th
    msg = f"{name}: tail-mean {val:.3g} >= {th}" if ok else f"{name}: tail-mean {val:.3g} < {th}"
    return ok, msg


def _check_tail_mean_le(s, name, p):
    t = _tail(s, p)
    if t.empty:
        return True, f"{name}: no data (skipped)"
    val = float(t.mean())
    th = p["threshold"]
    ok = val <= th
    msg = f"{name}: tail-mean {val:.3g} <= {th}" if ok else f"{name}: tail-mean {val:.3g} > {th} (entropy collapsed?)"
    return ok, msg


_DISPATCH = {
    "finite": _check_finite,
    "ge": _check_ge,
    "gt": _check_gt,
    "le": _check_le,
    "bounded": _check_bounded,
    "monotonic": _check_monotonic,
    "integer": _check_integer,
    "spikes": _check_spikes,
    "trend": _check_trend,
    "quantile_le": _check_quantile_le,
    "tail_mean_ge": _check_tail_mean_ge,
    "tail_mean_le": _check_tail_mean_le,
}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path}: empty CSV (0 rows)")
    return df


def columns_present(df: pd.DataFrame, required: List[str], csv_name: str) -> Check:
    missing = [c for c in required if c not in df.columns]
    ok = not missing
    msg = (
        f"{csv_name}: all {len(required)} required columns present"
        if ok
        else f"{csv_name}: missing columns {missing}"
    )
    return Check("schema:columns", "", "columns_present", "hard", ok, msg)


def run_spec(df: pd.DataFrame, spec: CheckSpec, csv_name: str = "") -> Check:
    name = f"{csv_name}:{spec.column}".strip(":")

    if spec.column and spec.column not in df.columns:
        return Check(spec.id, spec.column, spec.kind, spec.severity, True, f"{name}: skipped (missing column, covered by schema check)")

    if spec.kind == "equal_to":
        other = spec.params["other"]
        if other not in df.columns:
            return Check(spec.id, spec.column, spec.kind, spec.severity, False, f"{name}: comparison column '{other}' missing")
        a = _num(df[spec.column]).to_numpy(dtype=float)
        b = _num(df[other]).to_numpy(dtype=float)
        rtol = spec.params.get("rtol", 1e-3)
        ok = bool(np.allclose(a, b, rtol=rtol, atol=1e-6, equal_nan=True))
        msg = f"{name} == {other}" if ok else f"{name} != {other} (duplicate metrics drifted)"
        return Check(spec.id, spec.column, spec.kind, spec.severity, ok, msg)

    series = df[spec.column]
    fn = _DISPATCH[spec.kind]
    ok, msg = fn(series, name, spec.params)
    return Check(spec.id, spec.column, spec.kind, spec.severity, ok, msg)


def validate_ppo(df: pd.DataFrame, csv_name: str = "", solve: bool = False) -> List[Check]:
    checks = [columns_present(df, PPO_COLUMNS, csv_name)]
    checks += [run_spec(df, s, csv_name) for s in PPO_SPECS]
    if solve:
        checks += [run_spec(df, s, csv_name) for s in PPO_SOLVE_SPECS]
    return checks


def validate_sac(df: pd.DataFrame, csv_name: str = "", solve: bool = False) -> List[Check]:
    checks = [columns_present(df, SAC_COLUMNS, csv_name)]
    checks += [run_spec(df, s, csv_name) for s in SAC_SPECS]
    if solve:
        checks += [run_spec(df, s, csv_name) for s in SAC_SOLVE_SPECS]
    # Q1/Q2 overestimation-gap divergence (project-specific failure mode)
    checks.append(_q_gap(df, csv_name))
    return checks


def _q_gap(df: pd.DataFrame, csv_name: str) -> Check:
    cid = "train:q1_q2_gap"
    for col in ("train/loss/q1", "train/loss/q2"):
        if col not in df.columns:
            return Check(cid, "", "q_gap", "warn", True, f"{csv_name}: skipped (missing {col})")
    gap = (_num(df["train/loss/q1"]) - _num(df["train/loss/q2"])).abs().dropna()
    if gap.empty:
        return Check(cid, "", "q_gap", "warn", True, f"{csv_name}: no Q values to compare")
    early = gap.iloc[: max(1, len(gap) // 4)].mean()
    late = gap.iloc[-max(1, len(gap) // 4):].mean()
    ok = float(late) <= max(2.0 * float(early), 1.0)
    msg = (
        f"{csv_name}: Q1/Q2 gap stable ({early:.3g} -> {late:.3g})"
        if ok
        else f"{csv_name}: Q1/Q2 disagreement grew {early:.3g} -> {late:.3g} (overestimation?)"
    )
    return Check(cid, "", "q_gap", "warn", ok, msg)


def detect_algo(path: str, df: Optional[pd.DataFrame] = None) -> str:
    base = os.path.basename(path).lower()
    if "sac" in base:
        return "sac"
    if "ppo" in base:
        return "ppo"
    cols = set(df.columns) if df is not None else set()
    if "train/approx_kl" in cols:
        return "ppo"
    if "train/actor_loss" in cols:
        return "sac"
    raise ValueError(f"{path}: cannot determine algorithm (name lacks ppo/sac and columns ambiguous)")


def validate_file(path: str, solve: bool = False) -> tuple:
    df = load_csv(path)
    algo = detect_algo(path, df)
    if algo == "ppo":
        return "ppo", path, validate_ppo(df, csv_name=os.path.basename(path), solve=solve)
    return "sac", path, validate_sac(df, csv_name=os.path.basename(path), solve=solve)


def validate_folder(dirpath: str, solve: bool = False) -> dict:
    results = {}
    for fname in ("ppo_progress.csv", "sac_progress.csv"):
        p = os.path.join(dirpath, fname)
        if os.path.isfile(p):
            algo, path, checks = validate_file(p, solve=solve)
            results[algo] = (path, checks)
    if not results:
        raise FileNotFoundError(f"{dirpath}: no ppo_progress.csv / sac_progress.csv found")
    return results


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------

_SYMBOL = {"hard": "FAIL", "warn": "WARN", "solve": "SOLVE"}


def format_report(algo: str, path: str, checks: List[Check]) -> str:
    lines = [f"{algo.upper()}  {path}"]
    for c in checks:
        if c.severity == "warn" and c.ok:
            continue  # only surface warnings that actually fired
        mark = "PASS" if c.ok else _SYMBOL[c.severity]
        lines.append(f"  {mark:>4}  [{c.severity}]  {c.message}")
    return "\n".join(lines)


def summarize(results: dict) -> int:
    """Return process exit code (0 = all hard checks pass)."""
    exit_code = 0
    for algo, (path, checks) in results.items():
        hard = [c for c in checks if c.severity == "hard"]
        warn = [c for c in checks if c.severity == "warn"]
        n_hard_fail = sum(1 for c in hard if not c.ok)
        n_warn_fail = sum(1 for c in warn if not c.ok)
        if n_hard_fail:
            exit_code = 1
        print(f"{algo.upper()}: {sum(1 for c in hard if c.ok)}/{len(hard)} hard pass, "
              f"{n_warn_fail} warning(s)")
    return exit_code


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate PPO/SAC progress CSVs")
    parser.add_argument("path", help="folder containing ppo_progress.csv / sac_progress.csv, or a single CSV")
    parser.add_argument("--solve", action="store_true", help="also run task-solve thresholds (CartPole/Pendulum defaults)")
    args = parser.parse_args(argv)

    if os.path.isdir(args.path):
        results = validate_folder(args.path, solve=args.solve)
    else:
        algo, path, checks = validate_file(args.path, solve=args.solve)
        results = {algo: (path, checks)}

    for algo, (path, checks) in results.items():
        print(format_report(algo, path, checks))
    print()
    return summarize(results)


if __name__ == "__main__":
    raise SystemExit(main())
