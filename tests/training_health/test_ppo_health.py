import os
import warnings

import pytest

from dirty_mlx_ml.reinforcement.validation import PPO_COLUMNS, PPO_SPECS, columns_present, run_spec


@pytest.mark.parametrize("spec", PPO_SPECS, ids=[s.id for s in PPO_SPECS])
def test_ppo_field(ppo_df, spec):
    df, path = ppo_df
    check = run_spec(df, spec, os.path.basename(path))
    if check.severity == "hard":
        assert check.ok, check.message
    elif not check.ok:
        warnings.warn(check.message, stacklevel=2)


def test_ppo_required_columns(ppo_df):
    df, path = ppo_df
    check = columns_present(df, PPO_COLUMNS, os.path.basename(path))
    assert check.ok, check.message
