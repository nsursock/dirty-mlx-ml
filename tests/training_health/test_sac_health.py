import os
import warnings

import pytest

from dirty_mlx_ml.reinforcement.validation import SAC_COLUMNS, SAC_SPECS, columns_present, run_spec


@pytest.mark.parametrize("spec", SAC_SPECS, ids=[s.id for s in SAC_SPECS])
def test_sac_field(sac_df, spec):
    df, path = sac_df
    check = run_spec(df, spec, os.path.basename(path))
    if check.severity == "hard":
        assert check.ok, check.message
    elif not check.ok:
        warnings.warn(check.message, stacklevel=2)


def test_sac_required_columns(sac_df):
    df, path = sac_df
    check = columns_present(df, SAC_COLUMNS, os.path.basename(path))
    assert check.ok, check.message
