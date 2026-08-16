import glob
import os

import pytest

from dirty_mlx_ml.reinforcement.validation import load_csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def pytest_addoption(parser):
    parser.addoption(
        "--csv-dir",
        action="store",
        default=None,
        help="directory to discover ppo_progress.csv / sac_progress.csv from",
    )


def _discover(name: str, csv_dir=None):
    if csv_dir:
        p = os.path.join(csv_dir, name)
        return [p] if os.path.isfile(p) else []
    return sorted(glob.glob(os.path.join(ROOT, "logs", "**", name), recursive=True))


def pytest_generate_tests(metafunc):
    csv_dir = metafunc.config.getoption("--csv-dir")
    if "ppo_csv_path" in metafunc.fixturenames:
        metafunc.parametrize("ppo_csv_path", _discover("ppo_progress.csv", csv_dir) or [None])
    if "sac_csv_path" in metafunc.fixturenames:
        metafunc.parametrize("sac_csv_path", _discover("sac_progress.csv", csv_dir) or [None])


@pytest.fixture
def ppo_df(ppo_csv_path):
    if ppo_csv_path is None:
        pytest.skip("no ppo_progress.csv found (run training or pass --csv-dir)")
    return load_csv(ppo_csv_path), ppo_csv_path


@pytest.fixture
def sac_df(sac_csv_path):
    if sac_csv_path is None:
        pytest.skip("no sac_progress.csv found (run training or pass --csv-dir)")
    return load_csv(sac_csv_path), sac_csv_path
