"""Validate PPO/SAC progress CSVs in a folder (or a single CSV).

Usage:
    python bench/reinforcement/validate_csvs.py logs/sac
    python bench/reinforcement/validate_csvs.py logs/ppo/ppo_progress.csv
    python bench/reinforcement/validate_csvs.py logs/sac --solve
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dirty_mlx_ml.reinforcement.validation import main

if __name__ == "__main__":
    raise SystemExit(main())
