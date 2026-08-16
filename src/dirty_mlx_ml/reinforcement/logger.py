import csv
import os
import time
from collections import defaultdict
from datetime import datetime


class CSVLogger:
    """SB3-style key/value logger → {name}_progress.csv (time/*, rollout/*, train/*)."""

    def __init__(self, log_dir: str | None = None, name: str = "progress"):
        if log_dir is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            log_dir = os.path.join("logs", ts)
        os.makedirs(log_dir, exist_ok=True)
        self.dir = log_dir
        # ppo_progress.csv / sac_progress.csv / progress.csv
        fname = f"{name}_progress.csv" if name != "progress" else "progress.csv"
        if name in ("ppo", "sac"):
            fname = f"{name}_progress.csv"
        self.path = os.path.join(log_dir, fname)
        self._vals: dict[str, float] = {}
        self._means: dict[str, list[float]] = defaultdict(list)
        self._keys: list[str] = []
        self._rows: list[dict] = []
        self.start_time = time.time()

    def record(self, key: str, value):
        if value is None:
            return
        self._vals[key] = float(value)

    def record_mean(self, key: str, value):
        if value is None:
            return
        self._means[key].append(float(value))

    def dump(self, step: int = 0):
        row = dict(self._vals)
        for k, vs in self._means.items():
            if vs:
                row[k] = sum(vs) / len(vs)
        if not row:
            return
        for k in row:
            if k not in self._keys:
                self._keys.append(k)
        self._rows.append({k: row.get(k, "") for k in self._keys})
        
        # Sort keys: time/*, rollout/*, train/*
        def key_sort(k):
            if k.startswith("time/"):
                return (0, k)
            elif k.startswith("rollout/"):
                return (1, k)
            elif k.startswith("train/"):
                return (2, k)
            else:
                return (3, k)
        
        sorted_keys = sorted(self._keys, key=key_sort)
        
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted_keys, extrasaction="ignore")
            w.writeheader()
            for r in self._rows:
                w.writerow({k: r.get(k, "") for k in sorted_keys})
        self._vals.clear()
        self._means.clear()

    def close(self):
        pass
