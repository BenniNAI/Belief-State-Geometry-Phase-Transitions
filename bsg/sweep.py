"""Run a grid of configs, one after another.

    python -m bsg.sweep configs/sweeps/noise_plateau.yaml [--dry-run]

Sweep file:
    base: ../factored_noise.yaml      # relative to the sweep file
    name: noise_plateau               # runs go to <out_dir>/<name>/<key=value,...>
    set:                              # fixed overrides (optional)
      train.steps: 5000
    grid:                             # every combination is run
      process.noise: [0.0, 0.01, 0.05, 0.1]
      model.init_std: [0.02, 0.05]
      # keys joined by commas are tied: all of them get the same value
      "process.components.0.couplings.0.eps,process.components.1.couplings.0.eps": [0.3, 1.0]

Finished runs (with done.json) are skipped, so an interrupted sweep can be restarted.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

from .config import load_config
from .train import run


_ABBREV = {"process": "", "components": "c", "factors": "f", "couplings": "cp", "model": "",
           "train": "", "data": ""}


def _short(key: str) -> str:
    """Compact but unambiguous tag for a dotted key, e.g.
    process.components.1.factors.0.x -> c_1_f_0_x,  model.init_std -> init_std."""
    return "_".join(p for p in (_ABBREV.get(x, x) for x in key.split(".")) if p)


def expand(sweep_path: str | Path):
    sweep_path = Path(sweep_path)
    sw = yaml.safe_load(open(sweep_path))
    base = (sweep_path.parent / sw["base"]).resolve()
    name = sw.get("name", sweep_path.stem)
    fixed = list((sw.get("set") or {}).items())
    grid = sw.get("grid") or {}
    keys = list(grid)
    jobs = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        overrides = list(fixed)
        for k, v in zip(keys, combo):
            overrides += [(kk.strip(), v) for kk in k.split(",")]
        tag = ",".join(f"{_short(k.split(',')[0].strip())}={v}" for k, v in zip(keys, combo)) or "base"
        cfg = load_config(base, overrides)
        cfg["name"] = f"{name}/{tag}"
        cfg.setdefault("sweep", {})
        cfg["sweep"] = {"name": name, "params": dict(zip(keys, combo))}
        jobs.append(cfg)
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    jobs = expand(a.sweep)
    print(f"{len(jobs)} runs")
    for i, cfg in enumerate(jobs):
        out = Path(cfg["out_dir"]) / cfg["name"]
        status = "done" if (out / "done.json").exists() else "todo"
        print(f"[{i + 1}/{len(jobs)}] {cfg['name']} ({status})")
        if a.dry_run or status == "done":
            continue
        run(cfg, out)


if __name__ == "__main__":
    main()
