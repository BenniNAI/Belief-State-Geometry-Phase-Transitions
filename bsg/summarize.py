"""Plateau / transition times across runs (e.g. a sweep), from reference-predictor KLs.

    python -m bsg.summarize runs/noise_plateau --ref factored --x process.noise

For each run and member, with reference predictor `ref` (e.g. factored, frozen_source):
    gap     = KL(bayes || ref)  -- excess loss of the reference itself (the plateau height)
    entry   = first checkpoint with KL(ref || model) < frac * gap     (model behaves like ref)
    exit    = first later checkpoint with excess loss < frac * gap    (model has moved past ref)
    plateau = exit - entry   in steps, and in lr * steps
Writes <dir>/transitions_<ref>.csv and, with --x, transitions_<ref>.png (plateau vs parameter).

These thresholds are a crude first pass; check them against the curves (bsg.plot curves).
A missing entry means the model never got close to the reference; a missing exit means it
had not left the plateau by the end of training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from .config import get_dotted  # noqa: E402
from .plot import load_metrics, series  # noqa: E402


def transitions(run: Path, ref: str, frac: float = 0.5):
    run = Path(run)
    refs = json.load(open(run / "reference.json"))
    if ref not in refs:
        return []
    gap = refs[ref]["mean"]
    cfg = yaml.safe_load(open(run / "config.yaml"))
    lr = float(cfg["train"]["lr"])
    rows = load_metrics(run)
    to_ref = series(rows, "kl", "", ref)
    excess = series(rows, "kl", "", "bayes")
    out = []
    for m in sorted(to_ref):
        st, kr = to_ref[m]
        _, ex = excess[m]
        entry = exit_ = np.nan
        if gap > 0:
            idx = np.nonzero(kr < frac * gap)[0]
            if len(idx):
                entry = st[idx[0]]
                later = np.nonzero((st > entry) & (ex < frac * gap))[0]
                if len(later):
                    exit_ = st[later[0]]
        out.append({"run": str(run), "member": m, "ref": ref, "gap": gap, "entry": entry, "exit": exit_,
                    "plateau": exit_ - entry, "plateau_lr_steps": lr * (exit_ - entry), "lr": lr,
                    "final_excess": float(ex[-1])})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="a run directory or a directory containing runs")
    ap.add_argument("--ref", default="factored")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--x", default=None, help="config key to plot on the x axis, e.g. process.noise")
    a = ap.parse_args(argv)
    root = Path(a.dir)
    runs = sorted(p.parent for p in root.rglob("metrics.csv"))
    results = []
    for r in runs:
        res = transitions(r, a.ref, a.frac)
        if a.x:
            cfg = yaml.safe_load(open(r / "config.yaml"))
            xv = get_dotted(cfg, a.x)
            for row in res:
                row[a.x] = xv
        results.extend(res)
    if not results:
        print(f"no runs with reference '{a.ref}' under {root}")
        return
    path = root / f"transitions_{a.ref}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(path)
    if a.x:
        xs = np.array([r[a.x] for r in results], dtype=float)
        ys = np.array([r["plateau"] for r in results], dtype=float)
        ok = np.isfinite(ys) & (xs > 0) & (ys > 0)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(xs[ok], ys[ok], s=14, alpha=0.6, color="#2f6db5", label="members")
        ux = np.unique(xs[ok])
        if len(ux):
            ax.plot(ux, [np.median(ys[ok][xs[ok] == u]) for u in ux], color="#222", lw=1.8, label="median")
        if len(ux) >= 2:
            p = np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)
            ax.plot(ux, np.exp(np.polyval(p, np.log(ux))), "--", color="#d1495b",
                    label=f"fit: plateau ~ x^{p[0]:.2f}")
        ax.set(xscale="log", yscale="log", xlabel=a.x, ylabel=f"plateau length at '{a.ref}' [steps]")
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = root / f"transitions_{a.ref}.png"
        fig.savefig(png, dpi=130)
        print(png)


if __name__ == "__main__":
    main()
