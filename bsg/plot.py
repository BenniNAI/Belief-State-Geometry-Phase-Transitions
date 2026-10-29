"""Plots for a finished (or running) run.

    python -m bsg.plot curves RUN_DIR [--target factors]
        -> RUN_DIR/curves.png: excess loss with predicted plateau heights, KL to each
           reference predictor, probe R^2 per layer, effective dimension per layer
    python -m bsg.plot geometry RUN_DIR [--target belief] [--layer concat|final|<k>] [--member 0] [--n 8]
        -> RUN_DIR/geometry_<target>_<layer>.png: probed belief geometry across training
    python -m bsg.plot layers RUN_DIR [--target belief] [--step N] [--member 0]
        -> RUN_DIR/layers_<target>_step<N>.png: one checkpoint probed from every layer
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REF_COLORS = {"bayes": "#222222", "unigram": "#9a9a9a", "last_token": "#c49a3a", "factored": "#2f6db5",
              "frozen_source": "#b5452f", "factored_frozen": "#7b4fa8"}


def load_metrics(run: Path):
    out = []
    with open(Path(run) / "metrics.csv") as f:
        for r in csv.DictReader(f):
            r["step"] = int(r["step"])
            r["member"] = int(r["member"])
            r["value"] = float(r["value"])
            out.append(r)
    return out


def series(rows, metric, layer="", target=""):
    """{member: (steps, values)} for one metric."""
    d = defaultdict(list)
    for r in rows:
        if r["metric"] == metric and r["layer"] == layer and r["target"] == target:
            d[r["member"]].append((r["step"], r["value"]))
    return {m: tuple(np.array(v) for v in zip(*sorted(pts))) for m, pts in d.items()}


def mean_series(s):
    members = list(s)
    steps = s[members[0]][0]
    vals = np.stack([s[m][1] for m in members])
    return steps, vals


def _xs(steps):
    return np.maximum(steps, 0.5)  # show step 0 on a log axis


def default_target(structure: dict) -> str:
    if structure.get("type") == "product":
        return "factors"
    if structure.get("type") == "nonergodic":
        return "belief"
    return "belief"


def plot_curves(run: Path, target: str | None = None):
    run = Path(run)
    rows = load_metrics(run)
    ref = json.load(open(run / "reference.json"))
    struct = json.load(open(run / "process.json"))["structure"]
    target = target or default_target(struct)
    cfg = yaml.safe_load(open(run / "config.yaml"))
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))

    # 1. excess loss + plateau heights
    a = ax[0, 0]
    s = series(rows, "kl", "", "bayes")
    for m, (st, v) in s.items():
        a.plot(_xs(st), v, color="#222222", alpha=0.25, lw=0.8)
    st, vals = mean_series(s)
    a.plot(_xs(st), vals.mean(0), color="#222222", lw=2, label="model (mean)")
    for k, info in ref.items():
        if k.startswith("_") or k == "bayes":
            continue
        a.axhline(info["mean"], ls="--", lw=1, color=REF_COLORS.get(k, "#888"), label=f"{k} predictor")
    a.set(xscale="log", yscale="log", xlabel="step", ylabel="excess loss  KL(bayes || model)  [nats/token]",
          title="Excess loss and predicted plateau heights")
    a.legend(fontsize=8)

    # 2. KL to each reference
    a = ax[0, 1]
    refs = sorted({r["target"] for r in rows if r["metric"] == "kl"})
    for k in refs:
        st, vals = mean_series(series(rows, "kl", "", k))
        a.plot(_xs(st), vals.mean(0), color=REF_COLORS.get(k, None), lw=1.8, label=k)
    a.set(xscale="log", yscale="log", xlabel="step", ylabel="KL(ref || model)",
          title="Distance to reference predictors (low = model behaves like it)")
    a.legend(fontsize=8)

    # 3. probe R^2 per layer
    a = ax[1, 0]
    layers = sorted({r["layer"] for r in rows if r["metric"] == "r2" and r["target"] == target},
                    key=lambda x: (x == "concat", int(x) if x.isdigit() else 0))
    cmap = plt.get_cmap("viridis")
    for i, layer in enumerate(layers):
        st, vals = mean_series(series(rows, "r2", layer, target))
        c = "#d1495b" if layer == "concat" else cmap(i / max(1, len(layers) - 1))
        a.plot(_xs(st), vals.mean(0), color=c, lw=1.6, label=f"layer {layer}")
    a.set(xscale="log", ylim=(-0.05, 1.02), xlabel="step", ylabel="held-out R^2",
          title=f"Linear probe for '{target}'")
    a.legend(fontsize=8)

    # 4. effective dimension
    a = ax[1, 1]
    for i, layer in enumerate([l for l in layers if l != "concat"]):
        st, vals = mean_series(series(rows, "n95", layer, ""))
        a.plot(_xs(st), vals.mean(0), color=cmap(i / max(1, len(layers) - 2)), lw=1.6, label=f"layer {layer}")
    a.set(xscale="log", xlabel="step", ylabel="# PCs for 95% variance",
          title="Effective dimension of the residual stream")
    a.legend(fontsize=8)

    # 5. R^2 for the main targets, all layers concatenated
    a = ax[0, 2]
    main = [t for t in ("belief", "factors", "weights", "source_beliefs", "source_factors")
            if any(r["metric"] == "r2" and r["target"] == t for r in rows)]
    tcol = plt.get_cmap("tab10")
    for i, t in enumerate(main):
        st, vals = mean_series(series(rows, "r2", "concat", t))
        a.plot(_xs(st), vals.mean(0), color=tcol(i), lw=1.8, label=t)
    a.set(xscale="log", ylim=(-0.05, 1.02), xlabel="step", ylabel="held-out R^2",
          title="Probe R^2 by target (all layers concatenated)")
    a.legend(fontsize=8)

    # 6. subspace overlap between factors / sources
    a = ax[1, 2]
    final = str(max(int(l) for l in layers if l.isdigit())) if layers else ""
    pairs = sorted({r["target"] for r in rows if r["metric"] == "overlap" and r["layer"] == final})
    for i, pr in enumerate(pairs):
        st, vals = mean_series(series(rows, "overlap", final, pr))
        a.plot(_xs(st), vals.mean(0), color=tcol(i), lw=1.6, label=pr)
    if pairs:
        a.set(xscale="log", ylim=(-0.02, 1.02), xlabel="step", ylabel="overlap (0 = orthogonal)",
              title=f"Subspace overlap, layer {final}")
        a.legend(fontsize=7)
    else:
        a.axis("off")
        a.text(0.5, 0.5, "no factor or source structure", ha="center", va="center", color="#888")

    fig.suptitle(f"{cfg['name']}", fontsize=12)
    fig.tight_layout()
    path = run / "curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _project(true, pred, ctx_pos):
    """2D coordinates and colours for plotting a belief target. Returns (true_2d, pred_2d,
    colours, is_rgb). Simplex targets (k = 3) are drawn in barycentric coordinates; larger
    targets are projected onto the top two principal components of the true beliefs."""
    k = true.shape[1]
    if k == 3:
        corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
        return true @ corners, pred @ corners, np.clip(true, 0, 1), True
    if k <= 2:
        t = np.stack([true[:, 0], ctx_pos], 1)
        p = np.stack([pred[:, 0], ctx_pos], 1)
        return t, p, ctx_pos, False
    mu = true.mean(0)
    _, _, Vh = np.linalg.svd(true - mu, full_matrices=False)
    P = Vh[:2].T
    # colour by true belief: top-3 principal components -> RGB, so each belief state keeps one colour
    C = (true - mu) @ Vh[:3].T
    lo, hi = C.min(0), C.max(0)
    rgb = (C - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)
    if rgb.shape[1] < 3:
        rgb = np.concatenate([rgb, np.full((len(rgb), 3 - rgb.shape[1]), 0.5)], 1)
    return (true - mu) @ P, (pred - mu) @ P, 0.15 + 0.7 * rgb, True


def _gt_size(true) -> float:
    """Larger markers for ground truth with few distinct belief states (e.g. RRXOR)."""
    return 14.0 if len(np.unique(np.round(true, 6), axis=0)) < 200 else 2.0


def _resolve_layer(saved, layer) -> str:
    saved = [str(s) for s in saved]
    numeric = [s for s in saved if s.isdigit()]
    name = max(numeric, key=int) if str(layer) == "final" and numeric else str(layer)
    if name not in saved:
        raise KeyError(f"layer '{layer}' has no snapshots; saved layers: {saved}")
    return name


def plot_geometry(run: Path, target: str = "belief", layer: str = "concat", member: int = 0, n: int = 8):
    run = Path(run)
    cfg = yaml.safe_load(open(run / "config.yaml"))
    ctx = int(cfg["data"]["ctx_len"])
    files = sorted((run / "snapshots").glob("step_*.npz"))
    if not files:
        raise FileNotFoundError("no snapshots yet")
    steps = np.array([int(f.stem.split("_")[1]) for f in files])
    # choose n checkpoints spread evenly in log(step)
    want = np.unique(np.round(np.geomspace(1, max(1, steps.max()), n - 1)).astype(int))
    pick = [0] + [int(np.argmin(np.abs(steps - w))) for w in want]
    pick = sorted(set(pick))
    snaps = [np.load(files[i], allow_pickle=False) for i in pick]
    lname = _resolve_layer(snaps[0]["layers"], layer)
    key_true, key_pred = f"true/{target}", f"pred/{lname}/{target}"
    if key_true not in snaps[0]:
        opts = sorted(k[5:] for k in snaps[0].files if k.startswith("true/"))
        raise KeyError(f"target '{target}' not in snapshots; available: {opts}")
    cols = len(snaps) + 1
    fig, ax = plt.subplots(1, cols, figsize=(2.6 * cols, 2.9))
    true = snaps[0][key_true].astype(float)
    pos = (snaps[0]["rows"] % ctx).astype(float)
    t2, _, col, rgb = _project(true, true, pos)
    kw = dict(s=2, linewidths=0)
    ax[0].scatter(t2[:, 0], t2[:, 1], c=col, cmap=None if rgb else "viridis", **{**kw, "s": _gt_size(true)})
    ax[0].set_title("ground truth", fontsize=9)
    for j, sn in enumerate(snaps, start=1):
        pred = sn[key_pred][member].astype(float)
        _, p2, col, rgb = _project(true, pred, pos)
        ax[j].scatter(p2[:, 0], p2[:, 1], c=col, cmap=None if rgb else "viridis", **kw)
        ax[j].set_title(f"step {int(sn['step']):,}", fontsize=9)
    lim_x = (t2[:, 0].min(), t2[:, 0].max())
    lim_y = (t2[:, 1].min(), t2[:, 1].max())
    pad = lambda lo, hi: (lo - 0.1 * (hi - lo + 1e-9), hi + 0.1 * (hi - lo + 1e-9))
    for a in ax:
        a.set_xlim(*pad(*lim_x))
        a.set_ylim(*pad(*lim_y))
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect("equal" if rgb else "auto")
    where = "all layers concatenated" if lname == "concat" else f"layer {lname}"
    fig.suptitle(f"{cfg['name']}: '{target}' probed from {where}, member {member}", fontsize=10)
    fig.tight_layout()
    path = run / f"geometry_{target}_{lname}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_layers(run: Path, target: str = "belief", step: int | None = None, member: int = 0):
    """One checkpoint (default: the last), probed from each saved layer side by side, with
    the held-out R^2 of each probe. Shows where in the network the geometry lives."""
    run = Path(run)
    cfg = yaml.safe_load(open(run / "config.yaml"))
    ctx = int(cfg["data"]["ctx_len"])
    files = sorted((run / "snapshots").glob("step_*.npz"))
    if not files:
        raise FileNotFoundError("no snapshots yet")
    steps = np.array([int(f.stem.split("_")[1]) for f in files])
    i = len(files) - 1 if step is None else int(np.argmin(np.abs(steps - step)))
    sn = np.load(files[i], allow_pickle=False)
    st = int(sn["step"])
    if f"true/{target}" not in sn:
        opts = sorted(k[5:] for k in sn.files if k.startswith("true/"))
        raise KeyError(f"target '{target}' not in snapshots; available: {opts}")
    layers = [str(l) for l in sn["layers"]]
    r2 = {r["layer"]: r["value"] for r in load_metrics(run)
          if r["metric"] == "r2" and r["target"] == target and r["step"] == st and r["member"] == member}
    true = sn[f"true/{target}"].astype(float)
    pos = (sn["rows"] % ctx).astype(float)
    t2, _, col, rgb = _project(true, true, pos)
    cols = len(layers) + 1
    fig, ax = plt.subplots(1, cols, figsize=(2.6 * cols, 2.9))
    kw = dict(s=2, linewidths=0, cmap=None if rgb else "viridis")
    ax[0].scatter(t2[:, 0], t2[:, 1], c=col, **{**kw, "s": _gt_size(true)})
    ax[0].set_title("ground truth", fontsize=9)
    for j, l in enumerate(layers, start=1):
        _, p2, _, _ = _project(true, sn[f"pred/{l}/{target}"][member].astype(float), pos)
        ax[j].scatter(p2[:, 0], p2[:, 1], c=col, **kw)
        name = "concat" if l == "concat" else ("embeddings" if l == "0" else f"after block {l}")
        ax[j].set_title(f"{name}\nR² = {r2[l]:.3f}" if l in r2 else name, fontsize=9)
    pad = lambda lo, hi: (lo - 0.1 * (hi - lo + 1e-9), hi + 0.1 * (hi - lo + 1e-9))
    for a in ax:
        a.set_xlim(*pad(t2[:, 0].min(), t2[:, 0].max()))
        a.set_ylim(*pad(t2[:, 1].min(), t2[:, 1].max()))
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect("equal" if rgb else "auto")
    fig.suptitle(f"{cfg['name']}: '{target}' at step {st:,} by layer, member {member}", fontsize=10)
    fig.tight_layout()
    path = run / f"layers_{target}_step{st}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves")
    c.add_argument("run")
    c.add_argument("--target", default=None)
    g = sub.add_parser("geometry")
    g.add_argument("run")
    g.add_argument("--target", default="belief")
    g.add_argument("--layer", default="concat", help="concat, final, or a layer index (0 = embeddings)")
    g.add_argument("--member", type=int, default=0)
    g.add_argument("--n", type=int, default=8)
    y = sub.add_parser("layers")
    y.add_argument("run")
    y.add_argument("--target", default="belief")
    y.add_argument("--step", type=int, default=None, help="checkpoint (default: last)")
    y.add_argument("--member", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "curves":
        print(plot_curves(a.run, a.target))
    elif a.cmd == "geometry":
        print(plot_geometry(a.run, a.target, a.layer, a.member, a.n))
    else:
        print(plot_layers(a.run, a.target, a.step, a.member))


if __name__ == "__main__":
    main()
