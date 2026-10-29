import csv
import json

import pytest
import torch

from bsg.analysis import _fit_affine, _r2, _subspaces
from bsg.config import load_config
import numpy as np

from bsg.plot import plot_curves, plot_geometry, plot_layers
from bsg.summarize import transitions
from bsg.sweep import expand
from bsg.train import run

TINY = ["train.steps=12", "train.n_seeds=2", "data.batch_size=16", "eval.n_sequences=64",
        "eval.probe_rows=2000", "eval.dense_until=2", "eval.per_decade=2", "eval.snapshot_points=200",
        "device=cpu", "data.ctx_len=10", "model.d_model=32", "model.d_mlp=64", "model.n_heads=2",
        "model.d_head=16", "model.n_layers=2", "train.log_every=4"]


def test_probe_recovers_linear_map():
    g = torch.Generator().manual_seed(0)
    Y = torch.rand(2000, 3, generator=g, dtype=torch.float64)
    A = torch.randn(3, 20, generator=g, dtype=torch.float64)
    X = Y @ A + 0.5 + 1e-3 * torch.randn(2000, 20, generator=g, dtype=torch.float64)
    W, b = _fit_affine(X[:1500], Y[:1500], 1e-8)
    assert _r2(Y[1500:], X[1500:] @ W + b) > 0.999


def test_subspace_overlap():
    g = torch.Generator().manual_seed(1)
    Ya = torch.rand(3000, 3, generator=g, dtype=torch.float64)
    Ya = Ya / Ya.sum(1, keepdim=True)  # simplex-valued: rank 2 after centering
    Yb = torch.rand(3000, 3, generator=g, dtype=torch.float64)
    Yb = Yb / Yb.sum(1, keepdim=True)
    Q, _ = torch.linalg.qr(torch.randn(16, 6, generator=g, dtype=torch.float64))
    Yb = 0.7 * Yb + 0.3 * Ya  # correlated targets, as for interacting factors
    X = Ya @ Q[:, :3].T + Yb @ Q[:, 3:].T
    (Qa, ra), (Qb, rb) = _subspaces(X, [Ya, Yb], 1e-8)
    assert ra == 2 and rb == 2
    assert float((Qa.T @ Qb).pow(2).sum() / 2) < 1e-6
    X2 = Ya @ Q[:, :3].T + Yb @ Q[:, :3].T  # same subspace
    (Qa2, _), (Qb2, _) = _subspaces(X2, [Ya, Yb], 1e-8)
    assert float((Qa2.T @ Qb2).pow(2).sum() / 2) > 0.95


@pytest.mark.parametrize("name", ["mess3_replication", "rrxor_replication", "factored_noise",
                                  "factored_two_channels", "nonergodic_two_mess3", "combined"])
def test_tiny_end_to_end(name, tmp_path):
    cfg = load_config(f"configs/{name}.yaml", TINY)
    out = run(cfg, tmp_path / name, quiet=True)
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    metrics = {r["metric"] for r in rows}
    assert {"loss", "kl", "r2", "rmse", "n95", "pr", "bayes_loss"} <= metrics
    excess = [float(r["value"]) for r in rows if r["metric"] == "kl" and r["target"] == "bayes"]
    assert all(v >= -1e-9 for v in excess)
    ref = json.load(open(out / "reference.json"))
    assert abs(ref["bayes"]["mean"]) < 1e-12
    assert (out / "done.json").exists() and list((out / "checkpoints").glob("*.pt"))
    assert plot_curves(out).exists()
    snap = np.load(sorted((out / "snapshots").glob("*.npz"))[-1])
    assert list(snap["layers"]) == ["0", "1", "2", "concat"]  # every layer + concat
    assert all(f"pred/{l}/belief" in snap for l in snap["layers"])
    assert plot_geometry(out, target="belief", n=3).name == "geometry_belief_concat.png"
    assert plot_geometry(out, target="belief", layer="final", n=3).name == "geometry_belief_2.png"
    assert plot_geometry(out, target="belief", layer="1", n=3).exists()
    assert plot_layers(out, target="belief").exists()
    if "factored" in ref:
        assert any(r["metric"] == "overlap" for r in rows)
        assert len(transitions(out, "factored")) == 2


def test_training_reduces_loss(tmp_path):
    cfg = load_config("configs/mess3_replication.yaml",
                      TINY + ["train.steps=300", "train.optimizer=adam", "train.lr=3e-3", "data.batch_size=64"])
    out = run(cfg, tmp_path / "learn", quiet=True)
    rows = [r for r in csv.DictReader(open(out / "metrics.csv")) if r["metric"] == "kl" and r["target"] == "bayes"]
    first = min(int(r["step"]) for r in rows)
    last = max(int(r["step"]) for r in rows)
    v0 = sum(float(r["value"]) for r in rows if int(r["step"]) == first)
    v1 = sum(float(r["value"]) for r in rows if int(r["step"]) == last)
    assert v1 < 0.3 * v0


@pytest.mark.parametrize("opt", ["sgd", "adam"])
def test_grad_accum_matches_full_batch(tmp_path, opt):
    base = TINY + ["train.steps=6", "eval.dense_until=0", "eval.per_decade=0", f"train.optimizer={opt}",
                   "train.lr=0.01"]
    w = []
    for k in (1, 4):
        out = run(load_config("configs/nonergodic_two_mess3.yaml", base + [f"train.grad_accum={k}"]),
                  tmp_path / f"acc{k}", quiet=True)
        w.append(torch.load(out / "checkpoints" / "step_000000006.pt")["model"])
    for name in w[0]:
        # the key bias has exactly zero true gradient (softmax ignores a constant shift), so Adam
        # turns floating-point noise in it into full-size updates; it does not affect outputs
        if opt == "adam" and name.endswith("b_K"):
            continue
        assert torch.allclose(w[0][name], w[1][name], atol=1e-5), name
    with pytest.raises(ValueError):
        run(load_config("configs/mess3_replication.yaml", TINY + ["train.grad_accum=3"]), tmp_path / "bad", quiet=True)


def test_finite_dataset_run(tmp_path):
    cfg = load_config("configs/mess3_replication.yaml", TINY + ["data.dataset_size=100"])
    assert (run(cfg, tmp_path / "finite", quiet=True) / "done.json").exists()


def test_sweep_expansion():
    jobs = expand("configs/sweeps/combined_phase_diagram.yaml")
    assert len(jobs) == 16
    j = jobs[5]
    c = j["process"]["components"]
    assert c[0]["couplings"][0]["eps"] == c[1]["couplings"][0]["eps"]
    assert len({job["name"] for job in jobs}) == 16
    assert len(expand("configs/sweeps/noise_plateau.yaml")) == 14
