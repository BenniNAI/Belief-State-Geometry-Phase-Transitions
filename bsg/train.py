"""Train an ensemble of small transformers on a generating process and record order
parameters at log-spaced checkpoints.

    python -m bsg.train configs/mess3_replication.yaml
    python -m bsg.train configs/factored_noise.yaml --set process.noise=0.05 train.steps=2000

Outputs (in runs/<name>/):
    config.yaml        resolved config
    process.json       process name, sizes, structure;  process_T.npy  transition matrices
    reference.json     KL(bayes || ref) of each reference predictor (predicted plateau heights)
    metrics.csv        long format: step, member, metric, layer, target, value
    train_log.csv      training loss per member every `log_every` steps
    per_position.npz   per-position loss and KL curves at every checkpoint
    snapshots/         probe predictions on held-out points, for geometry plots
    checkpoints/       ensemble weights (if eval.save_weights)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .analysis import evaluate, reference_summary
from .beliefs import DataStream, TorchProcess, make_eval_set
from .config import checkpoint_steps, load_config
from .model import EnsembleTransformer, ModelConfig, count_params, make_optimizer, per_member_loss
from .processes import build_process

METRIC_FIELDS = ["step", "member", "metric", "layer", "target", "value"]


def _device(cfg):
    if cfg.get("device"):
        return torch.device(cfg["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(cfg: dict, out_dir: str | Path | None = None, quiet: bool = False) -> Path:
    out = Path(out_dir) if out_dir else Path(cfg["out_dir"]) / cfg["name"]
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    dev = _device(cfg)
    torch.manual_seed(cfg["seed"])
    proc = build_process(cfg["process"])
    tp = TorchProcess(proc, dev)
    dc, mc, tc, ec = cfg["data"], cfg["model"], cfg["train"], cfg["eval"]
    bos, ctx, E, B = bool(dc["bos"]), int(dc["ctx_len"]), int(tc["n_seeds"]), int(dc["batch_size"])
    accum = int(tc.get("grad_accum") or 1)
    if B % accum:
        raise ValueError(f"data.batch_size ({B}) must be divisible by train.grad_accum ({accum})")

    with open(out / "process.json", "w") as f:
        json.dump({"name": proc.name, "vocab_size": tp.V, "n_states": tp.d,
                   "structure": proc.structure}, f, indent=1)
    np.save(out / "process_T.npy", proc.T)

    model = EnsembleTransformer(ModelConfig(**mc), tp.V + int(bos), tp.V, ctx, E, seed=cfg["seed"]).to(dev)
    fwd = torch.compile(model) if tc.get("compile") else model
    opt = make_optimizer(model.parameters(), tc["optimizer"], float(tc["lr"]), float(tc["weight_decay"]),
                         tc.get("betas", (0.9, 0.999)), float(tc.get("momentum", 0.0)))
    warm = int(tc.get("warmup_steps") or 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) if warm else 1.0)

    stream = DataStream(tp, E * B, ctx, bos, seed=cfg["seed"] + 10_000,
                        dataset_size=dc.get("dataset_size"), buffer_steps=int(dc.get("buffer_steps", 32)))
    ev = make_eval_set(tp, int(ec["n_sequences"]), ctx, bos, seed=cfg["seed"] + 20_000)
    with open(out / "reference.json", "w") as f:
        json.dump(reference_summary(ev), f, indent=1)

    total = int(tc["steps"])
    ckpts = set(checkpoint_steps(total, int(ec["dense_until"]), int(ec["per_decade"]), ec.get("every")))
    if ec.get("save_weights"):
        (out / "checkpoints").mkdir(exist_ok=True)
    meta = {"params_per_member": count_params(model), "tokens_per_step": E * B * ctx,
            "device": str(dev), "n_checkpoints": len(ckpts)}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    if not quiet:
        print(f"[{cfg['name']}] {proc.name}: V={tp.V}, d={tp.d}; {meta['params_per_member']:,} params x "
              f"{E} members on {dev}; {total:,} steps, {len(ckpts)} checkpoints")

    mf = open(out / "metrics.csv", "w", newline="")
    mw = csv.DictWriter(mf, fieldnames=METRIC_FIELDS)
    mw.writeheader()
    lf = open(out / "train_log.csv", "w", newline="")
    lw = csv.writer(lf)
    lw.writerow(["step", "member", "loss"])
    per_pos: dict[str, list] = {}
    ckpt_list: list[int] = []

    run_loss = torch.zeros(E, device=dev)
    n_acc = 0
    t0 = time.time()
    log_every = int(tc.get("log_every") or 100)
    model.train()
    for step in range(total + 1):
        if step in ckpts:
            model.eval()
            rows, pos, snap = evaluate(
                model, ev, proc.structure, step, probe_rows=int(ec["probe_rows"]),
                test_frac=float(ec["test_frac"]), chunk_seqs=int(ec["chunk_seqs"]),
                snapshot_points=int(ec["snapshot_points"]), seed=cfg["seed"],
                snapshot_layers=ec.get("snapshot_layers", "all"))
            model.train()
            mw.writerows(rows)
            mf.flush()
            ckpt_list.append(step)
            for k, v in pos.items():
                per_pos.setdefault(k, []).append(v)
            np.savez_compressed(out / "per_position.npz", steps=np.array(ckpt_list),
                                **{k: np.stack(v) for k, v in per_pos.items()})
            np.savez_compressed(out / "snapshots" / f"step_{step:09d}.npz", step=step, **snap)
            if ec.get("save_weights"):
                torch.save({"step": step, "model": model.state_dict()},
                           out / "checkpoints" / f"step_{step:09d}.pt")
            if not quiet:
                exc = np.mean([r["value"] for r in rows if r["metric"] == "kl" and r["target"] == "bayes"])
                mem = (f", peak GPU memory {torch.cuda.max_memory_allocated(dev) / 1e9:.1f} GB"
                       if dev.type == "cuda" else "")
                print(f"  step {step:>9,}  excess loss {exc:.3e}  ({time.time() - t0:,.0f}s{mem})", flush=True)
        if step == total:
            break
        inputs, targets = stream.next()
        x, y = inputs.view(E, B, ctx), targets.view(E, B, ctx)
        opt.zero_grad(set_to_none=True)
        if accum == 1:
            loss_e = per_member_loss(fwd(x), y)
            loss_e.sum().backward()
        else:  # equal-sized micro-batches: same gradient as the full batch, less memory
            loss_e = torch.zeros(E, device=dev)
            for i in range(accum):
                mb = slice(i * B // accum, (i + 1) * B // accum)
                part = per_member_loss(fwd(x[:, mb]), y[:, mb])
                (part.sum() / accum).backward()
                loss_e += part.detach() / accum
        opt.step()
        sched.step()
        run_loss += loss_e.detach()
        n_acc += 1
        if (step + 1) % log_every == 0:
            vals = (run_loss / n_acc).tolist()
            if any(math.isnan(v) for v in vals):
                raise FloatingPointError(f"NaN loss at step {step + 1}")
            lw.writerows([step + 1, e, v] for e, v in enumerate(vals))
            lf.flush()
            run_loss.zero_()
            n_acc = 0
    mf.close()
    lf.close()
    done = {"seconds": time.time() - t0}
    if dev.type == "cuda":
        done["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
    with open(out / "done.json", "w") as f:
        json.dump(done, f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None, help="output directory (default runs/<name>)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.name:
        cfg["name"] = a.name
    path = run(cfg, a.out, a.quiet)
    print(f"done: {path}")


if __name__ == "__main__":
    main()
