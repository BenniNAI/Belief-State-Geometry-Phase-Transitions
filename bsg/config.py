"""Config loading: YAML files merged over defaults, plus dotted command-line overrides
such as `--set train.lr=3e-4 process.noise=0.1 process.components.0.factors.1.x=0.2`."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

DEFAULTS = {
    "name": None,
    "seed": 0,
    "device": None,  # default: cuda if available, else cpu
    "out_dir": "runs",
    "process": None,
    "data": {
        "ctx_len": 10,
        "bos": False,  # prepend a BOS token (input position t has then seen t tokens)
        "batch_size": 64,  # sequences per step *per member*
        "dataset_size": None,  # None = fresh sequences every step; int = fixed finite dataset
        "buffer_steps": 32,  # batches sampled at once (speed only)
    },
    "model": {
        "n_layers": 4, "d_model": 64, "n_heads": 1, "d_head": None, "d_mlp": 256,
        "act": "relu", "norm": "layernorm", "pos": "learned", "init_std": 0.02,
        "attn_only": False,
    },
    "train": {
        "n_seeds": 4,  # ensemble members trained in parallel (seeds seed..seed+n-1)
        "steps": 10_000,
        "optimizer": "adam",  # sgd | adam | adamw
        "lr": 1e-3,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "momentum": 0.0,
        "warmup_steps": 0,
        "grad_accum": 1,  # split each batch into this many micro-batches (same gradient, less memory)
        "log_every": 100,
        "compile": False,  # torch.compile the model (Linux + CUDA recommended)
    },
    "eval": {
        "n_sequences": 2048,
        "dense_until": 50,  # evaluate every step up to here
        "per_decade": 20,  # then this many log-spaced checkpoints per decade
        "every": None,  # optionally also every N steps
        "probe_rows": 50_000,  # (sequence, position) pairs used for probes
        "test_frac": 0.2,
        "chunk_seqs": 256,
        "snapshot_points": 2000,
        "snapshot_layers": "all",  # or a list, e.g. ["final", "concat"]
        "save_weights": True,
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "process":
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value):
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def get_dotted(cfg: dict, key: str):
    node = cfg
    for p in key.split("."):
        node = node[int(p)] if isinstance(node, list) else node[p]
    return node


def parse_overrides(items) -> list[tuple[str, object]]:
    out = []
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"override must look like key=value, got {it!r}")
        k, v = it.split("=", 1)
        val = yaml.safe_load(v)
        if isinstance(val, str):  # YAML 1.1 reads "3e-4" as a string
            try:
                val = float(val)
            except ValueError:
                pass
        out.append((k.strip(), val))
    return out


def load_config(path: str | Path | None = None, overrides=None, extra: dict | None = None) -> dict:
    user = {}
    if path is not None:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    if extra:
        cfg = deep_merge(cfg, extra)
    for item in overrides or []:
        k, v = parse_overrides([item])[0] if isinstance(item, str) else item
        set_dotted(cfg, k, v)
    if cfg["process"] is None:
        raise ValueError("config must define a `process`")
    if not cfg["name"]:
        cfg["name"] = Path(path).stem if path else "run"
    return cfg


def checkpoint_steps(total: int, dense_until: int = 50, per_decade: int = 20, every=None) -> list[int]:
    s = set(range(0, min(dense_until, total) + 1))
    if per_decade and total > 1:
        k = int(np.ceil(per_decade * np.log10(total))) + 1
        s.update(int(round(x)) for x in np.logspace(0, np.log10(total), k))
    if every:
        s.update(range(0, total + 1, int(every)))
    s.add(total)
    return sorted(x for x in s if 0 <= x <= total)
