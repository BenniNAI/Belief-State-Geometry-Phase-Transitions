"""Order parameters measured at each checkpoint.

For every ensemble member this computes, on a fixed evaluation set:
  loss            cross-entropy on the sampled targets
  kl[ref]         mean KL(p_ref || p_model) per token, for each reference predictor;
                  kl[bayes] is the excess loss over the Bayes-optimal predictor
  r2[layer,tgt]   held-out R^2 of an affine probe from the residual stream to a belief
                  target (layer "0" = embeddings, "k" = after block k, "concat" = all)
  n95, pr         effective dimension of the residual stream: number of principal
                  components for 95% variance, and the participation ratio
  overlap         subspace overlap between targets that should be orthogonal if the
                  representation is factored (0 = orthogonal, 1 = identical)
plus per-position KL curves and probe "snapshots" (held-out predictions, by default for every
layer and for "concat") for geometry plots.
"""
from __future__ import annotations

import torch

from .beliefs import EvalSet


def subspace_groups(structure: dict, names) -> list[list[str]]:
    groups = []
    if structure.get("type") == "product":
        groups.append([f"factor{f}" for f in range(len(structure["factor_dims"]))])
    elif structure.get("type") == "nonergodic":
        comps = structure["components"]
        groups.append([f"wsource{c}" for c in range(len(comps))])
        for c, comp in enumerate(comps):
            st = comp["structure"]
            if st.get("type") == "product":
                groups.append([f"source{c}_factor{f}" for f in range(len(st["factor_dims"]))])
    names = set(names)
    return [[g for g in grp if g in names] for grp in groups if len([g for g in grp if g in names]) > 1]


def _fit_affine(X: torch.Tensor, Y: torch.Tensor, ridge: float):
    """Ridge-stabilised least squares with intercept. X (N, D), Y (N, K), float64."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    A = Xc.T @ Xc
    lam = ridge * A.diagonal().mean().clamp_min(1e-12)
    W = torch.linalg.solve(A + lam * torch.eye(A.shape[0], dtype=A.dtype, device=A.device), Xc.T @ Yc)
    return W, my - mx @ W


def _r2(Y: torch.Tensor, Yhat: torch.Tensor) -> float:
    ss_res = (Y - Yhat).pow(2).sum()
    ss_tot = (Y - Y.mean(0)).pow(2).sum().clamp_min(1e-30)
    return float(1.0 - ss_res / ss_tot)


def _subspaces(X: torch.Tensor, Ys: list[torch.Tensor], ridge: float):
    """For each target Y_k, an orthonormal basis (D, r_k) of the directions along which it
    is embedded in activations X. All targets are regressed jointly (X ~ sum_k Y_k C_k), so
    correlations between targets do not leak into each other's subspace."""
    Xc = X - X.mean(0)
    Yc = [Y - Y.mean(0) for Y in Ys]
    Yall = torch.cat(Yc, 1)
    G = Yall.T @ Yall
    lam = ridge * G.diagonal().mean().clamp_min(1e-12)
    C = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), Yall.T @ Xc)
    out, off = [], 0
    for Y in Yc:
        k = Y.shape[1]
        ev = torch.linalg.eigvalsh(Y.T @ Y)
        r = int((ev > 1e-9 * ev.max().clamp_min(1e-30)).sum())
        _, _, Vh = torch.linalg.svd(C[off:off + k], full_matrices=False)
        out.append((Vh[:r].T, r))
        off += k
    return out


def _subspace(X: torch.Tensor, Y: torch.Tensor, ridge: float):
    return _subspaces(X, [Y], ridge)[0]


@torch.no_grad()
def evaluate(model, ev: EvalSet, structure: dict, step: int, probe_rows: int = 50_000,
             test_frac: float = 0.2, chunk_seqs: int = 256, snapshot_points: int = 2000,
             ridge: float = 1e-6, max_snapshot_dim: int = 32, seed: int = 0,
             snapshot_layers="all"):
    dev = ev.inputs.device
    E = model.E
    n, L = ev.inputs.shape
    n_layers = len(model.blocks)
    layer_names = [str(i) for i in range(n_layers + 1)]

    total = n * L
    R = min(probe_rows, total)
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(total, generator=g)[:R].to(dev)
    acts = [torch.empty(E, R, model.cfg.d_model, device=dev) for _ in layer_names]

    kl_sum = {k: torch.zeros(E, L, dtype=torch.float64, device=dev) for k in ev.refs}
    ce_sum = torch.zeros(E, L, dtype=torch.float64, device=dev)
    bayes_ce_sum = torch.zeros(L, dtype=torch.float64, device=dev)

    for a in range(0, n, chunk_seqs):
        b = min(n, a + chunk_seqs)
        inp = ev.inputs[a:b].unsqueeze(0).expand(E, -1, -1)
        logits, resid = model(inp, return_resid=True)
        logp = torch.log_softmax(logits.double(), -1)  # (E, c, L, V)
        tgt = ev.targets[a:b]
        ce_sum += -logp.gather(-1, tgt.unsqueeze(0).unsqueeze(-1).expand(E, -1, -1, 1)).squeeze(-1).sum(1)
        pb = ev.refs["bayes"][a:b]
        bayes_ce_sum += -torch.log(pb.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).clamp_min(1e-300)).sum(0)
        for k, p in ev.refs.items():
            p = p[a:b].unsqueeze(0)
            kl_sum[k] += (torch.xlogy(p, p) - p * logp).sum(-1).sum(1)
        mask = (rows >= a * L) & (rows < b * L)
        if mask.any():
            pos_in_R = mask.nonzero().squeeze(1)
            local = rows[mask] - a * L
            for li, h in enumerate(resid):
                acts[li][:, pos_in_R] = h.reshape(E, -1, h.shape[-1])[:, local].float()

    out_rows = []

    def add(member, metric, layer, target, value):
        out_rows.append({"step": step, "member": member, "metric": metric, "layer": layer,
                         "target": target, "value": float(value)})

    ce_pos = (ce_sum / n).cpu().numpy()
    kl_pos = {k: (v / n).cpu().numpy() for k, v in kl_sum.items()}
    for e in range(E):
        add(e, "loss", "", "", ce_pos[e].mean())
        for k, v in kl_pos.items():
            add(e, "kl", "", k, v[e].mean())
    add(-1, "bayes_loss", "", "", float((bayes_ce_sum / n).mean()))

    # ---------------------------------------------------------------- probes
    tnames = list(ev.probe_targets)
    Ys = {k: v.reshape(total, -1)[rows].double() for k, v in ev.probe_targets.items()}
    Yall = torch.cat([Ys[k] for k in tnames], 1)
    splits, off = {}, 0
    for k in tnames:
        splits[k] = slice(off, off + Ys[k].shape[1])
        off += Ys[k].shape[1]
    n_tr = max(1, int(R * (1 - test_frac)))
    tr, te = slice(0, n_tr), slice(n_tr, R)
    groups = subspace_groups(structure, tnames)

    snap_targets = [k for k in tnames if Ys[k].shape[1] <= max_snapshot_dim]
    S = min(snapshot_points, R - n_tr)
    snap_sl = slice(n_tr, n_tr + S)
    if snapshot_layers == "all":
        snap_layers = layer_names + ["concat"]
    else:
        snap_layers = [layer_names[-1] if str(l) == "final" else str(l) for l in snapshot_layers]
    snapshot = {"rows": rows[snap_sl].cpu().numpy(), "layers": snap_layers}
    for k in snap_targets:
        snapshot[f"true/{k}"] = Ys[k][snap_sl].float().cpu().numpy()

    for layer in layer_names + ["concat"]:
        preds = {k: [] for k in snap_targets} if layer in snapshot["layers"] else None
        for e in range(E):
            if layer == "concat":
                X = torch.cat([a_[e] for a_ in acts], 1).double()
            else:
                X = acts[int(layer)][e].double()
            W, b0 = _fit_affine(X[tr], Yall[tr], ridge)
            Yhat = X @ W + b0
            for k in tnames:
                sl = splits[k]
                if R - n_tr > 1:
                    add(e, "r2", layer, k, _r2(Yall[te, sl], Yhat[te, sl]))
                    add(e, "rmse", layer, k, (Yall[te, sl] - Yhat[te, sl]).pow(2).mean().sqrt())
            if preds is not None:
                for k in snap_targets:
                    preds[k].append(Yhat[snap_sl, splits[k]].float().cpu())
            if layer != "concat":
                Xc = X[tr] - X[tr].mean(0)
                lam = torch.linalg.eigvalsh(Xc.T @ Xc / max(1, n_tr - 1)).flip(0).clamp_min(0)
                cev = lam.cumsum(0) / lam.sum().clamp_min(1e-30)
                add(e, "n95", layer, "", int((cev < 0.95).sum()) + 1)
                add(e, "pr", layer, "", lam.sum() ** 2 / lam.pow(2).sum().clamp_min(1e-30))
                for grp in groups:
                    bases = dict(zip(grp, _subspaces(X[tr], [Yall[tr, splits[k]] for k in grp], ridge)))
                    for i, ka in enumerate(grp):
                        for kb in grp[i + 1:]:
                            (Qa, ra), (Qb, rb) = bases[ka], bases[kb]
                            ov = (Qa.T @ Qb).pow(2).sum() / max(1, min(ra, rb))
                            add(e, "overlap", layer, f"{ka}|{kb}", ov)
        if preds is not None:
            for k in snap_targets:
                snapshot[f"pred/{layer}/{k}"] = torch.stack(preds[k]).numpy().astype("float16")

    return out_rows, {"ce": ce_pos, **{f"kl/{k}": v for k, v in kl_pos.items()}}, snapshot


@torch.no_grad()
def reference_summary(ev: EvalSet) -> dict:
    """KL(bayes || ref) for every reference predictor: the excess loss a model would have
    if it implemented that reference exactly, i.e. the predicted height of a plateau."""
    pb = ev.refs["bayes"]
    out = {}
    for k, p in ev.refs.items():
        kl = (torch.xlogy(pb, pb) - pb * torch.log(p.clamp_min(1e-300))).sum(-1)  # (n, L)
        out[k] = {"mean": float(kl.mean()), "per_position": kl.mean(0).cpu().tolist()}
    ent = -(torch.xlogy(pb, pb)).sum(-1)
    out["_bayes_entropy"] = {"mean": float(ent.mean()), "per_position": ent.mean(0).cpu().tolist()}
    return out
