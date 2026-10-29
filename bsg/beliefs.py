"""Sampling, exact belief filtering, reference predictors and probe targets (torch).

Indexing convention: for a token sequence x_1..x_n, `beliefs[:, t]` is the Bayes-optimal
belief after observing the first t tokens (t = 0..n), and `probs[:, t]` is the
corresponding next-token distribution, i.e. the prediction for x_{t+1}.
`align()` maps these onto model input positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .processes import Process

_MAX_ELEMS = 25_000_000  # chunk size limit for (n, d, d) gathers


class TorchProcess:
    """A Process moved to a torch device, with samplers and filters."""

    def __init__(self, proc: Process, device: str | torch.device = "cpu"):
        self.proc = proc
        self.device = torch.device(device)
        self.V = proc.vocab_size
        self.d = proc.n_states
        dev, f64 = self.device, torch.float64
        self.T = torch.tensor(proc.T, dtype=f64, device=dev)  # (V, d, d)
        self.eta0 = torch.tensor(proc.eta0, dtype=f64, device=dev)  # (d,)
        self.E = self.T.sum(-1)  # (V, d): P(token x | state s) = E[x, s]
        self.is_hmm = proc.is_hmm
        if self.is_hmm:
            self.init_cdf = self._cdf(self.eta0.clamp_min(0)[None])[0]
            # P(x | s): rows indexed by s
            self.tok_cdf = self._cdf(self.E.T.clamp_min(0))  # (d, V)
            # P(s' | s, x): rows indexed by s * V + x
            nxt = self.T.permute(1, 0, 2).reshape(self.d * self.V, self.d).clamp_min(0)
            self.next_cdf = self._cdf(nxt)  # (d*V, d)

    @staticmethod
    def _cdf(p: torch.Tensor) -> torch.Tensor:
        tot = p.sum(-1, keepdim=True)
        p = torch.where(tot > 0, p / tot.clamp_min(1e-300), torch.full_like(p, 1.0 / p.shape[-1]))
        c = p.cumsum(-1)
        c[..., -1] = 1.0
        return c.contiguous()

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, n_seq: int, n_tokens: int, generator: torch.Generator | None = None,
               return_states: bool = False):
        """Sample token sequences (n_seq, n_tokens). Hidden-state sampling for HMMs,
        predictive-vector sampling for generalised HMMs."""
        if not self.is_hmm:
            if return_states:
                raise ValueError("GHMMs have no hidden states to return")
            return self._sample_ghmm(n_seq, n_tokens, generator)
        dev = self.device
        rand = lambda: torch.rand(n_seq, 1, generator=generator, device=dev, dtype=torch.float64)
        state = torch.searchsorted(self.init_cdf, rand().squeeze(1)).clamp_(max=self.d - 1)
        tokens = torch.empty(n_seq, n_tokens, dtype=torch.long, device=dev)
        states = torch.empty(n_seq, n_tokens + 1, dtype=torch.long, device=dev) if return_states else None
        if return_states:
            states[:, 0] = state
        for t in range(n_tokens):
            x = torch.searchsorted(self.tok_cdf[state], rand()).squeeze(1).clamp_(max=self.V - 1)
            state = torch.searchsorted(self.next_cdf[state * self.V + x], rand()).squeeze(1).clamp_(max=self.d - 1)
            tokens[:, t] = x
            if return_states:
                states[:, t + 1] = state
        return (tokens, states) if return_states else tokens

    @torch.no_grad()
    def _sample_ghmm(self, n_seq, n_tokens, generator):
        eta = self.eta0.expand(n_seq, -1).clone()
        tokens = torch.empty(n_seq, n_tokens, dtype=torch.long, device=self.device)
        for t in range(n_tokens):
            p = (eta @ self.E.T).clamp_min(0)
            p = p / p.sum(-1, keepdim=True)
            x = torch.multinomial(p, 1, generator=generator).squeeze(1)
            tokens[:, t] = x
            eta = self._step(eta, x)
        return tokens

    # ----------------------------------------------------------------- filtering
    def _step(self, eta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """One Bayes update for a batch of beliefs eta (n, d) and tokens x (n,)."""
        n = eta.shape[0]
        chunk = max(1, _MAX_ELEMS // (self.d * self.d))
        outs = []
        for i in range(0, n, chunk):
            Tx = self.T[x[i:i + chunk]]  # (c, d, d)
            outs.append(torch.bmm(eta[i:i + chunk].unsqueeze(1), Tx).squeeze(1))
        new = torch.cat(outs) if len(outs) > 1 else outs[0]
        z = new.sum(-1, keepdim=True)
        return new / torch.where(z.abs() > 0, z, torch.ones_like(z))

    @torch.no_grad()
    def filter(self, tokens: torch.Tensor) -> torch.Tensor:
        """Exact beliefs (n, L+1, d), float64. beliefs[:, t] = after t tokens."""
        n, L = tokens.shape
        eta = self.eta0.expand(n, -1).clone()
        out = torch.empty(n, L + 1, self.d, dtype=torch.float64, device=self.device)
        out[:, 0] = eta
        for t in range(L):
            eta = self._step(eta, tokens[:, t])
            out[:, t + 1] = eta
        return out

    def predict(self, beliefs: torch.Tensor) -> torch.Tensor:
        """Next-token distribution(s) from belief(s): (..., d) -> (..., V)."""
        return beliefs @ self.E.T

    @torch.no_grad()
    def projected_filter(self, tokens: torch.Tensor, project) -> torch.Tensor:
        """Filter that applies `project` to the belief after every update (assumed-density
        filtering). Returns next-token probs (n, L+1, V)."""
        n, L = tokens.shape
        eta = project(self.eta0.expand(n, -1).clone())
        probs = torch.empty(n, L + 1, self.V, dtype=torch.float64, device=self.device)
        probs[:, 0] = self.predict(eta)
        for t in range(L):
            eta = project(self._step(eta, tokens[:, t]))
            probs[:, t + 1] = self.predict(eta)
        return probs.clamp_min(0)


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------

def blocks(structure: dict, d: int):
    """(offset, size, prior weight, sub-structure) for each ergodic component."""
    if structure.get("type") == "nonergodic":
        return [(c["offset"], c["size"], c["weight"], c["structure"]) for c in structure["components"]]
    return [(0, d, 1.0, structure)]


def factor_marginals(cond: torch.Tensor, dims) -> list[torch.Tensor]:
    """Per-factor marginals of a joint belief over a product state space. cond: (..., prod(dims))."""
    lead = cond.shape[:-1]
    x = cond.reshape(*lead, *dims)
    nl = len(lead)
    out = []
    for f in range(len(dims)):
        other = [nl + g for g in range(len(dims)) if g != f]
        out.append(x.sum(dim=other) if other else x)
    return out


def _kron_rows(ms: list[torch.Tensor]) -> torch.Tensor:
    joint = ms[0]
    for m in ms[1:]:
        joint = (joint.unsqueeze(-1) * m.unsqueeze(-2)).flatten(-2)
    return joint


def make_projection(structure: dict, d: int, eta0: torch.Tensor, factored: bool = False,
                    frozen: bool = False):
    """Projection used by reference predictors.
    factored: replace each component's conditional belief by the product of its factor
              marginals (mean-field / Boyen-Koller projection).
    frozen:   reset the weight on each nonergodic component to its prior (no source inference).
    """
    bl = blocks(structure, d)

    def project(eta: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(eta)
        for off, size, w0, st in bl:
            b = eta[:, off:off + size]
            w = b.sum(-1, keepdim=True)
            fallback = (eta0[off:off + size] / eta0[off:off + size].sum()).expand_as(b)
            cond = torch.where(w > 1e-300, b / w.clamp_min(1e-300), fallback)
            if factored and st.get("type") == "product":
                cond = _kron_rows(factor_marginals(cond, st["factor_dims"]))
            weight = torch.full_like(w, w0) if frozen else w
            out[:, off:off + size] = weight * cond
        return out

    return project


def has_product(structure: dict) -> bool:
    if structure.get("type") == "product":
        return True
    if structure.get("type") == "nonergodic":
        return any(c["structure"].get("type") == "product" for c in structure["components"])
    return False


@torch.no_grad()
def reference_predictors(tp: TorchProcess, tokens: torch.Tensor, beliefs: torch.Tensor) -> dict:
    """Next-token predictions (n, L+1, V) of exactly computable reference predictors.
    Each corresponds to a candidate intermediate stage of learning:
      bayes           exact Bayes-optimal predictor
      unigram         ignores context (stationary token distribution)
      last_token      uses only the most recent token
      factored        mean-field: tracks each factor's belief, ignores entanglement
      frozen_source   (nonergodic) tracks within-source beliefs, never updates source weights
      factored_frozen (nonergodic + product) both simplifications
    """
    n, L = tokens.shape
    st = tp.proc.structure
    refs = {"bayes": tp.predict(beliefs).clamp_min(0)}
    uni = tp.predict(tp.eta0)
    refs["unigram"] = uni.expand(n, L + 1, -1).clone()
    # last-token predictor: belief = eta0 updated on the last token only
    last = torch.empty_like(refs["unigram"])
    last[:, 0] = uni
    for t in range(L):
        last[:, t + 1] = tp.predict(tp._step(tp.eta0.expand(n, -1), tokens[:, t]))
    refs["last_token"] = last.clamp_min(0)
    nonerg = st.get("type") == "nonergodic"
    if has_product(st):
        refs["factored"] = tp.projected_filter(tokens, make_projection(st, tp.d, tp.eta0, factored=True))
    if nonerg:
        refs["frozen_source"] = tp.projected_filter(tokens, make_projection(st, tp.d, tp.eta0, frozen=True))
        if has_product(st):
            refs["factored_frozen"] = tp.projected_filter(
                tokens, make_projection(st, tp.d, tp.eta0, factored=True, frozen=True))
    for k, v in refs.items():
        refs[k] = v / v.sum(-1, keepdim=True)
    return refs


def belief_targets(structure: dict, beliefs: torch.Tensor) -> dict:
    """Probe targets derived from exact joint beliefs (..., d).
      belief                  full joint belief (for nonergodic: concatenation of w_n * eta_n)
      factor{f}, factors      per-factor marginals (product processes)
      weights                 posterior over sources (nonergodic)
      wsource{n}              weighted block w_n * eta_n of the joint belief
      source{n}, source{n}_factor{f}, source_factors   conditional beliefs per source
    """
    d = beliefs.shape[-1]
    out = {"belief": beliefs}
    st = structure
    if st.get("type") == "product":
        ms = factor_marginals(beliefs, st["factor_dims"])
        for f, m in enumerate(ms):
            out[f"factor{f}"] = m
        out["factors"] = torch.cat(ms, -1)
    elif st.get("type") == "nonergodic":
        ws, conds, facs = [], [], []
        for c, (off, size, _, sub) in enumerate(blocks(st, d)):
            b = beliefs[..., off:off + size]
            w = b.sum(-1, keepdim=True)
            cond = b / w.clamp_min(1e-300)
            ws.append(w)
            conds.append(cond)
            out[f"source{c}"] = cond
            out[f"wsource{c}"] = b  # weighted block w_c * eta_c (one telescoping cone)
            if sub.get("type") == "product":
                for f, m in enumerate(factor_marginals(cond, sub["factor_dims"])):
                    out[f"source{c}_factor{f}"] = m
                    facs.append(m)
        out["weights"] = torch.cat(ws, -1)
        out["source_beliefs"] = torch.cat(conds, -1)
        if facs:
            out["source_factors"] = torch.cat(facs, -1)
    return out


# ---------------------------------------------------------------------------
# Model-facing data
# ---------------------------------------------------------------------------

def n_tokens_for(ctx_len: int, bos: bool) -> int:
    return ctx_len if bos else ctx_len + 1


def align(tokens: torch.Tensor, bos: bool, bos_id: int):
    """Model inputs/targets plus the slice of belief indices that matches each input
    position. With BOS, input position t has seen t tokens; without, t+1 tokens."""
    if bos:
        col = torch.full((tokens.shape[0], 1), bos_id, dtype=tokens.dtype, device=tokens.device)
        inputs = torch.cat([col, tokens[:, :-1]], 1)
        targets = tokens
        sl = slice(0, tokens.shape[1])
    else:
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        sl = slice(1, tokens.shape[1])
    return inputs, targets, sl


@dataclass
class EvalSet:
    inputs: torch.Tensor  # (n, ctx) long
    targets: torch.Tensor  # (n, ctx) long
    beliefs: torch.Tensor  # (n, ctx, d) float64, aligned to input positions
    refs: dict  # name -> (n, ctx, V) float64
    probe_targets: dict  # name -> (n, ctx, k) float64
    bos: bool


@torch.no_grad()
def make_eval_set(tp: TorchProcess, n_seq: int, ctx_len: int, bos: bool, seed: int = 12345) -> EvalSet:
    g = torch.Generator(device=tp.device).manual_seed(seed)
    tokens = tp.sample(n_seq, n_tokens_for(ctx_len, bos), generator=g)
    beliefs = tp.filter(tokens)
    refs = reference_predictors(tp, tokens, beliefs)
    inputs, targets, sl = align(tokens, bos, tp.V)
    b = beliefs[:, sl]
    return EvalSet(inputs=inputs, targets=targets, beliefs=b,
                   refs={k: v[:, sl] for k, v in refs.items()},
                   probe_targets=belief_targets(tp.proc.structure, b), bos=bos)


class DataStream:
    """Training batches. dataset_size=None draws fresh sequences every step (as in
    Shai et al. 2024); otherwise minibatches are drawn from a fixed finite dataset."""

    def __init__(self, tp: TorchProcess, batch: int, ctx_len: int, bos: bool, seed: int = 0,
                 dataset_size: int | None = None, buffer_steps: int = 32):
        self.tp, self.batch, self.ctx, self.bos = tp, batch, ctx_len, bos
        self.g = torch.Generator(device=tp.device).manual_seed(seed)
        self.n_tok = n_tokens_for(ctx_len, bos)
        self.buffer_steps = max(1, buffer_steps)
        self.dataset = None
        if dataset_size:
            self.dataset = tp.sample(int(dataset_size), self.n_tok, generator=self.g)
        self._buf, self._i = None, 0

    def next(self):
        if self.dataset is not None:
            idx = torch.randint(0, self.dataset.shape[0], (self.batch,), generator=self.g,
                                device=self.tp.device)
            tokens = self.dataset[idx]
        else:
            if self._buf is None or self._i >= self.buffer_steps:
                self._buf = self.tp.sample(self.batch * self.buffer_steps, self.n_tok, generator=self.g)
                self._i = 0
            tokens = self._buf[self._i * self.batch:(self._i + 1) * self.batch]
            self._i += 1
        inputs, targets, _ = align(tokens, self.bos, self.tp.V)
        return inputs, targets
