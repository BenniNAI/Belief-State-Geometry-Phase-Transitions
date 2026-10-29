"""Generating processes as token-labelled transition matrices.

Convention (as in Shai et al. 2024 and the Simplex papers):
    T[x, i, j] = P(emit token x and move to state j | current state i)

A predictive vector / belief eta (row vector) is updated by
    eta' = eta @ T[x] / (eta @ T[x] @ 1)

Everything here is plain numpy (float64). `bsg.beliefs` moves processes to torch
for fast sampling and filtering.

Structure metadata (`Process.structure`) records how a process was built, so that
analysis code can compute per-factor marginals, source weights, etc.:
    {"type": "single"}
    {"type": "product", "factor_dims": [...], "factor_vocab": [...],
     "couplings": [...], "noise": eps}
    {"type": "nonergodic", "components": [{"offset", "size", "weight", "name",
                                           "structure"}, ...]}
"""
from __future__ import annotations

import copy
import string
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Process:
    T: np.ndarray  # (V, d, d)
    eta0: np.ndarray  # (d,) initial belief over hidden states
    name: str = "process"
    structure: dict = field(default_factory=lambda: {"type": "single"})

    @property
    def vocab_size(self) -> int:
        return self.T.shape[0]

    @property
    def n_states(self) -> int:
        return self.T.shape[1]

    @property
    def is_hmm(self) -> bool:
        return bool((self.T >= -1e-12).all())

    @property
    def T_net(self) -> np.ndarray:
        return self.T.sum(axis=0)

    def check(self, atol: float = 1e-9) -> None:
        """Raise if the process is not a valid (G)HMM with a normalised initial state."""
        rows = self.T_net.sum(axis=1)
        if not np.allclose(rows, 1.0, atol=atol):
            raise ValueError(f"{self.name}: net transition rows do not sum to 1: {rows}")
        if not np.isclose(self.eta0.sum(), 1.0, atol=atol):
            raise ValueError(f"{self.name}: eta0 does not sum to 1")
        if self.is_hmm and (self.eta0 < -atol).any():
            raise ValueError(f"{self.name}: negative initial probabilities")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stationary_distribution(T_net: np.ndarray) -> np.ndarray:
    """Left eigenvector of the net transition matrix with eigenvalue 1, normalised to sum 1."""
    vals, vecs = np.linalg.eig(T_net.T)
    k = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, k])
    pi = pi / pi.sum()
    pi[np.abs(pi) < 1e-15] = 0.0
    return pi


def _single(T, name, eta0=None) -> Process:
    T = np.asarray(T, dtype=np.float64)
    if eta0 is None:
        eta0 = stationary_distribution(T.sum(axis=0))
    p = Process(T=T, eta0=np.asarray(eta0, dtype=np.float64), name=name,
                structure={"type": "single"})
    p.check()
    return p


# ---------------------------------------------------------------------------
# Elementary processes
# ---------------------------------------------------------------------------

def mess3(x: float = 0.15, alpha: float = 0.6, name: str | None = None) -> Process:
    """Mess3 (Marzen & Crutchfield 2017): 3 states, 3 tokens.

    T[tok][i, j] = (alpha if j == tok else beta) * (y if i == j else x),
    with beta = (1 - alpha) / 2 and y = 1 - 2x.
    Shai et al. (2024) used x = 0.05, alpha = 0.85.
    """
    beta = (1.0 - alpha) / 2.0
    y = 1.0 - 2.0 * x
    T = np.zeros((3, 3, 3))
    for tok in range(3):
        for i in range(3):
            for j in range(3):
                T[tok, i, j] = (alpha if j == tok else beta) * (y if i == j else x)
    return _single(T, name or f"mess3(x={x},a={alpha})")


def rrxor(name: str = "rrxor") -> Process:
    """Random-Random-XOR (5 states, 2 tokens), matrices from Shai et al. (2024) App. A.3."""
    T0 = np.array([[0, .5, 0, 0, 0],
                   [0, 0, 0, 0, .5],
                   [0, 0, 0, .5, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0]], dtype=float)
    T1 = np.array([[0, 0, .5, 0, 0],
                   [0, 0, 0, .5, 0],
                   [0, 0, 0, 0, .5],
                   [1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]], dtype=float)
    return _single(np.stack([T0, T1]), name)


def z1r(name: str = "z1r") -> Process:
    """Zero-One-Random: ...0 1 R 0 1 R... (states S0, S1, SR)."""
    T0 = np.array([[0, 1, 0], [0, 0, 0], [.5, 0, 0]], dtype=float)
    T1 = np.array([[0, 0, 0], [0, 0, 1], [.5, 0, 0]], dtype=float)
    return _single(np.stack([T0, T1]), name)


def coin(p: float = 0.5, name: str | None = None) -> Process:
    """Memoryless biased coin: token 0 with probability p."""
    return _single(np.array([[[p]], [[1.0 - p]]]), name or f"coin(p={p})")


def from_matrices(matrices, eta0=None, name: str = "custom") -> Process:
    return _single(np.asarray(matrices, dtype=float), name, eta0)


def relabel(proc: Process, perm, name: str | None = None) -> Process:
    """Same process with tokens renamed: token x is emitted as perm[x]. Useful as a coupling
    variant: a controller that relabels another factor's tokens is a strong interaction."""
    perm = [int(p) for p in perm]
    if sorted(perm) != list(range(proc.vocab_size)):
        raise ValueError("perm must be a permutation of the token alphabet")
    T = np.empty_like(proc.T)
    for x, px in enumerate(perm):
        T[px] = proc.T[x]
    p = Process(T=T, eta0=proc.eta0.copy(), name=name or f"{proc.name}{perm}",
                structure=copy.deepcopy(proc.structure))
    p.check()
    return p


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def decode_token(x, factor_vocab):
    """Joint token -> list of sub-tokens. Factor 1 is the least significant digit,
    matching Shai et al. (2026): x = z1 + V1*z2 + V1*V2*z3 + ..."""
    out = []
    for v in factor_vocab:
        out.append(x % v)
        x = x // v
    return out


def product(factors: list[Process], couplings: list[dict] | None = None,
            name: str | None = None) -> Process:
    """Joint process of several factors.

    Without couplings this is the tensor product T[x] = kron_n T_n[z_n]: the factors
    are conditionally independent and the Bayes-optimal belief stays a product of
    per-factor beliefs.

    Each coupling makes one factor's dynamics depend on another factor:
        {"controller": m, "target": n, "eps": e, "variants": [Process, ...],
         "control": "state" | "token"}
    For every value c of the controller, factor n uses
        (1 - e) * T_n + e * T_variant[c]
    - control="token": c is factor m's sub-token emitted on the same step (the
      dependency chain of Shai et al. 2026, Sec. 4.2). Because the controlling
      sub-token is observed, beliefs still factorise exactly.
    - control="state": c is factor m's *current hidden state*. This is unobserved, so
      beliefs become entangled and a factored representation is lossy.
    Each target factor may have at most one controller.
    """
    couplings = copy.deepcopy(couplings or [])
    K = len(factors)
    dims = [f.n_states for f in factors]
    vocab = [f.vocab_size for f in factors]
    if couplings and not all(f.is_hmm for f in factors):
        raise ValueError("couplings require HMM (non-negative) factors")
    if K > 8:
        raise ValueError("at most 8 factors supported")

    # Per-factor tensors A_n with axes (z_n, control, s_n, t_n).
    A = [f.T[:, None, :, :] for f in factors]
    ctrl_of = {}
    for cp in couplings:
        m, n, e = int(cp["controller"]), int(cp["target"]), float(cp["eps"])
        mode = cp.get("control", "state")
        if n in ctrl_of:
            raise ValueError(f"factor {n} already has a controller")
        if m == n:
            raise ValueError("a factor cannot control itself")
        variants = cp["variants"]
        n_ctrl = dims[m] if mode == "state" else vocab[m]
        if len(variants) != n_ctrl:
            raise ValueError(f"coupling {m}->{n} ({mode}) needs {n_ctrl} variants, got {len(variants)}")
        base = factors[n].T
        mats = []
        for v in variants:
            if v.T.shape != base.shape:
                raise ValueError("variant shape must match the target factor")
            mats.append((1.0 - e) * base + e * v.T)
        A[n] = np.stack(mats, axis=1)  # (V_n, n_ctrl, d_n, d_n)
        ctrl_of[n] = (m, mode)

    # einsum over 3K axes: z_1..z_K, s_1..s_K, t_1..t_K
    letters = string.ascii_letters
    z = letters[0:K]
    s = letters[K:2 * K]
    t = letters[2 * K:3 * K]
    operands, subs = [], []
    for n in range(K):
        if n in ctrl_of:
            m, mode = ctrl_of[n]
            c = s[m] if mode == "state" else z[m]
            subs.append(z[n] + c + s[n] + t[n])
            operands.append(A[n])
        else:
            subs.append(z[n] + s[n] + t[n])
            operands.append(A[n][:, 0])
    out = z[::-1] + s + t  # reversed token axes -> factor 1 least significant
    Tj = np.einsum(",".join(subs) + "->" + out, *operands, optimize=True)
    V = int(np.prod(vocab))
    d = int(np.prod(dims))
    Tj = Tj.reshape(V, d, d)

    if couplings:
        eta0 = stationary_distribution(Tj.sum(axis=0))
    else:
        eta0 = factors[0].eta0
        for f in factors[1:]:
            eta0 = np.kron(eta0, f.eta0)

    struct = {
        "type": "product",
        "factor_dims": dims,
        "factor_vocab": vocab,
        "factor_names": [f.name for f in factors],
        "couplings": [{k: v for k, v in cp.items() if k != "variants"} for cp in couplings],
        "noise": 0.0,
    }
    p = Process(T=Tj, eta0=eta0, name=name or " x ".join(f.name for f in factors),
                structure=struct)
    p.check()
    return p


def noisy_channel(proc: Process, eps: float, name: str | None = None) -> Process:
    """With probability 1-eps the emitted token is the true one; with probability eps it
    is replaced by a uniformly random *other* token (Shai et al. 2026, Eq. 33):
        T'[x] = (1-eps) T[x] + eps/(V-1) * sum_{x' != x} T[x']
    The latent dynamics and stationary distribution are unchanged; for a product
    process this breaks conditional independence of the factors."""
    V = proc.vocab_size
    if V < 2:
        raise ValueError("noisy channel needs at least 2 tokens")
    Tn = proc.T_net[None]
    T = (1.0 - eps) * proc.T + eps / (V - 1) * (Tn - proc.T)
    struct = copy.deepcopy(proc.structure)
    struct["noise"] = float(eps)
    p = Process(T=T, eta0=proc.eta0.copy(), name=name or f"{proc.name}+noise({eps})",
                structure=struct)
    p.check()
    return p


def nonergodic(components: list[Process], weights=None, name: str | None = None) -> Process:
    """Nonergodic composition (direct sum): one component is chosen at the start of each
    sequence with probability weights[n] and generates the whole sequence
    (Ray, Riechers & Shai 2026). All components must share the token alphabet."""
    V = components[0].vocab_size
    if any(c.vocab_size != V for c in components):
        raise ValueError("all components must have the same vocabulary size")
    K = len(components)
    w = np.full(K, 1.0 / K) if weights is None else np.asarray(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("component weights must sum to 1")
    d = sum(c.n_states for c in components)
    T = np.zeros((V, d, d))
    eta0 = np.zeros(d)
    comps, off = [], 0
    for c, wn in zip(components, w):
        k = c.n_states
        T[:, off:off + k, off:off + k] = c.T
        eta0[off:off + k] = wn * c.eta0
        comps.append({"offset": off, "size": k, "weight": float(wn), "name": c.name,
                      "structure": copy.deepcopy(c.structure)})
        off += k
    p = Process(T=T, eta0=eta0, name=name or " | ".join(c.name for c in components),
                structure={"type": "nonergodic", "components": comps})
    p.check()
    return p


# ---------------------------------------------------------------------------
# Building from config dictionaries
# ---------------------------------------------------------------------------

def build_process(spec: dict) -> Process:
    """Build a process from a (possibly nested) config dict, e.g.

        {"type": "mess3", "x": 0.05, "alpha": 0.85}
        {"type": "product", "factors": [...], "couplings": [...], "noise": 0.1}
        {"type": "nonergodic", "components": [...], "weights": [0.5, 0.5]}
        {"type": "noisy", "process": {...}, "eps": 0.1}
        {"type": "relabel", "process": {...}, "perm": [1, 2, 0]}
    """
    spec = dict(spec)
    kind = spec.pop("type")
    name = spec.pop("name", None)
    if kind == "mess3":
        return mess3(spec.get("x", 0.15), spec.get("alpha", 0.6), name)
    if kind == "rrxor":
        return rrxor(name or "rrxor")
    if kind == "z1r":
        return z1r(name or "z1r")
    if kind == "coin":
        return coin(spec.get("p", 0.5), name)
    if kind == "matrices":
        return from_matrices(spec["T"], spec.get("eta0"), name or "custom")
    if kind == "product":
        factors = [build_process(f) for f in spec["factors"]]
        couplings = []
        for cp in spec.get("couplings", []) or []:
            cp = dict(cp)
            cp["variants"] = [build_process(v) for v in cp["variants"]]
            couplings.append(cp)
        p = product(factors, couplings, name)
        noise = float(spec.get("noise", 0.0) or 0.0)
        return noisy_channel(p, noise, name) if noise > 0 else p
    if kind == "noisy":
        return noisy_channel(build_process(spec["process"]), float(spec["eps"]), name)
    if kind == "relabel":
        return relabel(build_process(spec["process"]), spec["perm"], name)
    if kind == "nonergodic":
        comps = [build_process(c) for c in spec["components"]]
        return nonergodic(comps, spec.get("weights"), name)
    raise ValueError(f"unknown process type: {kind}")
