"""Small decoder-only transformer, batched over an ensemble of independently
initialised members (seeds).

Every parameter has a leading ensemble dimension E. Members never interact: the
training loss is the *sum* of per-member mean losses, so gradients are per-member and
element-wise optimisers (SGD, Adam, AdamW) update each member exactly as if it were
trained alone. Training E seeds this way costs little more than one, because a model
this small leaves a GPU almost idle.

Member e is initialised from seed `seed + e`, independent of E.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_layers: int = 4
    d_model: int = 64
    n_heads: int = 1
    d_head: int | None = None  # default: d_model // n_heads
    d_mlp: int = 256
    act: str = "relu"  # relu | gelu | gated_gelu | gated_relu
    norm: str = "layernorm"  # layernorm | rmsnorm | none
    pos: str = "learned"  # learned | rotary | none
    init_std: float = 0.02  # std of all weight matrices (biases 0, norm gains 1)
    attn_only: bool = False


class _Block(nn.Module):
    def __init__(self, cfg: ModelConfig, E: int):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        dh = cfg.d_head or D // H
        M = cfg.d_mlp
        P = lambda *s: nn.Parameter(torch.empty(E, *s))
        if cfg.norm != "none":
            self.ln1_w = P(D)
            self.ln2_w = P(D)
            if cfg.norm == "layernorm":
                self.ln1_b = P(D)
                self.ln2_b = P(D)
        self.W_Q, self.W_K, self.W_V = P(D, H * dh), P(D, H * dh), P(D, H * dh)
        self.b_Q, self.b_K, self.b_V = P(H * dh), P(H * dh), P(H * dh)
        self.W_O, self.b_O = P(H * dh, D), P(D)
        if not cfg.attn_only:
            self.W_in, self.b_in = P(D, M), P(M)
            if cfg.act.startswith("gated"):
                self.W_gate = P(D, M)
            self.W_out, self.b_out = P(M, D), P(D)


class EnsembleTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_in: int, vocab_out: int, ctx_len: int,
                 n_members: int = 1, seed: int = 0):
        super().__init__()
        self.cfg, self.E, self.ctx_len = cfg, n_members, ctx_len
        self.vocab_in, self.vocab_out = vocab_in, vocab_out
        D, E = cfg.d_model, n_members
        self.d_head = cfg.d_head or D // cfg.n_heads
        self.W_E = nn.Parameter(torch.empty(E, vocab_in, D))
        if cfg.pos == "learned":
            self.W_pos = nn.Parameter(torch.empty(E, ctx_len, D))
        self.blocks = nn.ModuleList([_Block(cfg, E) for _ in range(cfg.n_layers)])
        if cfg.norm != "none":
            self.lnf_w = nn.Parameter(torch.empty(E, D))
            if cfg.norm == "layernorm":
                self.lnf_b = nn.Parameter(torch.empty(E, D))
        self.W_U = nn.Parameter(torch.empty(E, D, vocab_out))
        self.b_U = nn.Parameter(torch.empty(E, vocab_out))
        if cfg.pos == "rotary":
            if self.d_head % 2:
                raise ValueError("rotary embeddings need an even d_head")
            inv = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
            ang = torch.arange(ctx_len).float()[:, None] * inv[None]
            self.register_buffer("rot_cos", ang.cos(), persistent=False)
            self.register_buffer("rot_sin", ang.sin(), persistent=False)
        self.init_weights(seed)

    # ---------------------------------------------------------------- init
    @torch.no_grad()
    def init_weights(self, seed: int):
        named = sorted(self.named_parameters(), key=lambda kv: kv[0])
        for e in range(self.E):
            g = torch.Generator().manual_seed(seed + e)
            for name, p in named:
                leaf = name.split(".")[-1]
                if leaf.startswith("W_"):
                    vals = torch.randn(p.shape[1:], generator=g) * self.cfg.init_std
                elif leaf.endswith("_w"):  # norm gains
                    vals = torch.ones(p.shape[1:])
                else:  # biases
                    vals = torch.zeros(p.shape[1:])
                p[e].copy_(vals.to(p.device))

    # ------------------------------------------------------------- pieces
    @staticmethod
    def _lin(x, W, b=None):
        E, F_ = W.shape[0], W.shape[-1]
        out = torch.bmm(x.reshape(E, -1, x.shape[-1]), W).reshape(*x.shape[:-1], F_)
        return out if b is None else out + b.view(E, *([1] * (x.dim() - 2)), F_)

    def _norm(self, x, w, b=None):
        if self.cfg.norm == "none":
            return x
        shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (x.shape[-1],)
        if self.cfg.norm == "layernorm":
            x = F.layer_norm(x, (x.shape[-1],))
            return x * w.view(shape) + b.view(shape)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x * w.view(shape)

    def _rotary(self, x):  # x: (N, H, L, dh)
        L = x.shape[2]
        cos, sin = self.rot_cos[:L], self.rot_sin[:L]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

    def _attn(self, blk, h):
        E, B, L, _ = h.shape
        H, dh = self.cfg.n_heads, self.d_head
        q = self._lin(h, blk.W_Q, blk.b_Q).view(E * B, L, H, dh).transpose(1, 2)
        k = self._lin(h, blk.W_K, blk.b_K).view(E * B, L, H, dh).transpose(1, 2)
        v = self._lin(h, blk.W_V, blk.b_V).view(E * B, L, H, dh).transpose(1, 2)
        if self.cfg.pos == "rotary":
            q, k = self._rotary(q), self._rotary(k)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        z = z.transpose(1, 2).reshape(E, B, L, H * dh)
        return self._lin(z, blk.W_O, blk.b_O)

    def _mlp(self, blk, h):
        a = self.cfg.act
        pre = self._lin(h, blk.W_in, blk.b_in)
        if a == "relu":
            act = F.relu(pre)
        elif a == "gelu":
            act = F.gelu(pre)
        elif a == "gated_gelu":
            act = F.gelu(self._lin(h, blk.W_gate)) * pre
        elif a == "gated_relu":
            act = F.relu(self._lin(h, blk.W_gate)) * pre
        else:
            raise ValueError(f"unknown activation {a}")
        return self._lin(act, blk.W_out, blk.b_out)

    # ------------------------------------------------------------ forward
    def forward(self, inputs: torch.Tensor, return_resid: bool = False):
        """inputs: (E, B, L) token ids. Returns logits (E, B, L, V_out) and, if requested,
        the residual stream [embeddings, after block 1, ..., after block n] (pre final norm),
        each (E, B, L, D)."""
        E, B, L = inputs.shape
        idx = torch.arange(E, device=inputs.device).view(E, 1, 1)
        h = self.W_E[idx, inputs]
        if self.cfg.pos == "learned":
            h = h + self.W_pos[:, None, :L, :]
        resid = [h] if return_resid else None
        for blk in self.blocks:
            n1 = self._norm(h, getattr(blk, "ln1_w", None), getattr(blk, "ln1_b", None))
            h = h + self._attn(blk, n1)
            if not self.cfg.attn_only:
                n2 = self._norm(h, getattr(blk, "ln2_w", None), getattr(blk, "ln2_b", None))
                h = h + self._mlp(blk, n2)
            if return_resid:
                resid.append(h)
        hf = self._norm(h, getattr(self, "lnf_w", None), getattr(self, "lnf_b", None))
        logits = self._lin(hf, self.W_U, self.b_U)
        return (logits, resid) if return_resid else logits

    # ------------------------------------------------------------ members
    def member(self, e: int) -> "EnsembleTransformer":
        """A standalone copy of member e (an ensemble of size 1)."""
        m = EnsembleTransformer(self.cfg, self.vocab_in, self.vocab_out, self.ctx_len, 1)
        m.load_state_dict({k: v[e:e + 1].clone() for k, v in self.state_dict().items()})
        return m.to(self.W_E.device)


class SingleModel(nn.Module):
    """Wraps one member as an ordinary model: (B, L) tokens -> (B, L, V) logits.
    Handy for tools that expect a plain nn.Module, e.g. devinterp's LLC estimators."""

    def __init__(self, ens: EnsembleTransformer, e: int = 0):
        super().__init__()
        self.inner = ens.member(e)

    def forward(self, x):
        return self.inner(x.unsqueeze(0))[0]


def per_member_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy per member. logits (E, B, L, V), targets (E, B, L) -> (E,)."""
    E, V = logits.shape[0], logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none")
    return ce.view(E, -1).mean(1)


def make_optimizer(params, name: str = "adam", lr: float = 1e-3, weight_decay: float = 0.0,
                   betas=(0.9, 0.999), momentum: float = 0.0):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name}")


def count_params(model: EnsembleTransformer) -> int:
    """Parameters per member."""
    return sum(p[0].numel() for p in model.parameters())
