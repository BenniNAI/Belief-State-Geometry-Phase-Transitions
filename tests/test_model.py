import pytest
import torch

from bsg.model import EnsembleTransformer, ModelConfig, SingleModel, make_optimizer, per_member_loss

VARIANTS = [
    dict(),
    dict(act="gelu", norm="rmsnorm", pos="rotary", n_heads=2, d_head=8),
    dict(act="gated_gelu", norm="none", pos="none", n_heads=4),
    dict(attn_only=True, n_layers=2),
]


def make(E=3, seed=0, **kw):
    cfg = ModelConfig(**{**dict(n_layers=2, d_model=16, n_heads=1, d_mlp=32), **kw})
    return EnsembleTransformer(cfg, vocab_in=5, vocab_out=4, ctx_len=7, n_members=E, seed=seed)


@pytest.mark.parametrize("kw", VARIANTS)
def test_forward_shapes_and_causality(kw):
    m = make(**kw)
    x = torch.randint(0, 5, (3, 4, 7))
    logits, resid = m(x, return_resid=True)
    assert logits.shape == (3, 4, 7, 4) and len(resid) == m.cfg.n_layers + 1
    x2 = x.clone()
    x2[..., 5:] = (x2[..., 5:] + 1) % 5  # change the future
    assert torch.allclose(m(x2)[..., :5, :], logits[..., :5, :], atol=1e-5)


def test_member_init_independent_of_ensemble_size():
    big = make(E=3, seed=10)
    solo = make(E=1, seed=12)
    for (k, a), (_, b) in zip(big.state_dict().items(), solo.state_dict().items()):
        assert torch.equal(a[2], b[0]), k


@pytest.mark.parametrize("opt_name", ["sgd", "adam", "adamw"])
def test_members_train_independently(opt_name):
    """Training a 2-member ensemble must give member 0 the same weights as training it alone."""
    torch.manual_seed(0)
    ens, solo = make(E=2, seed=0), make(E=1, seed=0)
    x = torch.randint(0, 5, (2, 8, 7))
    y = torch.randint(0, 4, (2, 8, 7))
    oe = make_optimizer(ens.parameters(), opt_name, lr=1e-2, weight_decay=0.01)
    os_ = make_optimizer(solo.parameters(), opt_name, lr=1e-2, weight_decay=0.01)
    for _ in range(5):
        for m, o, xi, yi in [(ens, oe, x, y), (solo, os_, x[:1], y[:1])]:
            o.zero_grad()
            per_member_loss(m(xi), yi).sum().backward()
            o.step()
    for (k, a), (_, b) in zip(ens.state_dict().items(), solo.state_dict().items()):
        assert torch.allclose(a[0], b[0], atol=1e-5), k


def test_member_extraction_and_single_wrapper():
    m = make(E=3, act="gelu", pos="rotary", n_heads=2, d_head=8)
    x = torch.randint(0, 5, (3, 4, 7))
    full = m(x)
    assert torch.allclose(m.member(1)(x[1:2])[0], full[1], atol=1e-6)
    assert torch.allclose(SingleModel(m, 2)(x[2]), full[2], atol=1e-6)
