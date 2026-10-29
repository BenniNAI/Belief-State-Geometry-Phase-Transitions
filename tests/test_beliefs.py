import itertools

import numpy as np
import pytest
import torch

from bsg.beliefs import (DataStream, TorchProcess, align, belief_targets, make_eval_set,
                         reference_predictors)
from bsg.processes import mess3, noisy_channel, nonergodic, product, relabel, rrxor, z1r

torch.set_default_dtype(torch.float32)


def brute_force_belief(proc, xs):
    """P(s_t | x_1..x_t) by summing over all hidden paths (independent of the filter code)."""
    d = proc.n_states
    post = np.zeros(d)
    for path in itertools.product(range(d), repeat=len(xs) + 1):
        p = proc.eta0[path[0]]
        for k, x in enumerate(xs):
            p *= proc.T[x, path[k], path[k + 1]]
        post[path[-1]] += p
    return post / post.sum()


@pytest.mark.parametrize("proc", [mess3(0.05, 0.85), z1r(), rrxor()])
def test_filter_matches_brute_force(proc):
    tp = TorchProcess(proc)
    g = torch.Generator().manual_seed(0)
    toks = tp.sample(5, 4, generator=g)
    bel = tp.filter(toks)
    for i in range(5):
        for t in range(5):
            xs = toks[i, :t].tolist()
            assert np.allclose(bel[i, t].numpy(), brute_force_belief(proc, xs), atol=1e-10)


def _pair_freq_check(tp, n=200_000):
    g = torch.Generator().manual_seed(1)
    toks = tp.sample(n, 2, generator=g)
    V = tp.V
    emp = torch.bincount(toks[:, 0] * V + toks[:, 1], minlength=V * V).double() / n
    T, eta0 = tp.proc.T, tp.proc.eta0
    exact = np.array([[eta0 @ T[a] @ T[b] @ np.ones(tp.d) for b in range(V)] for a in range(V)]).ravel()
    se = np.sqrt(exact * (1 - exact) / n) + 1e-12
    assert np.all(np.abs(emp.numpy() - exact) < 5 * se + 1e-4)


@pytest.mark.parametrize("proc", [mess3(0.05, 0.85), rrxor(), nonergodic([mess3(.15, .6), mess3(.5, .66)])])
def test_sampler_pair_frequencies(proc):
    _pair_freq_check(TorchProcess(proc))


def test_predictive_vector_sampler_matches():
    tp = TorchProcess(mess3(0.05, 0.85))
    tp.is_hmm = False  # force the GHMM sampler path
    _pair_freq_check(tp, n=100_000)


def test_bayes_loss_equals_mean_entropy():
    """E[-log p(x_t+1 | past)] = E[H(p)] holds only if sampler and filter agree."""
    tp = TorchProcess(product([mess3(0.1, 0.8), mess3(0.05, 0.85)]))
    g = torch.Generator().manual_seed(2)
    toks = tp.sample(20_000, 12, generator=g)
    probs = tp.predict(tp.filter(toks))[:, :-1]
    nll = -torch.log(probs.gather(-1, toks.unsqueeze(-1)).squeeze(-1))
    ent = -(probs * probs.log()).sum(-1)
    diff = (nll - ent).mean()
    se = (nll - ent).std() / np.sqrt(nll.numel())
    assert abs(float(diff)) < 4 * float(se) + 1e-4


def test_nonergodic_states_stay_in_one_component():
    tp = TorchProcess(nonergodic([mess3(.15, .6), mess3(.5, .66)]))
    _, states = tp.sample(500, 30, generator=torch.Generator().manual_seed(3), return_states=True)
    comp = states >= 3
    assert torch.all(comp == comp[:, :1])


def _kl(p, q):
    return float((torch.xlogy(p, p) - p * torch.log(q.clamp_min(1e-300))).sum(-1).mean())


def _refs(proc, n=512, L=16):
    tp = TorchProcess(proc)
    toks = tp.sample(n, L, generator=torch.Generator().manual_seed(4))
    bel = tp.filter(toks)
    return reference_predictors(tp, toks, bel), bel, tp


SHARP = lambda: mess3(0.05, 0.85)
RELABEL = lambda: [SHARP(), relabel(SHARP(), [1, 2, 0]), relabel(SHARP(), [2, 0, 1])]


def test_factored_reference_exact_for_independent_factors():
    refs, _, _ = _refs(product([SHARP(), mess3(0.15, 0.6)]))
    assert _kl(refs["bayes"], refs["factored"]) < 1e-12


def test_token_coupling_keeps_beliefs_factored_state_coupling_does_not():
    tok = product([SHARP(), SHARP()], [{"controller": 0, "target": 1, "eps": 1.0, "variants": RELABEL(), "control": "token"}])
    st = product([SHARP(), SHARP()], [{"controller": 0, "target": 1, "eps": 1.0, "variants": RELABEL(), "control": "state"}])
    r_tok, _, _ = _refs(tok)
    r_st, _, _ = _refs(st)
    assert _kl(r_tok["bayes"], r_tok["factored"]) < 1e-10
    assert _kl(r_st["bayes"], r_st["factored"]) > 1e-4


def test_noise_entangles_factors():
    refs, _, _ = _refs(noisy_channel(product([SHARP(), SHARP()]), 0.3))
    assert _kl(refs["bayes"], refs["factored"]) > 1e-4


def test_reference_ordering_and_frozen_source():
    a, b = mess3(.15, .6), mess3(.5, .66)
    refs, bel, tp = _refs(nonergodic([a, b]), L=32)
    assert torch.allclose(refs["frozen_source"][:, 0], refs["bayes"][:, 0])
    assert _kl(refs["bayes"], refs["frozen_source"]) > 1e-4
    assert _kl(refs["bayes"], refs["unigram"]) > _kl(refs["bayes"], refs["frozen_source"])
    same, _, _ = _refs(nonergodic([mess3(.15, .6), mess3(.15, .6)]))
    assert _kl(same["bayes"], same["frozen_source"]) < 1e-12


def test_nonergodic_independent_products_factored_reference_exact():
    p = nonergodic([product([SHARP(), mess3(.15, .6)]), product([mess3(.3, .7), SHARP()])])
    refs, _, _ = _refs(p)
    assert _kl(refs["bayes"], refs["factored"]) < 1e-12
    assert _kl(refs["bayes"], refs["factored_frozen"]) > 1e-6


def test_belief_targets_nonergodic():
    p = nonergodic([product([SHARP(), mess3(.15, .6)]), product([mess3(.3, .7), SHARP()])], [0.4, 0.6])
    _, bel, _ = _refs(p, n=64)
    t = belief_targets(p.structure, bel)
    assert torch.allclose(t["weights"].sum(-1), torch.ones(64, 17, dtype=torch.float64))
    assert torch.allclose(t["wsource1"], t["weights"][..., 1:2] * t["source1"])
    assert torch.allclose(t["source0_factor0"].sum(-1), torch.ones(64, 17, dtype=torch.float64))
    assert t["source_factors"].shape[-1] == 12 and t["belief"].shape[-1] == 18


def test_align_conventions():
    toks = torch.arange(12).view(2, 6)
    inp, tgt, sl = align(toks, bos=True, bos_id=99)
    assert inp[0].tolist() == [99, 0, 1, 2, 3, 4] and tgt[0].tolist() == [0, 1, 2, 3, 4, 5]
    assert (sl.start, sl.stop) == (0, 6)
    inp, tgt, sl = align(toks, bos=False, bos_id=99)
    assert inp[0].tolist() == [0, 1, 2, 3, 4] and tgt[0].tolist() == [1, 2, 3, 4, 5]
    assert (sl.start, sl.stop) == (1, 6)


@pytest.mark.parametrize("bos", [True, False])
def test_eval_set_alignment(bos):
    tp = TorchProcess(mess3(0.05, 0.85))
    ev = make_eval_set(tp, 64, 8, bos)
    assert ev.inputs.shape == (64, 8) and ev.beliefs.shape == (64, 8, 3)
    # the Bayes reference at each input position predicts that position's target
    p = tp.predict(ev.beliefs)
    assert torch.allclose(p / p.sum(-1, keepdim=True), ev.refs["bayes"])
    if bos:
        assert torch.allclose(ev.beliefs[:, 0], tp.eta0.expand(64, -1))


def test_datastream_finite_dataset():
    tp = TorchProcess(mess3())
    ds = DataStream(tp, 16, 8, bos=True, dataset_size=10)
    seen = {tuple(r.tolist()) for _ in range(50) for r in ds.next()[1]}
    assert len(seen) <= 10
    fresh = DataStream(tp, 16, 8, bos=True)
    a, _ = fresh.next()
    assert a.shape == (16, 8) and int(a[0, 0]) == 3
