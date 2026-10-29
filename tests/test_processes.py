import numpy as np
import pytest

from bsg.config import checkpoint_steps, load_config
from bsg.processes import (build_process, coin, decode_token, mess3, noisy_channel, nonergodic,
                           product, relabel, rrxor, stationary_distribution, z1r)

V3 = lambda: [mess3(.15, .6), mess3(.11, .79), mess3(.5, .6)]


def test_mess3_matches_shai_2024():
    T = mess3(0.05, 0.85).T
    TA = np.array([[0.765, 0.00375, 0.00375], [0.0425, 0.0675, 0.00375], [0.0425, 0.00375, 0.0675]])
    TB = np.array([[0.0675, 0.0425, 0.00375], [0.00375, 0.765, 0.00375], [0.00375, 0.0425, 0.0675]])
    TC = np.array([[0.0675, 0.00375, 0.0425], [0.00375, 0.0675, 0.0425], [0.00375, 0.00375, 0.765]])
    assert np.allclose(T[0], TA) and np.allclose(T[1], TB) and np.allclose(T[2], TC)


@pytest.mark.parametrize("p", [mess3(), mess3(0.05, 0.85), mess3(0.5, 0.66), rrxor(), z1r(), coin(0.3)])
def test_elementary_valid_and_stationary(p):
    p.check()
    assert np.allclose(p.eta0 @ p.T_net, p.eta0)


def test_rrxor_stationary():
    assert np.allclose(rrxor().eta0, stationary_distribution(rrxor().T_net))


def test_product_is_kronecker_with_factor1_least_significant():
    a, b = mess3(0.15, 0.6), mess3(0.11, 0.79)
    p = product([a, b])
    for x in range(9):
        z1, z2 = decode_token(x, [3, 3])
        assert np.allclose(p.T[x], np.kron(a.T[z1], b.T[z2]))
    assert np.allclose(p.eta0, np.kron(a.eta0, b.eta0))


def test_three_factor_product():
    fs = [mess3(.15, .6), mess3(.1, .8), z1r()]
    p = product(fs)
    assert p.vocab_size == 18 and p.n_states == 27
    x = 2 + 3 * 1 + 9 * 1
    assert np.allclose(p.T[x], np.kron(np.kron(fs[0].T[2], fs[1].T[1]), fs[2].T[1]))


def test_state_coupling_entries():
    v = V3()
    p = product([mess3(), mess3()], [{"controller": 0, "target": 1, "eps": 1.0, "variants": v, "control": "state"}])
    m = mess3()
    for (z1, z2, s1, s2, t1, t2) in [(2, 1, 1, 2, 0, 2), (0, 0, 2, 1, 1, 1), (1, 2, 0, 0, 2, 1)]:
        assert np.isclose(p.T[z1 + 3 * z2][s1 * 3 + s2, t1 * 3 + t2], m.T[z1][s1, t1] * v[s1].T[z2][s2, t2])
    p.check()


def test_token_coupling_entries_and_eps_interpolation():
    v = V3()
    for eps in (0.0, 0.4, 1.0):
        p = product([mess3(), mess3()], [{"controller": 0, "target": 1, "eps": eps, "variants": v, "control": "token"}])
        for z1 in range(3):
            for z2 in range(3):
                mix = (1 - eps) * mess3().T[z2] + eps * v[z1].T[z2]
                assert np.allclose(p.T[z1 + 3 * z2], np.kron(mess3().T[z1], mix))


def test_coupling_eps0_equals_independent():
    p0 = product([mess3(), mess3()], [{"controller": 0, "target": 1, "eps": 0.0, "variants": V3()}])
    assert np.allclose(p0.T, product([mess3(), mess3()]).T)
    assert np.allclose(p0.eta0, product([mess3(), mess3()]).eta0)


def test_coupling_validation():
    with pytest.raises(ValueError):
        product([mess3(), mess3()], [{"controller": 0, "target": 1, "eps": .5, "variants": V3()[:2]}])
    with pytest.raises(ValueError):
        product([mess3(), mess3()], [{"controller": 1, "target": 1, "eps": .5, "variants": V3()}])


def test_noisy_channel():
    p = product([mess3(), mess3()])
    n = noisy_channel(p, 0.2)
    assert np.allclose(n.T_net, p.T_net)
    x = 4
    others = sum(p.T[y] for y in range(9) if y != x)
    assert np.allclose(n.T[x], 0.8 * p.T[x] + 0.2 / 8 * others)
    assert n.structure["noise"] == 0.2 and n.structure["factor_dims"] == [3, 3]


def test_relabel():
    m = mess3(0.1, 0.8)
    r = relabel(m, [1, 2, 0])
    assert np.allclose(r.T[1], m.T[0]) and np.allclose(r.T[0], m.T[2])
    assert np.allclose(r.T_net, m.T_net)


def test_nonergodic_blocks():
    a, b = mess3(.15, .6), mess3(.5, .66)
    p = nonergodic([a, b], [0.3, 0.7])
    assert np.allclose(p.T[:, :3, 3:], 0) and np.allclose(p.T[:, 3:, :3], 0)
    assert np.allclose(p.T[:, :3, :3], a.T) and np.allclose(p.T[:, 3:, 3:], b.T)
    assert np.allclose(p.eta0, np.concatenate([0.3 * a.eta0, 0.7 * b.eta0]))
    with pytest.raises(ValueError):
        nonergodic([mess3(), rrxor()])


def test_build_process_nested():
    spec = {"type": "nonergodic", "weights": [0.5, 0.5], "components": [
        {"type": "product", "factors": [{"type": "mess3"}, {"type": "mess3", "x": 0.5}], "noise": 0.05},
        {"type": "product", "factors": [{"type": "mess3"}, {"type": "mess3"}],
         "couplings": [{"controller": 0, "target": 1, "eps": 0.5, "control": "state", "variants": [
             {"type": "mess3"},
             {"type": "relabel", "perm": [1, 2, 0], "process": {"type": "mess3"}},
             {"type": "relabel", "perm": [2, 0, 1], "process": {"type": "mess3"}}]}]}]}
    p = build_process(spec)
    assert p.n_states == 18 and p.vocab_size == 9
    comps = p.structure["components"]
    assert comps[0]["structure"]["noise"] == 0.05
    assert comps[1]["structure"]["couplings"][0]["eps"] == 0.5


@pytest.mark.parametrize("name", ["mess3_replication", "rrxor_replication", "factored_noise", "factored_coupling",
                                  "factored_two_channels", "nonergodic_two_mess3", "combined"])
def test_example_configs_build(name):
    cfg = load_config(f"configs/{name}.yaml")
    build_process(cfg["process"]).check()


def test_config_overrides():
    cfg = load_config("configs/combined.yaml",
                      ["train.lr=3e-4", "process.components.1.factors.0.x=0.2", "data.dataset_size=1000"])
    assert cfg["train"]["lr"] == 3e-4
    assert cfg["process"]["components"][1]["factors"][0]["x"] == 0.2
    assert cfg["data"]["dataset_size"] == 1000
    assert cfg["eval"]["probe_rows"] == 50_000  # defaults preserved


def test_checkpoint_steps():
    s = checkpoint_steps(1_000_000, dense_until=10, per_decade=10)
    assert s[:11] == list(range(11)) and s[-1] == 1_000_000
    assert len(s) < 80 and all(a < b for a, b in zip(s, s[1:]))
