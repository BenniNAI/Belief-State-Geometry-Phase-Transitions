# Belief state geometry: phase transitions in learning

Code for training small transformers on hidden-Markov-model data with known belief geometry, and for tracking order parameters of how that geometry is learned: factored vs joint representations (Shai et al. 2026, *Transformers learn factored representations*), source inference in nonergodic mixtures (Ray, Riechers & Shai 2026, *The geometry of nonergodic composition*), and the original Mess3/RRXOR results (Shai et al. 2024).

Everything is in PyTorch. Several seeds are trained in parallel inside one model, so sweeps with many seeds cost little more than single runs.

## Setup (GPU machine)

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# PyTorch with CUDA: get the current command from https://pytorch.org/get-started/locally/
# RTX 50-series (Blackwell, e.g. RTX 5070) needs PyTorch >= 2.7 built for CUDA 12.8 or newer
# (cu128 or later); older builds such as cu126 fail with "no kernel image is available".
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q                            # 68 tests, ~15 s; run from this folder
```

On Windows, plain `pip install torch` gives a CPU-only build, so use the index URL. Runs use the GPU automatically when one is visible (override with `--set device=cpu` or `device=cuda:1`).

**GPU memory.** Training memory grows with `n_seeds x batch_size x ctx_len`. `train.grad_accum: k` splits each batch into k micro-batches with the same gradient and roughly 1/k of the activation memory. The progress lines and `done.json` report peak GPU memory. Approximate training memory, measured on CPU, so treat as rough:

| Config | As published | Shipped setting |
|---|---|---|
| `mess3_replication`, `rrxor_replication` | < 0.2 GB | same |
| `factored_coupling`, `factored_two_channels` | ~6-8 GB | same |
| `factored_noise` (batch 25000) | ~20 GB | `grad_accum: 4`, ~6 GB |
| `nonergodic_two_mess3` (context 128) | ~12 GB | `grad_accum: 2`, ~6 GB |
| `combined` (4 seeds, context 64) | ~14 GB | `grad_accum: 2`, ~7 GB |

With 12 GB these all fit. On larger GPUs set `train.grad_accum=1` for a little more speed.

## Quick start

```bash
python -m bsg.train configs/mess3_replication.yaml
python -m bsg.plot curves   runs/mess3_replication
python -m bsg.plot geometry runs/mess3_replication --target belief   # across training, all layers concatenated
python -m bsg.plot layers   runs/mess3_replication --target belief   # one checkpoint, layer by layer
```

`geometry` takes `--layer concat` (default), `--layer final`, or a layer index (`0` = embeddings, `k` = after block k). Examples of each plot, from short test runs, are in [docs/example_plots](docs/example_plots/README.md).

Any config value can be overridden from the command line with dotted keys, including inside the process spec:

```bash
python -m bsg.train configs/combined.yaml --set train.steps=5000 process.components.1.factors.0.x=0.2
```

## How the experiments map onto the project

| Step | Config | What to look at |
|---|---|---|
| Replications | `mess3_replication`, `rrxor_replication`, `nonergodic_two_mess3`, `factored_noise` | fractal in the final layer; RRXOR only in `concat`; telescoping cones (`belief`, `weights`); factored dimension |
| 1. Plateau length vs interaction strength | `factored_noise`, `factored_coupling`; sweeps `noise_plateau`, `coupling_plateau` | KL to `factored`; `python -m bsg.summarize runs/<sweep> --ref factored --x <key>` |
| 2. One transition or several? | `factored_two_channels` (strong channel 0→1, weak 1→2) | overlaps `factor0|factor1` vs `factor1|factor2`; per-factor R²; KL curves for steps |
| 3. Order of transitions | `combined`; sweep `combined_phase_diagram` (interaction strength × source distinguishability) | exit times at `factored` vs `frozen_source` |

Sweeps run every grid point in turn and skip finished runs, so they can be restarted:

```bash
python -m bsg.sweep configs/sweeps/coupling_plateau.yaml
python -m bsg.summarize runs/coupling_plateau --ref factored --x process.couplings.0.eps
```

## Generating processes

Defined in `bsg/processes.py`; configs describe them as nested dictionaries.

| `type` | Meaning |
|---|---|
| `mess3` (`x`, `alpha`) | 3-state, 3-token HMM with fractal belief geometry |
| `rrxor`, `z1r`, `coin` (`p`) | other classic examples |
| `relabel` (`process`, `perm`) | same process with tokens renamed |
| `product` (`factors`, `couplings`, `noise`) | joint process of several factors; joint token = z₁ + V₁z₂ + V₁V₂z₃ + … |
| `noisy` (`process`, `eps`) | with probability eps the token is replaced by a random other token (Shai et al. 2026, Eq. 33) |
| `nonergodic` (`components`, `weights`) | one component is drawn per sequence and generates all of it |

A **coupling** makes one factor's dynamics depend on another: for each value c of the controller, the target factor uses `(1 - eps) * base + eps * variants[c]`.

- With `control: state` the controller is the other factor's **hidden state**. It is unobserved, so beliefs become entangled and a factored representation loses information.
- With `control: token` the controller is the other factor's emitted sub-token, as in the dependency chain of Shai et al. (2026). It is observed, so beliefs still factorise exactly.

The test suite checks both claims.

Each target factor can have one controller, and couplings can be chained (see `factored_two_channels`). Using relabelled copies of the target as variants gives a strong interaction whose size is set by eps.

## Reference predictors and order parameters

Each candidate stage of learning corresponds to a predictor that can be computed exactly from the generator (`bsg/beliefs.py`):

| Reference | Behaviour |
|---|---|
| `bayes` | exact Bayes-optimal predictor |
| `unigram` | ignores context |
| `last_token` | uses only the latest token |
| `factored` | tracks each factor's belief separately and ignores entanglement. It applies the exact update, then projects back to a product of marginals each step (mean-field filtering). |
| `frozen_source` | nonergodic only: tracks beliefs within each source but never updates which source is active |
| `factored_frozen` | both simplifications |

`reference.json` stores KL(bayes ‖ ref) for each one: the excess loss a model would have if it implemented that reference exactly, i.e. the predicted plateau height.

At every checkpoint `metrics.csv` records, per seed:

- `kl` with target `bayes`: the excess loss.
- `kl` with other targets: KL(ref ‖ model). When this is small, the model behaves like that reference.
- `r2` / `rmse`: held-out affine probes from each layer (`0` = embeddings, `k` = after block k, `concat` = all layers) to belief targets: `belief`, `factor{f}`, `factors`, `weights`, `wsource{n}` (one cone), `source{n}`, `source{n}_factor{f}`, `source_beliefs`, `source_factors`.
- `n95` and `pr`: number of principal components for 95% of variance, and the participation ratio.
- `overlap`: overlap between the activation subspaces of factors (or sources). 0 means orthogonal, i.e. a factored representation. All targets in a group are fitted jointly, so correlated beliefs don't create false overlap.

`per_position.npz` has KL and loss per context position at every checkpoint. `snapshots/` holds held-out probe predictions from every layer and from `concat` (set `eval.snapshot_layers`, e.g. `["final", "concat"]`, to save fewer).

## Things worth knowing

- **Loss gaps can be tiny.** With the factor parameters used in Shai et al. 2026 (Mess3, x = 0.15, alpha = 0.6), the factored predictor is only ~3e-6 (noise 0.1) to ~8e-6 (noise 0.3) nats/token worse than Bayes, so there is little pressure to leave the factored solution. Measured gaps for other settings (context 16):

  | Setting | eps 0.1 | 0.3 | 0.6 | 1.0 |
  |---|---|---|---|---|
  | state coupling, relabelled variants, Mess3 (0.15, 0.6) | 9e-7 | 1.3e-5 | 1.5e-4 | 1.1e-3 |
  | same, Mess3 (0.1, 0.8) (default in configs) | 3e-5 | 2e-4 | 1e-3 | 5.7e-3 |
  | same, Mess3 (0.05, 0.85) | 1.2e-4 | 7e-4 | 2.6e-3 | 1.2e-2 |

  The gap rises steeply with eps (local exponents between about 2 and 4), so it's worth plotting plateau lengths against the gap as well as against eps.
- **The geometry need not be in any single layer.** When distinct belief states share a next-token distribution (RRXOR, and potentially entangled factors), the final layer can drop the distinctions, and the geometry is found only in the concatenation of layers (Shai et al. 2024). The order parameters are recorded for every layer and for `concat`, and `bsg.plot layers` shows where the geometry sits at a given checkpoint. `concat` has more features than any single layer, so compare it with step 0 and the embedding layer.
- **Probe R² starts high.** At initialisation, recent-token features already predict much of the belief (R² ≈ 0.8–0.93 for Mess3). Use the layer-0 curve and step 0 as the baseline, and prefer the KL order parameters for locating transitions.
- **`bsg.summarize` thresholds are a first pass.** Entry means KL(ref ‖ model) < frac × gap; exit means excess loss < frac × gap. Check them against `curves.png` before trusting a scaling exponent.
- **Confounders.** Plateau lengths depend on `model.init_std`, `train.lr` and the optimizer as well as the interaction. The sweeps vary at least one of these; report times in lr × steps as well (`plateau_lr_steps`).
- **Finite data.** `data.dataset_size` trains on a fixed dataset instead of fresh samples, which is the setting for SLT-style Bayesian transitions in n.
- **Precision.** Everything runs in full float32 (PyTorch's default; TF32 is not enabled). Keep it that way for experiments whose loss gaps are 1e-5 nats/token or smaller, where reduced-precision matmuls could add noise comparable to the effect.
- **Speed.** On Linux with CUDA, `--set train.compile=true` usually speeds up small models (not recommended on Windows). Evaluation cost grows with `eval.n_sequences`, `eval.probe_rows` and the number of checkpoints (`eval.dense_until`, `eval.per_decade`).
- **LLC.** `bsg.model.SingleModel(ensemble, member)` wraps one seed as an ordinary `nn.Module` mapping (batch, seq) tokens to logits, which is what devinterp's LLC estimators expect. Checkpoints are in `runs/<name>/checkpoints/`. LLC tracking isn't built in yet.

## Layout

```
bsg/processes.py   generating processes (numpy)
bsg/beliefs.py     sampling, exact filtering, reference predictors, probe targets
bsg/model.py       transformer with an ensemble dimension; optimisers
bsg/analysis.py    order parameters at each checkpoint
bsg/train.py       training loop and outputs
bsg/sweep.py       grids of runs
bsg/plot.py        curves and geometry figures
bsg/summarize.py   plateau / transition times across runs
configs/           experiment configs; configs/sweeps/ grids
docs/example_plots/  example figures from short test runs
tests/             pytest suite
```

Conventions: `beliefs[:, t]` is the belief after t tokens. With `data.bos: true`, input position t has seen t tokens. Without BOS (as in Shai et al. 2024), it has seen t + 1 tokens, starting from the stationary distribution.

## Licence

MIT. See [LICENSE](LICENSE).
