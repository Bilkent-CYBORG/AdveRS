# AdveRS: Robust Satisficing GP Bandits Under Adversarial Attacks

Code for the paper:

> A. Saday\*, Y. C. Yıldırım\*, C. Tekin. *Robust Satisficing Gaussian Process Bandits Under Adversarial Attacks.*
> NeurIPS 2025. [arXiv:2506.01625](https://arxiv.org/abs/2506.01625)

The learner picks a point, an adversary perturbs it, and the learner observes a noisy reward at the perturbed point.
The AdveRS algorithms (RS-1, RS-2, RS-G) aim to keep the reward above a threshold τ under unknown perturbations.
They are compared with StableOpt (RO), run with several ambiguity radii r.

## Setup

Uses [uv](https://docs.astral.sh/uv/) with the pinned versions in `uv.lock` (Python 3.10):

```bash
uv sync
```

Then use `uv run python ...`, or activate `.venv`.

## Code

| File | Contents |
|---|---|
| `BO_loop.py` | GP-UCB loops with an adversary: `BO_loop_RS`, `BO_loop_RO` and their contextual (insulin) versions |
| `adv_RS1.py`, `adv_RS2.py` | RS-1 / RS-G and RS-2 acquisition rules |
| `adv_RO.py` | StableOpt (RO) acquisition rule |
| `adversarial_models.py` | Attacks: random and worst case within an ε-ball |
| `synthetic.py` | Synthetic function experiment (Figs 3, 4, 8, 9) |
| `inv_pend.py` | Inverted pendulum controller tuning (Fig 7) |
| `insulin_dosage.py` | Insulin dosage with perturbed carbohydrate context (Figs 5, 10) |
| `sensitivity_exp.py` | Effect of τ and r (Table 2) |
| `util_arrays/` | Precomputed objectives: pendulum cost grid, insulin PBG grid |
| `helpers/run_parallel.py` | Runs the scripts in parallel chunks and merges the results |
| `helpers/plot_figures.py` | Draws the paper figures from `output_arrays/` |

Attack modes: `0` random within the ε-ball, `1` worst point under the learner's LCB, `2` worst point under the
true function, `3` Gaussian perturbation (Table 2).

## Reproducing the paper

Each script runs on its own, e.g.

```bash
uv run python synthetic.py                  # Fig 3a / 4
uv run python synthetic.py --epsilon 1.5    # Fig 3b
uv run python synthetic.py --kernel rbf     # Figs 8 / 9 (misspecified kernel)
uv run python inv_pend.py                   # Fig 7
uv run python insulin_dosage.py             # Fig 5  (add --thompson for Fig 10)
uv run python sensitivity_exp.py            # Table 2
```

The scripts run all algorithms and attacks one after another, 100 runs each. That takes a long time,
the pendulum RO runs most of all. To use many cores, run everything through the helper, e.g. in tmux:

```bash
uv run python helpers/run_parallel.py --workers 28    # all experiments; rerun the same command to resume
uv run python helpers/plot_figures.py                  # figures/
```

Results are random: the original runs had no fixed seed, so a rerun matches the paper statistically, not exactly.
In our rerun with this code (100 runs per algorithm, Welch t-test against the paper's arrays), 0 of 118
algorithm/attack cells differed at p < 0.01. The scripts now take `--seed`, so a given rerun is repeatable.

## Notes

- **Table 2 bug (fixed here).** Under the Gaussian attack, `epsilon_t` is the perturbation std. The paper's Table 2 run
  passed `epsilon_t**2 = 0.09` to RS-2 and RO instead of 0.3, so their rows faced a weaker attack than RS-1 and RS-G.
  `sensitivity_exp.py` now uses 0.3 for all algorithms, so its numbers differ from the published table.
- Changes since the first release: UCB/LCB now use the posterior std every round (an earlier version used the variance
  after round 0); RS-1 no longer crashes when the whole UCB is above τ; β = 1 for all experiments; RS-G uses the
  plain Euclidean distance; optional Thompson sampling (`--thompson`).

## Citation

```bibtex
@inproceedings{saday2025adveRS,
  title     = {Robust Satisficing Gaussian Process Bandits Under Adversarial Attacks},
  author    = {Saday, Artun and Y{\i}ld{\i}r{\i}m, Ya{\c{s}}ar Cahit and Tekin, Cem},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```
