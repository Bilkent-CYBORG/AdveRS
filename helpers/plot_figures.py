"""
Draw the paper figures from the arrays in output_arrays/ (written by the scripts or run_parallel.py).

    python helpers/plot_figures.py                 # all figures whose data exist
    python helpers/plot_figures.py --figs 2 6      # only Figs 2 and 6 (need no BO runs)
    python helpers/plot_figures.py --usetex        # LaTeX text, as in the paper

Regret curves show the mean over runs with error bars of std/2. Figures go to figures/.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import cumulative_trapezoid  # noqa: E402
from sklearn.metrics.pairwise import euclidean_distances  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import insulin_dosage  # noqa: E402
import inv_pend  # noqa: E402
import synthetic  # noqa: E402
from adv_RO import find_x_ro_action  # noqa: E402
from adv_RS1 import find_x_rs_action_RS1, find_x_rs_action_RS1_diabetes_contextual  # noqa: E402
from adv_RS2 import find_x_rs_action_RS2  # noqa: E402

ATTACKS = ["Random Attack", "LCB Attack", "Worst Case Attack"]


def styles(r_small, r_large, ro_symbol=r"\epsilon_t"):
    return [
        ("RS1", "r", "^", "--", r"RS-G, $p=1$ (RS-1)"),
        ("RS2", "m", "^", "--", r"RS-G, $p=\infty$ (RS-2)"),
        ("RSG", "orange", "^", "--", r"RS-G, $p=2$"),
        ("RO_true", "b", "d", "--", rf"RO $r={ro_symbol}$"),
        ("RO_small", "g", "d", "--", rf"RO $r={r_small} \cdot {ro_symbol}$"),
        ("RO_large", "darkred", "d", "--", rf"RO $r={r_large} \cdot {ro_symbol}$"),
        ("RS1T", "r", "^", ":", r"RS-G-TS, $p=1$ (RS-1)"),
        ("RS2T", "m", "^", ":", r"RS-G-TS, $p=\infty$ (RS-2)"),
        ("RSGT", "orange", "^", ":", r"RS-G-TS, $p=2$"),
    ]


def ucb_styles(*a, **k):
    """Styles without the Thompson sampling variants (only Fig 10 shows those)."""
    return [s for s in styles(*a, **k) if not s[0].endswith("T")]


def regret_curves(ax, files, Y, tau, shift, T, markevery, algs):
    """Plot mean cumulative regret max(tau - shift - f(x_t), 0) with std/2 error bars."""
    drawn = False
    for alg, c, mk, ls, lab in algs:
        f = files(alg)
        if not os.path.exists(f):
            continue
        d = np.load(f)
        # Per-run shifts (insulin) cover the first d.shape[0] runs, e.g. 20 Thompson runs in Fig 10
        sh = shift[: d.shape[0]] if np.ndim(shift) else shift
        reg = np.maximum(tau - sh - Y.reshape(-1)[d[:, 1, :].astype(int)], 0)
        cum = np.cumsum(reg, axis=1)
        m, s = cum.mean(0), cum.std(0)
        ax.plot(m, c=c, ls=ls, marker=mk, markevery=markevery, ms=7, label=lab)
        ax.errorbar(range(0, T, markevery), m[::markevery], yerr=s[::markevery] / 2,
                    fmt="none", capsize=5, capthick=2, c=c)
        drawn = True
    ax.set_xlabel("Round t")
    return drawn


def synthetic_grid(epsilon, rs_regret, path, title_eps, r_small=0.5, r_large=4):
    """Figs 3, 4, 8, 9: three attacks side by side."""
    kappa = 0.0
    if rs_regret:
        _, kappa = find_x_rs_action_RS1(X=synthetic.X, Y=synthetic.Z, tau=synthetic.tau,
                                        dist_metric=euclidean_distances)
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))
    drawn = False
    for a in range(3):
        files = lambda alg: os.path.join(path, synthetic.file_name(alg, a, NUM_ITER, epsilon))  # noqa: E731
        drawn |= regret_curves(axs[a], files, synthetic.Z, synthetic.tau, kappa * epsilon, synthetic.T, 16,
                               ucb_styles(r_small, r_large))
        axs[a].set_ylabel("RS Regret" if rs_regret else "Lenient Regret")
        axs[a].set_title(ATTACKS[a])
    axs[1].legend(title=rf"$\tau={synthetic.tau}$, $\epsilon_t={title_eps}$")
    return fig, drawn


def inv_pend_grid(rs_regret):
    """Fig 7: lenient (top) or RS (bottom) regret for the inverted pendulum."""
    kappa = 0.0
    if rs_regret:
        _, kappa = find_x_rs_action_RS1(X=inv_pend.X, Y=inv_pend.Y, tau=inv_pend.tau,
                                        dist_metric=euclidean_distances)
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))
    drawn = False
    for a in range(3):
        files = lambda alg: os.path.join("output_arrays/inv_pend", inv_pend.file_name(alg, a, NUM_ITER))  # noqa
        drawn |= regret_curves(axs[a], files, inv_pend.Y, inv_pend.tau, kappa * inv_pend.epsilon_t, inv_pend.T,
                               41, ucb_styles(0.3, 3))
        axs[a].set_ylabel("RS Regret" if rs_regret else "Lenient Regret")
        axs[a].set_title(ATTACKS[a])
    axs[0].legend(title=rf"$\tau=0$, $\epsilon_t={inv_pend.epsilon_t}$")
    return fig, drawn


def insulin_figure(with_ts):
    """Figs 5 / 10: lenient and RS regret for insulin dosage."""
    m = insulin_dosage
    path = "output_arrays/insulin"
    ctx = [os.path.join(path, f"diabetes_N_{NUM_ITER}_T_{m.T}_{k}.npy")
           for k in ("C_true_inds", "C_ref_inds", "fbg_true_inds")]
    if not all(os.path.exists(f) for f in ctx):
        return None, False
    C_true, C_ref, fbg = (np.load(f) for f in ctx)

    # RS regret term: kappa(tau, c_ref) * |c_true - c_ref| per run and round
    carb_scale = (m.carb_values.max() - m.carb_values.min()) / len(m.carb_values)
    shift = np.empty(C_ref.shape)
    for i in range(C_ref.shape[0]):
        for t in range(C_ref.shape[1]):
            _, k = find_x_rs_action_RS1_diabetes_contextual(
                C=m.carb_values, Y=m.f_objective[:, :, fbg[i, t]], ind_ref_cho=C_ref[i, t], tau=m.tau,
                dist_metric=euclidean_distances)
            shift[i, t] = k * abs(C_true[i, t] - C_ref[i, t]) * carb_scale
    shift = np.maximum(shift, 0)

    algs = styles("1/3", 3, r"\sigma_s") if with_ts else ucb_styles("1/3", 3, r"\sigma_s")
    algs = [(a, c, mk, ls, lab.replace(r"1/3 \cdot \sigma_s", r"\sigma_s / 3")) for a, c, mk, ls, lab in algs]
    files = lambda alg: os.path.join(path, m.file_name(alg, NUM_ITER))  # noqa: E731
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    drawn = regret_curves(axs[0], files, m.Y_mat, m.tau, 0.0, m.T, 80, algs)
    regret_curves(axs[1], files, m.Y_mat, m.tau, shift, m.T, 80, algs)
    axs[0].set_ylabel("Lenient Regret")
    axs[1].set_ylabel("RS Regret")
    for ax in axs:
        ax.legend(title=r"$\tau=-10$, $\epsilon_t \sim \mathcal{N}(0, \sigma_s^2)$, $\sigma_s=32$", fontsize=9)
    return fig, drawn


def synthetic_solutions():
    """Robust solutions of each formulation on the true synthetic function (Figs 2 and 6)."""
    X, Z, tau = synthetic.X, synthetic.Z, synthetic.tau
    sq = synthetic.distance_metric
    eps = 0.5
    i_rs1, _ = find_x_rs_action_RS1(X=X, Y=Z, tau=tau, dist_metric=euclidean_distances, pow=1)
    i_rs2, _ = find_x_rs_action_RS2(X=X, Y=Z, tau=tau, dist_metric=sq)
    i_rsg, _ = find_x_rs_action_RS1(X=X, Y=Z, tau=tau, dist_metric=euclidean_distances, pow=2)
    return [
        ("RS1", "r", "^", r"RS-G, $p=1$ (RS-1)", X[i_rs1]),
        ("RS2", "m", "^", r"RS-G, $p=\infty$ (RS-2)", X[i_rs2]),
        ("RSG", "orange", "^", r"RS-G, $p=2$", X[i_rsg]),
        ("RO_true", "b", "d", rf"RO $r={eps}$", X[find_x_ro_action(X=X, Y=Z, r_ro=eps**2, dist_metric=sq)]),
        ("RO_large", "darkred", "d", rf"RO $r={eps * 4}$",
         X[find_x_ro_action(X=X, Y=Z, r_ro=(eps * 4) ** 2, dist_metric=sq)]),
        ("RO_small", "g", "d", rf"RO $r={eps * 0.5}$",
         X[find_x_ro_action(X=X, Y=Z, r_ro=(eps * 0.5) ** 2, dist_metric=sq)]),
    ]


def fig2(sols):
    """Fig 2 (left): objective, tau contour and the robust solutions."""
    fig, ax = plt.subplots(figsize=(4.5, 5))
    cf = ax.contourf(synthetic.X_grid, synthetic.Y_grid, synthetic.Z_grid, levels=30, cmap="plasma")
    fig.colorbar(cf, ax=ax, location="top", label=r"$f(\mathbf{x})$")
    ax.contour(synthetic.X_grid, synthetic.Y_grid, synthetic.Z_grid, levels=[synthetic.tau],
               colors="black", linestyles="dashed")
    ax.plot([], [], "k--", label=rf"$\tau={synthetic.tau}$")
    for alg, c, mk, lab, x in sols:
        if alg != "RSG":  # the paper shows RS-1, RS-2 and RO
            ax.scatter(x[0], x[1], c=c, marker=mk, s=100, label=lab, zorder=3)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(fontsize=8, loc="upper left")
    return fig


def fig6(sols):
    """Fig 6: worst-case reward around each solution and the area above it, vs perturbation size."""
    X, Z, tau = synthetic.X, synthetic.Z, synthetic.tau
    epsilons = np.linspace(0, 1.4, 50)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
    for _, c, _, lab, x in sols:
        dists = euclidean_distances(X, x.reshape(1, -1)).ravel()
        worst = np.array([Z[dists <= e].min() for e in epsilons])
        area = cumulative_trapezoid(np.maximum(0, tau - worst), epsilons, initial=0)
        axs[0].plot(epsilons, worst, label=lab, color=c)
        axs[1].plot(epsilons, area, label=lab, color=c)
    axs[0].axhline(tau, linestyle="--", color="k")
    axs[0].set_xlabel(r"$\epsilon$ (from 0 to 1/3 of domain range)")
    axs[0].set_ylabel(r"$\min_{\delta \in \Delta_\epsilon(x^*)} f(x^*+\delta)$")
    axs[1].set_xlabel(r"$\epsilon$ (from 0 to 1/3 of domain range)")
    axs[1].set_ylabel(r"Area($\epsilon$)")
    axs[1].legend(fontsize=9)
    return fig


def save(fig, drawn, name):
    if fig is None or not drawn:
        print(f"skip {name} (no data)")
        if fig is not None:
            plt.close(fig)
        return
    fig.tight_layout()
    path = os.path.join("figures", name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    global NUM_ITER
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figs", nargs="+", default=["2", "3", "4", "5", "6", "7", "8", "9", "10"])
    p.add_argument("--num-iter", type=int, default=100, help="N in the result file names")
    p.add_argument("--usetex", action="store_true")
    a = p.parse_args()
    NUM_ITER = a.num_iter
    np.random.seed(0)  # RO breaks ties at random (r = 2 has three tied solutions in Figs 2 and 6)
    os.makedirs("figures", exist_ok=True)
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "axes.titlesize": 16, "legend.fontsize": 10,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "font.family": "serif",
                         "text.usetex": a.usetex})

    syn = "output_arrays/synthetic"
    if "2" in a.figs or "6" in a.figs:
        sols = synthetic_solutions()
        if "2" in a.figs:
            save(fig2(sols), True, "fig2_solutions.pdf")
        if "6" in a.figs:
            save(fig6(sols), True, "fig6_worst_case_auc.pdf")
    if "3" in a.figs:
        save(*synthetic_grid(0.5, False, f"{syn}/poly_eps_0.5", 0.5), "fig3a_regret_synthetic.pdf")
        save(*synthetic_grid(1.5, False, f"{syn}/poly_eps_1.5", 1.5), "fig3b_regret_synthetic_eps1.5.pdf")
    if "4" in a.figs:
        save(*synthetic_grid(0.5, True, f"{syn}/poly_eps_0.5", 0.5), "fig4_rs_regret_synthetic.pdf")
    if "8" in a.figs:
        save(*synthetic_grid(0.5, False, f"{syn}/rbf_eps_0.5", 0.5), "fig8_regret_misspecified.pdf")
    if "9" in a.figs:
        save(*synthetic_grid(0.5, True, f"{syn}/rbf_eps_0.5", 0.5), "fig9_rs_regret_misspecified.pdf")
    if "7" in a.figs:
        save(*inv_pend_grid(False), "fig7_regret_inv_pend.pdf")
        save(*inv_pend_grid(True), "fig7_rs_regret_inv_pend.pdf")
    if "5" in a.figs:
        save(*insulin_figure(False), "fig5_regret_insulin.pdf")
    if "10" in a.figs:
        f10 = os.path.join("output_arrays/insulin", insulin_dosage.file_name("RSGT", NUM_ITER))
        if os.path.exists(f10):
            save(*insulin_figure(True), "fig10_regret_insulin_ts.pdf")
        else:
            print("skip fig10 (run with --thompson)")


if __name__ == "__main__":
    main()
