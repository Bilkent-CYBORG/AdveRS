"""
Effect of tau and r (paper Table 2): synthetic function under Gaussian perturbations N(0, 0.3^2), 20 runs.

    python sensitivity_exp.py                  # all tau values, then print the table
    python sensitivity_exp.py --taus -30 -20   # a subset (e.g. one process per tau)
    python sensitivity_exp.py --table-only     # print the table from saved summaries

Note: the paper's Table 2 was produced before a bug fix. RS-2 and RO then received
epsilon_t**2 = 0.09 as the Gaussian std instead of 0.3, i.e. a weaker attack than RS-1/RS-G.
"""

import argparse
import os
from functools import partial

import GPy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from BO_loop import BO_loop_RS
from BO_loop import BO_loop_RO
from adv_RO import find_x_ro_action
from adv_RS1 import find_x_rs_action_RS1
from adv_RS2 import find_x_rs_action_RS2

TAU_VALUES = np.linspace(-30.0, 10.0, 5)  # tau values from -30 to 10
R_VALUES = np.linspace(0.1, 2.0, 5)  # r values from 0.1 to 2.0
NUM_ITER = 20
BETA = 1
OUTPUT_PATH = f"output_arrays/sensitivity_exp/iter_{NUM_ITER}_beta_{BETA}"


def f_poly(x, y):
    """
    Computes the value of the polynomial function f_poly(x, y) similar to the one in Bogunovic et. al.

    Domain:
    The function is defined on a uniformly spaced grid of points in the range:
    x ∈ (-0.95, 3.2)
    y ∈ (-0.45, 4.4)

    Parameters:
    x (float or np.array): Array of input coordinates
    y (float or np.array): Array of input coordinates

    Returns:
    float or np.array: The computed value of the polynomial at the given (x, y) coordinates.
    """
    return (
        -2 * x**6
        + 12.2 * x**5
        - 21.2 * x**4
        - 6.2 * x
        + 6.4 * x**3
        + 4.7 * x**2
        - y**6
        + 11 * y**5
        - 43.3 * y**4
        + 10 * y
        + 74.8 * y**3
        - 56.9 * y**2
        + 4.1 * x * y
        + 0.05 * y**2 * x**2
        - 0.4 * y**2 * x
        - 0.4 * x**2 * y
    )


def run_sensitivity_experiment(tau_values=None, num_iter=NUM_ITER):
    """
    Run sensitivity experiment to analyze the effect of tau and r parameters
    on lenient regret for RS and RO algorithms.
    """
    # Define the range and resolution
    x_min, x_max = -0.95, 3.2
    y_min, y_max = -0.45, 4.4
    resolution = 32  # Number of points along each axis

    # Create arrays of x and y values
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)

    # Create a meshgrid from x and y arrays
    X_grid, Y_grid = np.meshgrid(x, y)
    X = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    Z_grid = f_poly(X_grid, Y_grid)
    Z = f_poly(X[:, 0], X[:, 1])
    Z = Z.reshape(-1, 1)

    # Kernel and experiment parameters
    kern_poly = GPy.kern.Poly(2, order=6, scale=1.11, bias=2.84, variance=1.32)

    # Experiment parameters
    T = 100
    distance_metric = partial(euclidean_distances, squared=True)
    sigma_noise = 1
    num_initial_points = 10
    rsg_p = 2
    beta = BETA

    # Parameters for sensitivity analysis
    if tau_values is None:
        tau_values = TAU_VALUES
    r_values = R_VALUES

    # Fixed epsilon for this experiment (perturbations from N(0, 0.3))
    epsilon_t = 0.3

    # Attack mode 3: Gaussian perturbation with std epsilon_t
    attack_mode = 3

    partial_acq_rsg = partial(find_x_rs_action_RS1, pow=rsg_p)

    # Results storage
    results = {}

    # Create output directory for saving results
    output_path = OUTPUT_PATH
    os.makedirs(output_path, exist_ok=True)

    print("Starting Sensitivity Experiment")
    print(f"Tau values: {tau_values}")
    print(f"R values: {r_values}")
    print(f"Epsilon: {epsilon_t}")
    print(f"Attack mode: {attack_mode}")
    print(f"Number of iterations: {num_iter}")
    print(f"Results will be saved to: {output_path}")

    # Run experiments for different tau values
    for tau in tau_values:
        print(f"\n=== Running experiments for tau = {tau} ===")

        # RSG Algorithm
        print(f"Running RSG for tau={tau}")
        history_RSG = BO_loop_RS(
            X=X,
            Y=Z,
            acquisition_function=partial_acq_rsg,
            kernel=kern_poly,
            sigma_noise=sigma_noise,
            num_init_samples=num_initial_points,
            distance_metric=euclidean_distances,
            attack_mode=attack_mode,
            tau=tau,
            epsilon_t=epsilon_t,
            T=T,
            print_samples=False,
            num_iter=num_iter,
            beta=beta,
        )

        selected_indices = history_RSG[:, 1, :].astype(int)
        lenient_regret = np.maximum(tau - Z[selected_indices], 0)
        avg_lenient_regret_RSG = np.mean(np.sum(lenient_regret, axis=1))

        print(f"RSG average lenient regret: {avg_lenient_regret_RSG:.2f}")

        # Save RSG results immediately
        rsg_filename = (
            f"RSG_tau_{tau:.1f}_epsilon_{epsilon_t}_attack_{attack_mode}_N_{num_iter}.npy"
        )
        np.save(os.path.join(output_path, rsg_filename), history_RSG)

        # RS1 Algorithm
        print(f"Running RS1 for tau={tau}")
        history_RS1 = BO_loop_RS(
            X=X,
            Y=Z,
            acquisition_function=find_x_rs_action_RS1,
            kernel=kern_poly,
            sigma_noise=sigma_noise,
            num_init_samples=num_initial_points,
            distance_metric=euclidean_distances,
            attack_mode=attack_mode,
            tau=tau,
            epsilon_t=epsilon_t,
            T=T,
            print_samples=False,
            num_iter=num_iter,
            beta=beta,
        )

        selected_indices = history_RS1[:, 1, :].astype(int)
        lenient_regret = np.maximum(tau - Z[selected_indices], 0)
        avg_lenient_regret_RS1 = np.mean(np.sum(lenient_regret, axis=1))

        print(f"RS1 average lenient regret: {avg_lenient_regret_RS1:.2f}")

        # Save RS1 results immediately
        rs1_filename = (
            f"RS1_tau_{tau:.1f}_epsilon_{epsilon_t}_attack_{attack_mode}_N_{num_iter}.npy"
        )
        np.save(os.path.join(output_path, rs1_filename), history_RS1)

        # RS2 Algorithm
        print(f"Running RS2 for tau={tau}")
        history_RS2 = BO_loop_RS(
            X=X,
            Y=Z,
            acquisition_function=find_x_rs_action_RS2,
            kernel=kern_poly,
            sigma_noise=sigma_noise,
            num_init_samples=num_initial_points,
            distance_metric=distance_metric,
            attack_mode=attack_mode,
            tau=tau,
            # Attack mode 3 uses epsilon_t as the Gaussian std (no distance metric), so it is not squared
            epsilon_t=epsilon_t,
            T=T,
            print_samples=False,
            num_iter=num_iter,
            beta=beta,
        )

        # Calculate lenient regret for RS2
        selected_indices_RS2 = history_RS2[:, 1, :].astype(int)
        lenient_regret_RS2 = np.maximum(tau - Z[selected_indices_RS2], 0)
        avg_lenient_regret_RS2 = np.mean(np.sum(lenient_regret_RS2, axis=1))

        print(f"RS2 average lenient regret: {avg_lenient_regret_RS2:.2f}")

        # Save RS2 results immediately
        rs2_filename = (
            f"RS2_tau_{tau:.1f}_epsilon_{epsilon_t}_attack_{attack_mode}_N_{num_iter}.npy"
        )
        np.save(os.path.join(output_path, rs2_filename), history_RS2)

        # Store RS results
        if tau not in results:
            results[tau] = {}
        results[tau]["RSG"] = avg_lenient_regret_RSG
        results[tau]["RS1"] = avg_lenient_regret_RS1
        results[tau]["RS2"] = avg_lenient_regret_RS2

        # RO Algorithms with different r values
        for r in r_values:
            print(f"Running RO with r={r} for tau={tau}")

            history_RO = BO_loop_RO(
                X=X,
                Y=Z,
                num_init_samples=num_initial_points,
                kernel=kern_poly,
                distance_metric=distance_metric,
                sigma_noise=sigma_noise,
                attack_mode=attack_mode,
                r_ro=r**2,
                epsilon_t=epsilon_t,  # Gaussian std, not squared (see RS2)
                T=T,
                print_samples=False,
                num_iter=num_iter,
                beta=beta,
            )

            # Calculate lenient regret for RO
            selected_indices_RO = history_RO[:, 1, :].astype(int)
            lenient_regret_RO = np.maximum(tau - Z[selected_indices_RO], 0)
            avg_lenient_regret_RO = np.mean(np.sum(lenient_regret_RO, axis=1))

            print(f"RO (r={r}) average lenient regret: {avg_lenient_regret_RO:.2f}")

            # Save RO results immediately
            ro_filename = f"RO_r_{r:.2f}_tau_{tau:.1f}_epsilon_{epsilon_t}_attack_{attack_mode}_N_{num_iter}.npy"
            np.save(os.path.join(output_path, ro_filename), history_RO)

            # Store RO results
            results[tau][f"RO_r_{r}"] = avg_lenient_regret_RO

        # Save summary results after completing all algorithms for this tau
        summary_filename = f"summary_tau_{tau:.1f}_results.npy"
        tau_results = {
            "tau": tau,
            "RSG_regret": avg_lenient_regret_RSG,
            "RS1_regret": avg_lenient_regret_RS1,
            "RS2_regret": avg_lenient_regret_RS2,
            "RO_regrets": {f"r_{r:.2f}": results[tau][f"RO_r_{r}"] for r in r_values},
        }
        np.save(os.path.join(output_path, summary_filename), tau_results)
        print(f"Saved summary results for tau={tau:.1f}")

    return results, tau_values, r_values


def print_latex_table(results, tau_values, r_values):
    """
    Print the results in LaTeX table format matching the paper.
    """
    print("\n" + "=" * 80)
    print("LATEX TABLE RESULTS")
    print("=" * 80)

    # Print table header
    print("\\begin{table}[!h]")
    print("\\centering")
    print("\\begin{tabular}{@{}l c c c c c@{}}")
    print("    \\toprule")

    # Header row
    header = "    {Alg. \\textbackslash ~ $\\tau$}"
    for tau in tau_values:
        header += f" & {{${tau:.2f}$}}"
    header += " \\\\"
    print(header)
    print("    \\midrule")

    # RS1 row
    rs1_row = "    RS1"
    for tau in tau_values:
        rs1_row += f" & {results[tau]['RS1']:.2f}"
    rs1_row += " \\\\"
    print(rs1_row)

    # RS2 row (with bold formatting for best values)
    rs2_row = "    RS2"
    for tau in tau_values:
        # Check if RS2 is the best for this tau
        all_values = [results[tau]["RS1"], results[tau]["RS2"], results[tau]["RSG"]]
        all_values.extend([results[tau][f"RO_r_{r}"] for r in r_values])
        is_best = results[tau]["RS2"] == min(all_values)

        if is_best:
            rs2_row += f" & \\textbf{{{results[tau]['RS2']:.2f}}}"
        else:
            rs2_row += f" & {results[tau]['RS2']:.2f}"
    rs2_row += " \\\\"
    print(rs2_row)

    # RSG row
    rsg_row = "    RSG"
    for tau in tau_values:
        rsg_row += f" & {results[tau]['RSG']:.2f}"
    rsg_row += " \\\\"
    print(rsg_row)

    # RO rows
    for r in r_values:
        ro_row = f"    RO, r={r:.2f}"
        for tau in tau_values:
            ro_row += f" & {results[tau][f'RO_r_{r}']:.2f}"
        ro_row += " \\\\"
        print(ro_row)

    # Table footer
    print("    \\bottomrule")
    print("\\end{tabular}")
    print(
        "\\caption{Average lenient regrets for synthetic experiment for RS-1, RS-2 and RO algorithms with varying $\\tau$ and $r$ values.}"
    )
    print("\\label{tab:tau_vs_r}")
    print("\\end{table}")


def print_summary_table(results, tau_values, r_values):
    """
    Print a human-readable summary table.
    """
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Print header
    header = f"{'Algorithm':<15}"
    for tau in tau_values:
        header += f"{tau:>10.2f}"
    print(header)
    print("-" * len(header))

    # Print RS1
    rs1_row = f"{'RS1':<15}"
    for tau in tau_values:
        rs1_row += f"{results[tau]['RS1']:>10.2f}"
    print(rs1_row)

    # Print RS2
    rs2_row = f"{'RS2':<15}"
    for tau in tau_values:
        rs2_row += f"{results[tau]['RS2']:>10.2f}"
    print(rs2_row)

    # Print RSG
    rsg_row = f"{'RSG':<15}"
    for tau in tau_values:
        rsg_row += f"{results[tau]['RSG']:>10.2f}"
    print(rsg_row)

    # Print RO algorithms
    for r in r_values:
        ro_row = f"{'RO, r=' + str(r):<15}"
        for tau in tau_values:
            ro_row += f"{results[tau][f'RO_r_{r}']:>10.2f}"
        print(ro_row)

    print("-" * len(header))

    # Find best algorithm for each tau
    print("\nBest algorithm for each tau:")
    for tau in tau_values:
        all_algs = ["RS1", "RS2", "RSG"] + [f"RO_r_{r}" for r in r_values]
        best_alg = min(all_algs, key=lambda alg: results[tau][alg])
        best_value = results[tau][best_alg]
        print(f"tau = {tau:6.2f}: {best_alg:<12} (regret = {best_value:.2f})")


def load_results(tau_values=TAU_VALUES, r_values=R_VALUES, output_path=OUTPUT_PATH):
    """Rebuild the results dict from the per-tau summary files."""
    results = {}
    for tau in tau_values:
        d = np.load(os.path.join(output_path, f"summary_tau_{tau:.1f}_results.npy"), allow_pickle=True).item()
        results[tau] = {"RSG": d["RSG_regret"], "RS1": d["RS1_regret"], "RS2": d["RS2_regret"]}
        for r in r_values:
            results[tau][f"RO_r_{r}"] = d["RO_regrets"][f"r_{r:.2f}"]
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--taus", nargs="+", type=float, default=None, help="subset of tau values")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--table-only", action="store_true", help="only print the table from saved summaries")
    args = p.parse_args()

    if not args.table_only:
        np.random.seed(args.seed)
        taus = TAU_VALUES if args.taus is None else np.array(args.taus)
        run_sensitivity_experiment(tau_values=taus)
    if args.table_only or args.taus is None:
        results = load_results()
        print_summary_table(results, TAU_VALUES, R_VALUES)
        print_latex_table(results, TAU_VALUES, R_VALUES)
