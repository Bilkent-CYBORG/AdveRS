"""
Synthetic experiment (paper Figs 3, 4, 8, 9).

    python synthetic.py                      # Fig 3a / 4   (eps = 0.5)
    python synthetic.py --epsilon 1.5        # Fig 3b
    python synthetic.py --kernel rbf         # Figs 8 / 9   (misspecified kernel)
    python synthetic.py --thompson           # also run the Thompson sampling variants

Results go to output_arrays/synthetic/<kernel>_eps_<epsilon>/.
"""

import argparse
import os
from functools import partial

import GPy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from BO_loop import BO_loop_RS
from BO_loop import BO_loop_RO
from adv_RS1 import find_x_rs_action_RS1
from adv_RS2 import find_x_rs_action_RS2

ALGS = ["RS1", "RS2", "RSG", "RO_true", "RO_small", "RO_large"]
TS_ALGS = ["RS1T", "RS2T", "RSGT"]


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

tau = -10
T = 100
beta = 1
rsg_p = 2
distance_metric = partial(euclidean_distances, squared=True)
sigma_noise = 1
num_initial_points = 10


def make_kernel(name):
    if name == "rbf":
        # Misspecified kernel (paper App. E.2)
        return GPy.kern.RBF(2, lengthscale=0.1, variance=10)
    return GPy.kern.Poly(2, order=6, scale=1.11, bias=2.84, variance=1.32)


def file_name(alg, attack_mode, num_iter, epsilon_t):
    """Output file name for one algorithm / attack."""
    r = {"RO_true": epsilon_t * 1, "RO_small": epsilon_t * 0.5, "RO_large": epsilon_t * 4}
    if alg in r:
        return f"f_poly_N_{num_iter}_r_{r[alg]:.2f}_epsilon_{epsilon_t}_T_{T}_attack_{attack_mode}_RO.npy"
    suffix = {"RS1": "RS1", "RS2": "RS2", "RSG": f"RSG_p{rsg_p}",
              "RS1T": "RS1T", "RS2T": "RS2T", "RSGT": f"RSG_p{rsg_p}T"}[alg]
    return f"f_poly_N_{num_iter}_tau_{tau}_epsilon_{epsilon_t}_T_{T}_attack_{attack_mode}_{suffix}.npy"


def run(alg, attack_mode, num_iter, epsilon_t, kernel):
    r_ro_true = epsilon_t * 1
    r_ro_large = epsilon_t * 4
    r_ro_small = epsilon_t * 0.5
    common = dict(
        X=X, Y=Z, kernel=kernel, sigma_noise=sigma_noise, num_init_samples=num_initial_points,
        attack_mode=attack_mode, T=T, print_samples=False, num_iter=num_iter, beta=beta,
    )
    thompson = alg.endswith("T")
    base = alg[:-1] if thompson else alg

    # RS-1 and RS-G use Euclidean distances; RS-2 and RO use squared distances (so eps and r are squared)
    if base == "RS1":
        return BO_loop_RS(acquisition_function=find_x_rs_action_RS1, distance_metric=euclidean_distances,
                          tau=tau, epsilon_t=epsilon_t, thompson=thompson, **common)
    if base == "RSG":
        partial_acq_rsg = partial(find_x_rs_action_RS1, pow=rsg_p)
        return BO_loop_RS(acquisition_function=partial_acq_rsg, distance_metric=euclidean_distances,
                          tau=tau, epsilon_t=epsilon_t, thompson=thompson, **common)
    if base == "RS2":
        return BO_loop_RS(acquisition_function=find_x_rs_action_RS2, distance_metric=distance_metric,
                          tau=tau, epsilon_t=epsilon_t**2, thompson=thompson, **common)
    r_ro = {"RO_true": r_ro_true, "RO_small": r_ro_small, "RO_large": r_ro_large}[alg]
    return BO_loop_RO(distance_metric=distance_metric, r_ro=r_ro**2, epsilon_t=epsilon_t**2, **common)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epsilon", type=float, default=0.5, help="perturbation budget (0.5 or 1.5 in the paper)")
    p.add_argument("--kernel", choices=["poly", "rbf"], default="poly", help="rbf = misspecified kernel")
    p.add_argument("--algs", nargs="+", default=ALGS, choices=ALGS + TS_ALGS)
    p.add_argument("--thompson", action="store_true", help="also run the Thompson sampling variants")
    p.add_argument("--attacks", nargs="+", type=int, default=[0, 1, 2], help="0 random, 1 LCB, 2 worst case")
    p.add_argument("--num-iter", type=int, default=100, help="number of independent runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output folder (default: output_arrays/synthetic/...)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    algs = args.algs + (TS_ALGS if args.thompson else [])
    path = args.out or f"output_arrays/synthetic/{args.kernel}_eps_{args.epsilon}"
    os.makedirs(path, exist_ok=True)

    for attack_mode in args.attacks:
        print(f"Attack mode: {attack_mode}")
        print(f"Num_iter: {args.num_iter}")
        for alg in algs:
            print(f"Starting {alg} loop")
            history = run(alg, attack_mode, args.num_iter, args.epsilon, make_kernel(args.kernel))
            np.save(os.path.join(path, file_name(alg, attack_mode, args.num_iter, args.epsilon)), history)
