"""
Inverted pendulum experiment (paper Fig 7).

    python inv_pend.py
    python inv_pend.py --thompson     # also run the Thompson sampling variants

The controller cost on a 4D parameter grid is precomputed in util_arrays/inverted_pend_4d_res8.npy.
Results go to output_arrays/inv_pend/.
"""

import argparse
import os
from functools import partial
from itertools import product

import GPy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from adv_RS1 import find_x_rs_action_RS1
from adv_RS2 import find_x_rs_action_RS2
from BO_loop import BO_loop_RS
from BO_loop import BO_loop_RO

ALGS = ["RS1", "RS2", "RSG", "RO_true", "RO_small", "RO_large"]
TS_ALGS = ["RS1T", "RS2T", "RSGT"]

# Load the cost matrix
dir = "util_arrays"
cost_matrix = np.load(f"{dir}/inverted_pend_4d_res8.npy")

# Standardize the cost matrix
f_objective = -cost_matrix
f_objective = (f_objective - f_objective.mean()) / f_objective.std()
f_objective = np.clip(f_objective, -2, 2)

# Define the range and resolution
theta_1_range = [-3, 2]
theta_u_range = [1, 5]
resolution = 8

# Create a 4d meshgrid from theta arrays
theta_1 = np.linspace(theta_1_range[0], theta_1_range[1], resolution)
theta_u = np.linspace(theta_u_range[0], theta_u_range[1], resolution)
product_list = [theta_1, theta_1, theta_1, theta_u]
X = np.array(list(product(*product_list)))
Y = f_objective.flatten()
Y = Y.reshape(-1, 1)

T = 250
tau = 0.0
epsilon_t = 2.5
r_ro_true = epsilon_t * 1
r_ro_large = epsilon_t * 3
r_ro_small = epsilon_t * 0.3
beta = 1
rsg_p = 2

distance_metric = partial(euclidean_distances, squared=True)
sigma_noise = 5e-2
num_initial_points = 5


def make_kernel():
    return GPy.kern.RBF(input_dim=4, variance=0.4, lengthscale=[1.26, 1.9, 5.24, 1.86], ARD=True)


def file_name(alg, attack_mode, num_iter):
    """Output file name for one algorithm / attack."""
    post = f"epsilon_{epsilon_t}_sigma_{sigma_noise:.0e}_T_{T}_attack_{attack_mode}"
    r = {"RO_true": r_ro_true, "RO_small": r_ro_small, "RO_large": r_ro_large}
    if alg in r:
        return f"inv_pend4d_res_{resolution}_N_{num_iter}_r_{r[alg]:.2f}_{post}_RO.npy"
    suffix = {"RS1": "RS1", "RS2": "RS2", "RSG": f"RSG_p{rsg_p}",
              "RS1T": "RS1T", "RS2T": "RS2T", "RSGT": f"RSG_p{rsg_p}T"}[alg]
    return f"inv_pend4d_res_{resolution}_N_{num_iter}_tau_{tau:.3f}_{post}_{suffix}.npy"


def run(alg, attack_mode, num_iter):
    common = dict(
        X=X, Y=Y, kernel=make_kernel(), sigma_noise=sigma_noise, num_init_samples=num_initial_points,
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
    p.add_argument("--algs", nargs="+", default=ALGS, choices=ALGS + TS_ALGS)
    p.add_argument("--thompson", action="store_true", help="also run the Thompson sampling variants")
    p.add_argument("--attacks", nargs="+", type=int, default=[0, 1, 2], help="0 random, 1 LCB, 2 worst case")
    p.add_argument("--num-iter", type=int, default=100, help="number of independent runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="output_arrays/inv_pend")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    algs = args.algs + (TS_ALGS if args.thompson else [])
    os.makedirs(args.out, exist_ok=True)

    for attack_mode in args.attacks:
        print(f"Attack mode: {attack_mode}")
        print(f"Num_iter: {args.num_iter}")
        for alg in algs:
            print(f"Starting {alg} loop")
            history = run(alg, attack_mode, args.num_iter)
            np.save(os.path.join(args.out, file_name(alg, attack_mode, args.num_iter)), history)
