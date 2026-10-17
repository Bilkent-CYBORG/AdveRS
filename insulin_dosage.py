"""
Insulin dosage experiment (paper Figs 5, 10).

    python insulin_dosage.py
    python insulin_dosage.py --thompson     # also run the Thompson sampling variants (Fig 10)

The carbohydrate context shifts are drawn from --context-seed and saved next to the results,
so every algorithm sees the same contexts. Results go to output_arrays/insulin/.
"""

import argparse
import os
from datetime import datetime
from functools import partial
from itertools import product

import GPy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from BO_loop import BO_loop_RS_diabetes
from BO_loop import BO_loop_RO_diabetes
from adv_RS1 import find_x_rs_action_RS1_diabetes_contextual
from adv_RS2 import find_x_rs_action_RS2_diabetes_contextual

ALGS = ["RS1", "RS2", "RSG", "RO_true", "RO_small", "RO_large"]
TS_ALGS = ["RS1T", "RS2T", "RSGT"]


def ins_index_to_flat_index(ins_ind, cho_ind, fbg_ind, len_cho, len_fbg):
    """
    Convert the 3D index to a flat index.
    """
    return (ins_ind * len_cho * len_fbg) + (cho_ind * len_fbg) + fbg_ind


insulin_values = np.linspace(0, 15, 16).reshape(-1, 1)
carb_values = np.linspace(20, 100, 20).reshape(-1, 1)
fbg_values = np.linspace(90, 130, 5).reshape(-1, 1)

# Load the cost matrix
cost_matrix_dir = "util_arrays"
cost_matrix = np.load(f"{cost_matrix_dir}/pbg_obj_func_110.npy")

cost_matrix = cost_matrix[:, :, 1:2]
fbg_values = fbg_values[1:2]

# Standardize the f_objective using mean and std.
f_objective = cost_matrix
obj_mean = f_objective.mean()
obj_std = f_objective.std()
f_objective = (f_objective - obj_mean) / obj_std

X = np.array(list(product(insulin_values, carb_values, fbg_values)))
X_flat = X.reshape(-1, 3)
Y_mat = f_objective

T = 500
tau = (-10.0 - obj_mean) / obj_std
shift_std = 8
shift_std_in_carb = ((carb_values.max() - carb_values.min()) / len(carb_values)) * shift_std
epsilon_t = shift_std_in_carb * 1.0
r_ro_true = shift_std_in_carb * 1.0
r_ro_large = shift_std_in_carb * 3
r_ro_small = shift_std_in_carb * (1.0 / 3)
sigma_noise = 1
num_initial_points = 5
rsg_p = 2
beta = 1

distance_metric = partial(euclidean_distances, squared=True)


def make_kernel():
    return GPy.kern.RBF(input_dim=3, variance=1.0, lengthscale=[3.0, 6.0, 1.0], ARD=True)


def file_name(alg, num_iter):
    """Output file name for one algorithm."""
    post = f"epsilon_{epsilon_t}_sigma_{sigma_noise:.0e}_T_{T}"
    r = {"RO_true": r_ro_true, "RO_small": r_ro_small, "RO_large": r_ro_large}
    if alg in r:
        return f"diabetes_N_{num_iter}_r_{r[alg]:.2f}_{post}_RO.npy"
    suffix = {"RS1": "RS1", "RS2": "RS2", "RSG": f"RSG_p{rsg_p}",
              "RS1T": "RS1T", "RS2T": "RS2T", "RSGT": f"RSG_p{rsg_p}T"}[alg]
    return f"diabetes_N_{num_iter}_tau_{tau:.3f}_{post}_{suffix}.npy"


def draw_contexts(num_iter, seed):
    """True carb index, announced (reference) carb index and fbg index for every run and round."""
    rng = np.random.RandomState(seed)
    C_true_inds = rng.choice(len(carb_values), size=(num_iter, T), replace=True)
    C_ref_shifts = np.round(rng.normal(0, shift_std, size=(num_iter, T))).astype(int)
    C_ref_inds = np.clip(C_true_inds + C_ref_shifts, a_min=0, a_max=len(carb_values) - 1)
    fbg_true_inds = rng.choice(len(fbg_values), size=(num_iter, T), replace=True)
    return C_true_inds, C_ref_inds, fbg_true_inds


def run(alg, num_iter, C_true_inds, C_ref_inds, fbg_true_inds):
    common = dict(
        X=X_flat, Y=Y_mat, C_ref_inds=C_ref_inds, C_true_inds=C_true_inds, fbg_true_inds=fbg_true_inds,
        kernel=make_kernel(), sigma_noise=sigma_noise, num_init_samples=num_initial_points,
        T=T, print_samples=False, num_iter=num_iter, beta=beta,
    )
    thompson = alg.endswith("T")
    base = alg[:-1] if thompson else alg

    if base == "RS1":
        return BO_loop_RS_diabetes(acquisition_function=find_x_rs_action_RS1_diabetes_contextual,
                                   distance_metric=euclidean_distances, tau=tau, thompson=thompson, **common)
    if base == "RSG":
        find_x_rsg_action_diabetes_contextual = partial(find_x_rs_action_RS1_diabetes_contextual, pow=rsg_p)
        return BO_loop_RS_diabetes(acquisition_function=find_x_rsg_action_diabetes_contextual,
                                   distance_metric=euclidean_distances, tau=tau, thompson=thompson, **common)
    if base == "RS2":
        return BO_loop_RS_diabetes(acquisition_function=find_x_rs_action_RS2_diabetes_contextual,
                                   distance_metric=distance_metric, tau=tau, thompson=thompson, **common)
    r_ro = {"RO_true": r_ro_true, "RO_small": r_ro_small, "RO_large": r_ro_large}[alg]
    return BO_loop_RO_diabetes(distance_metric=distance_metric, r_ro=r_ro**2, **common)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algs", nargs="+", default=ALGS, choices=ALGS + TS_ALGS)
    p.add_argument("--thompson", action="store_true", help="also run the Thompson sampling variants")
    p.add_argument("--num-iter", type=int, default=100, help="number of independent runs")
    p.add_argument("--seed", type=int, default=0, help="seed for the BO loops")
    p.add_argument("--context-seed", type=int, default=0, help="seed for the carbohydrate context draws")
    p.add_argument("--out", default="output_arrays/insulin")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    algs = args.algs + (TS_ALGS if args.thompson else [])
    os.makedirs(args.out, exist_ok=True)

    # Same context draws for every algorithm
    C_true_inds, C_ref_inds, fbg_true_inds = draw_contexts(args.num_iter, args.context_seed)
    np.save(os.path.join(args.out, f"diabetes_N_{args.num_iter}_T_{T}_C_true_inds.npy"), C_true_inds)
    np.save(os.path.join(args.out, f"diabetes_N_{args.num_iter}_T_{T}_C_ref_inds.npy"), C_ref_inds)
    np.save(os.path.join(args.out, f"diabetes_N_{args.num_iter}_T_{T}_fbg_true_inds.npy"), fbg_true_inds)

    np.random.seed(args.seed)
    print(f"Num_iter: {args.num_iter}")
    for alg in algs:
        print(f"Starting {alg} loop")
        start_time = datetime.now()
        history = run(alg, args.num_iter, C_true_inds, C_ref_inds, fbg_true_inds)
        print(f"{alg} run time: {datetime.now() - start_time}")
        np.save(os.path.join(args.out, file_name(alg, args.num_iter)), history)
