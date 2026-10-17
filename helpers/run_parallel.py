"""
Run the experiment scripts in parallel and merge the results.

Each (experiment, algorithm, attack) is split into chunks of runs, and each chunk runs the
original script in its own process with its own seed. Finished chunks are skipped on restart,
so it is safe to run inside tmux/screen and resume after an interruption.

    python helpers/run_parallel.py --workers 28                         # everything in the paper
    python helpers/run_parallel.py --exps synthetic insulin --workers 8
    python helpers/run_parallel.py --merge-only

Merged arrays have the same names and shapes as a single-process run of the script.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGS = ["RS1", "RS2", "RSG", "RO_true", "RO_small", "RO_large"]
TS_ALGS = ["RS1T", "RS2T", "RSGT"]

# script, extra args, output folder, attacks (None = no attack argument)
EXPERIMENTS = {
    "synthetic": ("synthetic.py", ["--epsilon", "0.5"], "output_arrays/synthetic/poly_eps_0.5", [0, 1, 2]),
    "synthetic_eps1.5": ("synthetic.py", ["--epsilon", "1.5"], "output_arrays/synthetic/poly_eps_1.5", [0, 1, 2]),
    "synthetic_miss": ("synthetic.py", ["--kernel", "rbf"], "output_arrays/synthetic/rbf_eps_0.5", [0, 1, 2]),
    "inv_pend": ("inv_pend.py", [], "output_arrays/inv_pend", [0, 1, 2]),
    "insulin": ("insulin_dosage.py", [], "output_arrays/insulin", None),
}
TABLE2 = "sensitivity"


def build_tasks(args):
    tasks = []
    for exp in args.exps:
        if exp == TABLE2:
            for tau in np.linspace(-30.0, 10.0, 5):
                tasks.append(dict(exp=exp, tau=float(tau)))
            continue
        script, extra, out, attacks = EXPERIMENTS[exp]
        # Thompson sampling is only in the paper for insulin (Fig 10); on the 4096-point pendulum grid
        # posterior sampling makes a single run take over 10 minutes
        algs = ALGS + (TS_ALGS if args.thompson and exp == "insulin" else [])
        for alg in algs:
            for atk in attacks or [None]:
                for start in range(0, args.num_iter, args.chunk):
                    stop = min(start + args.chunk, args.num_iter)
                    tasks.append(dict(exp=exp, alg=alg, attack=atk, start=start, stop=stop))
    return tasks


def task_tag(t):
    if t["exp"] == TABLE2:
        return f"{TABLE2}_tau_{t['tau']:.1f}"
    atk = "" if t["attack"] is None else f"_atk{t['attack']}"
    return f"{t['exp']}_{t['alg']}{atk}_runs_{t['start']:03d}_{t['stop']:03d}"


def chunk_dir(t):
    return os.path.join(EXPERIMENTS[t["exp"]][2], "_chunks", f"runs_{t['start']:03d}_{t['stop']:03d}")


def done_marker(t):
    if t["exp"] == TABLE2:
        return os.path.join("output_arrays", "sensitivity_exp", "_done", task_tag(t))
    return os.path.join(chunk_dir(t), "_done", task_tag(t))


def command(t, args):
    if t["exp"] == TABLE2:
        seed = args.seed + zlib.crc32(task_tag(t).encode()) % 100000
        return [sys.executable, "sensitivity_exp.py", "--taus", str(t["tau"]), "--seed", str(seed)]
    script, extra, _, _ = EXPERIMENTS[t["exp"]]
    # Seed depends on the task only, so reruns give the same results
    seed = args.seed + zlib.crc32(task_tag(t).encode()) % 100000
    cmd = [sys.executable, script, *extra, "--algs", t["alg"], "--num-iter", str(t["stop"] - t["start"]),
           "--seed", str(seed), "--out", chunk_dir(t)]
    if t["attack"] is not None:
        cmd += ["--attacks", str(t["attack"])]
    if t["exp"] == "insulin":
        # Same context draws for every algorithm within a chunk
        cmd += ["--context-seed", str(args.seed + t["start"])]
    return cmd


def run(t, args, log_dir):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", PYTHONWARNINGS="ignore")
    t0 = time.time()
    with open(os.path.join(log_dir, task_tag(t) + ".log"), "w") as lf:
        rc = subprocess.call(command(t, args), cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if rc == 0:
        os.makedirs(os.path.dirname(done_marker(t)), exist_ok=True)
        open(done_marker(t), "w").close()
    return task_tag(t), rc, time.time() - t0


def merge(exps, num_iter):
    """Concatenate chunk results along the run axis and save them under the full-run name."""
    for exp in exps:
        if exp == TABLE2:
            continue
        out = EXPERIMENTS[exp][2]
        chunks = sorted(glob.glob(os.path.join(out, "_chunks", "runs_*")))
        groups = {}
        for c in chunks:
            start = int(re.search(r"runs_(\d+)_", c).group(1))
            for f in glob.glob(os.path.join(c, "*.npy")):
                name = re.sub(r"_N_\d+_", f"_N_{num_iter}_", os.path.basename(f))
                groups.setdefault(name, []).append((start, f))
        for name, parts in sorted(groups.items()):
            parts.sort()
            arr = np.concatenate([np.load(f) for _, f in parts], axis=0)
            if arr.shape[0] != num_iter:
                print(f"skip (incomplete, {arr.shape[0]}/{num_iter} runs): {name}")
                continue
            np.save(os.path.join(out, name), arr)
            print(f"merged {arr.shape} -> {os.path.join(out, name)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exps", nargs="+", default=list(EXPERIMENTS) + [TABLE2],
                   choices=list(EXPERIMENTS) + [TABLE2])
    p.add_argument("--thompson", action="store_true", help="also run Thompson sampling for insulin (Fig 10)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--num-iter", type=int, default=100, help="runs per algorithm (Table 2 always uses 20)")
    p.add_argument("--chunk", type=int, default=10, help="runs per process")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)

    if not args.merge_only:
        log_dir = os.path.join("output_arrays", "_logs")
        os.makedirs(log_dir, exist_ok=True)
        tasks = [t for t in build_tasks(args) if not os.path.exists(done_marker(t))]
        # Slowest experiments first, so the tail is short
        order = {"inv_pend": 0, TABLE2: 1, "insulin": 2}
        tasks.sort(key=lambda t: order.get(t["exp"], 3))
        print(f"{len(tasks)} tasks, {args.workers} workers", flush=True)
        t_start = time.time()
        failed = 0
        with ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(run, t, args, log_dir) for t in tasks]
            for i, fu in enumerate(as_completed(futs), 1):
                tag, rc, dt = fu.result()
                failed += rc != 0
                status = "ok" if rc == 0 else f"FAILED rc={rc} (see {log_dir}/{tag}.log)"
                print(f"[{i}/{len(tasks)}] {tag} {status} {dt / 60:.1f} min "
                      f"(elapsed {(time.time() - t_start) / 3600:.2f} h)", flush=True)
        if failed:
            print(f"{failed} tasks failed; rerun the same command to retry them.")

    merge(args.exps, args.num_iter)
    if TABLE2 in args.exps:
        subprocess.call([sys.executable, "sensitivity_exp.py", "--table-only"], cwd=ROOT)


if __name__ == "__main__":
    main()
