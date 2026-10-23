"""
run_all.py - run the full HAR experiment grid in parallel processes.

    python run_all.py [--workers 5] [--threads 4]

Every experiment is cached in results/*.json, so this script can be
interrupted and restarted; finished experiments are skipped.  The notebook
uses the same cache.  Small recurrent models are latency-bound on a CPU, so
several 4-thread processes are much faster than one 20-thread process.
"""
import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

ALL = ["knn", "svm", "lstm", "lstm_cnn", "bilstm", "inceptiontime", "pacsi"]

EXPERIMENTS = (
    # 1) reproduction of the papers: 7 activities, paper-style random split + honest split
    [(m, "activity7", p, 0) for p in ("random", "cross_session") for m in ALL]
    # 2) end goal: empty / static / moving, held-out sessions and held-out room
    + [(m, "presence_motion3", p, 0) for p in ("cross_session", "cross_room") for m in ALL]
    + [("lstm_cnn", "presence_motion3", "random", 0)]
    # 3) [Shang21] study: 2 vs 4 channels, LSTM vs LSTM-CNN, binary static/dynamic
    + [(m, "static_dynamic2", "cross_session", 0) for m in ("knn", "svm", "lstm", "lstm_cnn", "lstm_cnn_2ch")]
    + [("lstm_cnn_2ch", "activity7", "cross_session", 0)]
    # 4) [Quy25] ablations
    + [(m, "activity7", "cross_session", 0) for m in
       ("pacsi_amp_only", "pacsi_phase_only", "pacsi_no_gre", "pacsi_no_grn")]
    # 5) seed variance of the core model on the goal task
    + [("lstm_cnn", "presence_motion3", "cross_session", s) for s in (1, 2)]
)

HEAVY = {"pacsi", "bilstm", "inceptiontime"}


def _job(args):
    model, task, protocol, seed, threads = args
    import torch
    torch.set_num_threads(threads)
    from csi_experiments import run_har
    t0 = time.time()
    try:
        r = run_har(model, task, protocol, seed, verbose=False, threads=threads)
        return f"OK   {model:18s} {task:17s} {protocol:13s} s{seed}  acc {r['test']['accuracy']:.3f}  F1 {r['test']['f1_macro']:.3f}  ({time.time() - t0:.0f}s)"
    except Exception:
        return f"FAIL {model} {task} {protocol} s{seed}\n{traceback.format_exc()}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # heavy models first so they don't end up running alone at the end
    jobs = sorted(EXPERIMENTS, key=lambda e: e[0].split("_")[0] not in HEAVY)
    jobs = [j for j in jobs if not os.path.exists(f"results/har__{j[0]}__{j[1]}__{j[2]}__s{j[3]}.json")]
    print(f"{len(jobs)} experiments to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_job, j + (a.threads,)) for j in jobs]
        for f in as_completed(futs):
            print(f.result(), flush=True)
    sys.exit(0)
