"""Run Member 2 experiments; output directories must match their saved configuration."""
import argparse
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.ml.experiment import run
from pathsnap.ml.models import BASELINES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--phase", choices=["all", "compare", "tune", "final"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tune-model", choices=list(BASELINES), help="Promising family to tune; default: CV winner")
    args = parser.parse_args()
    if args.trials < 6 or args.folds < 2 or args.jobs < 1:
        parser.error("Use >=6 trials (to include Bayesian-guided trials), >=2 folds, >=1 jobs")
    run(args.archive, args.work, args.phase, args.seed, args.folds, args.trials, args.jobs, args.tune_model)
