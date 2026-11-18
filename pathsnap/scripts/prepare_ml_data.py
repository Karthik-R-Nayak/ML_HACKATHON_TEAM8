"""Generate validated candidate/observation tables without changing Member 1 code."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.ml.data import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", choices=["train", "student_test"], default="train")
    args = parser.parse_args()
    print(prepare(args.archive, args.output, args.partition, 80 if args.partition == "train" else 10))
