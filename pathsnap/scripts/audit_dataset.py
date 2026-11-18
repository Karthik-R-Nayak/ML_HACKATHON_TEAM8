#!/usr/bin/env python
"""Audit EVERY trip in a partition folder. Read-only; writes CSV + a flag summary.

usage: python scripts/audit_dataset.py ROOT --out reports/audit_train [--node-base 0 --arc-base 0]
       [--nodes F --arcs F] [--limit N]
Run once for the training folder (80 trips) and once for student_test (10 trips).
"""
import argparse, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from pathsnap.audit import run_audit
from pathsnap.config import IOConfig, PipelineConfig

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("--out", required=True)
ap.add_argument("--node-base", type=int, default=0); ap.add_argument("--arc-base", type=int, default=0)
ap.add_argument("--nodes"); ap.add_argument("--arcs"); ap.add_argument("--limit", type=int)
a = ap.parse_args()
logging.basicConfig(level=logging.WARNING)
cfg = PipelineConfig(io=IOConfig(node_id_base=a.node_base, arc_id_base=a.arc_base))
trips, nets, runs = run_audit(a.root, a.out, cfg, a.nodes, a.arcs, a.limit)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(f"\n{len(trips)} trips, {len(nets)} network(s)")
print(nets.to_string(index=False))
cols = [c for c in ["trip_id", "n_obs", "labeled_fraction", "recall", "zero_candidate_obs", "miss_radius", "miss_cap",
                    "gt_dist_p95_m", "route_connected_base0", "route_connected_base1", "flags"] if c in trips]
print(trips[cols].sort_values("labeled_fraction" if "labeled_fraction" in trips else "trip_id").round(4).head(25).to_string(index=False))
print(f"\nfull tables + audit_summary.md in {a.out}")
