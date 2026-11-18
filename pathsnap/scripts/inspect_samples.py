#!/usr/bin/env python
"""Validate parsing, id conventions, recall and alignment on a few REAL trips before the full run.

usage: python scripts/inspect_samples.py DATA_ROOT [--n 5] [--node-base 0] [--arc-base 0]
                                         [--nodes F --arcs F]
"""
import argparse, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from pathsnap.candidates import build_index
from pathsnap.config import IOConfig, PipelineConfig
from pathsnap.dataset import discover_trips, process_trip
from pathsnap.io import load_trip_files, parse_arcs, parse_nodes
from pathsnap.labels import recall_by_radius, recall_report
from pathsnap.network import build_network
from pathsnap.validate import check_id_bases, trip_summary

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("--n", type=int, default=5)
ap.add_argument("--node-base", type=int, default=0); ap.add_argument("--arc-base", type=int, default=0)
ap.add_argument("--nodes"); ap.add_argument("--arcs")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
cfg = PipelineConfig(io=IOConfig(node_id_base=a.node_base, arc_id_base=a.arc_base))
items = discover_trips(a.root, a.nodes, a.arcs)[: a.n]
print(f"{len(items)} sample trips")
nets, rows, all_obs = {}, [], []
for trip_id, tp, rp, npath, apath in items:
    key = (npath, apath)
    if key not in nets:
        nodes, arcs = parse_nodes(npath), parse_arcs(apath)
        print(f"\nnetwork {npath.name}: {len(nodes)} nodes, {len(arcs)} arcs")
        print(nodes.head(3).to_string()); print(arcs.head(3).to_string())
        net = build_network(nodes, arcs, cfg.io)
        nets[key] = (net, build_index(net, cfg.cand), nodes, arcs)
    net, index, nodes, arcs = nets[key]
    track, route = load_trip_files(tp, rp, cfg.io)
    print(f"\n[{trip_id}] track rows={len(track)} route arcs={None if route is None else len(route)}")
    print(track.head(3).to_string())
    if route is not None:
        for w in check_id_bases(len(nodes), arcs, route, len(arcs)):
            print("  WARNING:", w)
    c, o = process_trip(trip_id, net, index, track, route, cfg)
    if route is not None:
        rows.append(trip_summary(trip_id, track, route, o)); all_obs.append(o)
if rows:
    print("\n=== per-trip ===")
    print(pd.DataFrame(rows).round(3).to_string())
    obs = pd.concat(all_obs)
    print("\n=== overall ===")
    for k, v in recall_report(obs).items(): print(f"  {k}: {v}")
    print("  recall by radius:", {k: round(v, 3) for k, v in recall_by_radius(obs, [25, 50, 75, 100, 150]).items()})
    print("\nCHECK: low labeled_fraction -> wrong id base / projection / search radius;"
          " large miss_radius -> increase candidate radius; large miss_cap -> raise max_candidates.")
