#!/usr/bin/env python
"""Generate the PathSnap training feature and label datasets."""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from pathsnap.align import ALIGNED
from pathsnap.candidates import build_index
from pathsnap.config import IOConfig, PipelineConfig
from pathsnap.dataset import discover_trips, process_trip, training_matrix
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.io import load_trip_files, parse_arcs, parse_nodes
from pathsnap.labels import LABEL_COLUMNS, META_COLUMNS, recall_report
from pathsnap.network import build_network


def write_table(df: pd.DataFrame, output_dir: Path, stem: str) -> list[str]:
    written = []
    csv_path = output_dir / f"{stem}.csv.gz"
    df.to_csv(csv_path, index=False, compression="gzip")
    written.append(str(csv_path))
    try:
        parquet_path = output_dir / f"{stem}.parquet"
        df.to_parquet(parquet_path, index=False)
        written.append(str(parquet_path))
    except Exception as exc:
        logging.warning("Could not write Parquet for %s: %s", stem, exc)
    return written


def poor_recall_trips(trip_reports: list[dict], limit: int = 10) -> list[dict]:
    rows = [r for r in trip_reports if r.get("labeled_fraction", 0) > 0]
    return sorted(rows, key=lambda r: r.get("recall", 0))[:limit]


def markdown_summary(
    trip_count: int,
    cand_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    report: dict,
    trip_reports: list[dict],
    failures: list[dict],
) -> str:
    valid_candidates = cand_df[cand_df["label_valid"].fillna(False).astype(bool)]
    n_obs = int(len(obs_df))
    n_labeled = int(report.get("n_labeled", 0))
    positive = int(valid_candidates["label"].eq(1).sum())
    negative = int(valid_candidates["label"].eq(0).sum())
    coverage = n_labeled / max(n_obs, 1)

    lines = [
        "# Training Feature Generation Summary",
        "",
        f"- Trips processed: {trip_count}",
        f"- Observations: {n_obs}",
        f"- Candidate rows: {len(cand_df)}",
        f"- Valid labeled observations: {n_labeled}",
        f"- Positive labels: {positive}",
        f"- Negative labels: {negative}",
        f"- Label coverage: {coverage:.6f}",
        f"- Candidate recall: {report.get('recall', 0.0):.6f}",
        f"- Candidate miss rate: {report.get('miss_rate', 0.0):.6f}",
    ]
    for k in (1, 3, 5, 10):
        lines.append(f"- Recall@{k}: {report.get(f'recall@{k}', 0.0):.6f}")

    lines += [
        "",
        "## Trips with particularly poor recall",
        "",
        "| Trip ID | Recall | Miss rate | Labeled observations | Miss: radius | Miss: cap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    poor = poor_recall_trips(trip_reports)
    if poor:
        for r in poor:
            lines.append(
                f"| {r['trip_id']} | {r.get('recall', 0.0):.4f} | "
                f"{r.get('miss_rate', 0.0):.4f} | {r.get('n_labeled', 0)} | "
                f"{r.get('miss_radius', 0)} | {r.get('miss_cap', 0)} |"
            )
    else:
        lines.append("| (none) | - | - | - | - | - |")

    lines += ["", "## Processing failures", ""]
    if failures:
        for failure in failures:
            lines.append(
                f"- `{failure['trip_id']}`: {failure['error_type']}: {failure['error']}"
            )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate training candidate features and labels")
    parser.add_argument(
        "--root",
        default=str(PROJECT_ROOT.parent / "Student folder" / "train"),
        help="Training dataset root",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "outputs"),
        help="Output directory",
    )
    parser.add_argument("--node-base", type=int, default=0)
    parser.add_argument("--arc-base", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = PipelineConfig(io=IOConfig(node_id_base=args.node_base, arc_id_base=args.arc_base))
    items = discover_trips(args.root)
    if args.limit:
        items = items[:args.limit]

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    nets, candidates, observations = {}, [], []
    trip_reports, failures = [], []

    for _trip_id, track_path, route_path, nodes_path, arcs_path in items:
        trip_id = track_path.stem
        try:
            key = (str(nodes_path), str(arcs_path))
            if key not in nets:
                net = build_network(parse_nodes(nodes_path), parse_arcs(arcs_path), cfg.io)
                nets[key] = (net, build_index(net, cfg.cand))
            net, index = nets[key]
            track, route = load_trip_files(track_path, route_path, cfg.io)
            cand_df, obs_df = process_trip(trip_id, net, index, track, route, cfg)

            if "label" not in cand_df:
                cand_df["label"] = np.int8(0)
            if "label_valid" not in cand_df:
                cand_df["label_valid"] = False
            if "aligned_route_pos" not in cand_df:
                cand_df["aligned_route_pos"] = -1

            candidates.append(cand_df)
            observations.append(obs_df)

            trip_report = recall_report(obs_df)
            trip_report["trip_id"] = trip_id
            trip_reports.append(trip_report)
        except Exception as exc:
            failures.append({
                "trip_id": trip_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    cand_df = (
        pd.concat(candidates, ignore_index=True)
        if candidates
        else pd.DataFrame(columns=["trip_id", "obs_idx", "arc_id", *FEATURE_COLUMNS, *LABEL_COLUMNS])
    )
    obs_df = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame()

    if cand_df.empty:
        feature_columns = ["trip_id", "obs_idx", "arc_id", *FEATURE_COLUMNS, "label", "label_valid", "aligned_route_pos"]
    else:
        feature_columns = [
            "trip_id", "obs_idx", "arc_id", *FEATURE_COLUMNS,
            "label", "label_valid", "aligned_route_pos",
        ]
    cand_output = cand_df[[c for c in feature_columns if c in cand_df.columns]].copy()

    obs_output = obs_df.copy()
    if not obs_output.empty:
        if "status" in obs_output:
            obs_output["alignment_status"] = obs_output["status"]
        else:
            obs_output["alignment_status"] = "no_route"
        if "gt_dist_m" not in obs_output:
            obs_output["gt_dist_m"] = np.nan
        if "gt_in_candidates" not in obs_output:
            obs_output["gt_in_candidates"] = False
        if "miss_reason" not in obs_output:
            obs_output["miss_reason"] = ""

    report = recall_report(obs_df) if "status" in obs_df else {
        "n_obs": len(obs_df),
        "n_labeled": 0,
        "labeled_fraction": 0.0,
        "recall": 0.0,
        "miss_rate": 1.0,
    }

    write_table(cand_output, output_dir, "training_features")
    write_table(obs_output, output_dir, "observation_summary")
    # Also preserve the two existing pipeline artifacts for compatibility.
    cand_df.to_csv(output_dir / "candidates.csv.gz", index=False, compression="gzip")
    obs_df.to_csv(output_dir / "observations.csv.gz", index=False, compression="gzip")

    summary = markdown_summary(
        len(observations),
        cand_output,
        obs_output,
        report,
        trip_reports,
        failures,
    )
    (output_dir / "generation_summary.md").write_text(summary, encoding="utf-8")

    print(f"Wrote {len(cand_output)} candidate rows and {len(obs_output)} observation rows")
    print(f"Wrote outputs under {output_dir}")
    return 0 if not failures else 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
