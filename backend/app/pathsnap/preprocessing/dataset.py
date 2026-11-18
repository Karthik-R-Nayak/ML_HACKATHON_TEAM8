"""Per-trip pipeline, dataset discovery, group-aware splits."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .align import align_route
from .candidates import build_index, generate_candidates
from .config import PipelineConfig
from .features import FEATURE_COLUMNS, build_candidate_features, observation_features
from .io import load_trip_files, parse_arcs, parse_nodes
from .labels import LABEL_COLUMNS, META_COLUMNS, attach_labels, observation_table, recall_report
from .network import build_network

log = logging.getLogger(__name__)


def process_trip(trip_id, net, index, track, route_arc_ids, cfg: PipelineConfig = PipelineConfig()):
    """Returns (candidate_df, obs_df). route_arc_ids=None -> inference-only (no labels)."""
    if cfg.feat.heading_mode != "central":
        raise ValueError("PathSnap training and inference require heading_mode='central'")
    x, y = net.projector.forward(track["lon"].to_numpy(), track["lat"].to_numpy())
    xy = np.column_stack([x, y])
    t = track["t"].to_numpy()
    # ---- feature path: no route access ----
    cands, n_within = generate_candidates(net, index, xy, cfg.cand)
    obs_feat = observation_features(xy, t, cfg.feat)
    cand_df = build_candidate_features(net, cands, obs_feat)
    cand_df.insert(0, "trip_id", trip_id)
    # ---- label path ----
    align = None
    if route_arc_ids is not None:
        ridx = net.route_to_idx(route_arc_ids)
        align = align_route(xy, ridx, net, cfg.align)
        cand_df = attach_labels(cand_df, align)
    obs_df = observation_table(trip_id, net, cands, obs_feat, n_within, align, cfg.cand)
    obs_df["gps_lon"] = track["lon"].to_numpy()
    obs_df["gps_lat"] = track["lat"].to_numpy()
    if "orig_row" in track.columns:
        obs_df["source_obs_idx"] = track["orig_row"].to_numpy()
    return cand_df, obs_df


def training_matrix(cand_df: pd.DataFrame):
    """Return candidate rows from usable labeled observations only.

    Candidate-miss observations stay in observation-level evaluation reports, but
    cannot contribute an all-negative training group. A group is one (trip_id,
    obs_idx) and must have valid labels and at least one positive candidate.
    """
    if cand_df.empty:
        d = cand_df
    else:
        group_keys = ["trip_id", "obs_idx"]
        grouped = cand_df.groupby(group_keys, sort=False, dropna=False)
        eligible = grouped["label_valid"].all() & grouped["label"].any()
        eligible_keys = eligible[eligible].index
        key_index = pd.MultiIndex.from_frame(cand_df[group_keys])
        d = cand_df.loc[key_index.isin(eligible_keys)]
    return d[FEATURE_COLUMNS], d["label"], d["trip_id"]


def group_folds(trip_ids, n_splits=5, seed=0):
    """Yield (train_trips, val_trips); a trip is never in both."""
    ids = np.array(sorted(set(trip_ids)))
    np.random.default_rng(seed).shuffle(ids)
    folds = np.array_split(ids, n_splits)
    for k in range(n_splits):
        yield np.concatenate([f for j, f in enumerate(folds) if j != k]), folds[k]


def discover_trips(root, nodes_path=None, arcs_path=None):
    """Find tracks (+ optional routes). Network: explicit paths, per-trip stem match, or a single shared pair."""
    root = Path(root)
    tracks = sorted(root.rglob("*.track"))
    shared_n = [nodes_path] if nodes_path else sorted(root.rglob("*.nodes"))
    shared_a = [arcs_path] if arcs_path else sorted(root.rglob("*.arcs"))
    items = []
    for tp in tracks:
        n = tp.with_suffix(".nodes"); a = tp.with_suffix(".arcs")
        if not (n.exists() and a.exists()):
            if len(shared_n) != 1 or len(shared_a) != 1:
                raise ValueError(f"cannot choose a network for {tp}: found {len(shared_n)} .nodes / "
                                 f"{len(shared_a)} .arcs; pass nodes_path/arcs_path")
            n, a = Path(shared_n[0]), Path(shared_a[0])
        r = tp.with_suffix(".route")
        rel = tp.relative_to(root)
        # Dataset trip folders use the trip ID as both directory and file stem.
        # Preserve the real ID instead of returning "id/id" for that layout.
        trip_id = rel.parent.name if rel.parent.name == tp.stem else str(rel.with_suffix(""))
        items.append((trip_id, tp, r if r.exists() else None, n, a))
    return items


def run_pipeline(root, out_dir=None, cfg: PipelineConfig = PipelineConfig(), nodes_path=None, arcs_path=None,
                 limit=None, trip_ids=None):
    items = discover_trips(root, nodes_path, arcs_path)
    if trip_ids is not None:
        items = [it for it in items if it[0] in set(trip_ids)]
    if limit:
        items = items[:limit]
    nets, cands, obss = {}, [], []
    for trip_id, tp, rp, npath, apath in items:
        key = (str(npath), str(apath))
        if key not in nets:
            net = build_network(parse_nodes(npath), parse_arcs(apath), cfg.io)
            nets[key] = (net, build_index(net, cfg.cand))
        net, index = nets[key]
        track, route = load_trip_files(tp, rp, cfg.io)
        c, o = process_trip(trip_id, net, index, track, route, cfg)
        cands.append(c); obss.append(o)
    cand_df = pd.concat(cands, ignore_index=True) if cands else pd.DataFrame()
    obs_df = pd.concat(obss, ignore_index=True) if obss else pd.DataFrame()
    if out_dir:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        cand_df.to_csv(out / "candidates.csv.gz", index=False)
        obs_df.to_csv(out / "observations.csv.gz", index=False)
    report = recall_report(obs_df) if "status" in obs_df else {}
    return cand_df, obs_df, report


def run_training_pipeline(dataset_root, out_dir=None, cfg: PipelineConfig = PipelineConfig(),
                          nodes_path=None, arcs_path=None, limit=None):
    """Preprocess training trips only; never traverse student/evaluator partitions."""
    train_root = Path(dataset_root) / "train"
    if not train_root.is_dir():
        raise ValueError("training dataset root must contain a train/ directory")
    return run_pipeline(train_root, out_dir=out_dir, cfg=cfg, nodes_path=nodes_path,
                        arcs_path=arcs_path, limit=limit)
