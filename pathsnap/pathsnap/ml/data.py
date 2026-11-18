"""Build validated training tables using Member 1's unchanged feature pipeline."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

import joblib
import numpy as np
import pandas as pd

from ..align import ALIGNED
from ..audit import audit_ids, audit_network
from ..candidates import build_index
from ..config import PipelineConfig
from ..dataset import discover_trips, process_trip
from ..features import FEATURE_COLUMNS
from ..io import load_trip_files, parse_arcs, parse_nodes
from ..labels import recall_report
from ..network import build_network

KEY = ["trip_id", "obs_idx"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def extract_partition(archive, destination, partition):
    """Only extract the explicitly allowed partition, rejecting traversal paths."""
    if partition not in ("train", "student_test"):
        raise ValueError("Evaluator data is not supported by the experiment runner")
    destination = Path(destination).resolve()
    prefix = f"Student folder/{partition}/"
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            relative = Path(name[len(prefix):])
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive entry: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            content = z.read(name)
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f"Existing raw file differs: {target}")
            if not target.exists():
                target.write_bytes(content)
    return destination


def validate_tables(candidates, observations):
    """Fail on ambiguous identities/labels; allow genuine missing candidates."""
    c, o = candidates, observations
    if o.duplicated(KEY).any() or c.duplicated(KEY + ["arc_id"]).any():
        raise ValueError("Duplicate observation or candidate identities")
    if not c["label"].isin([0, 1]).all() or c["label_valid"].isna().any():
        raise ValueError("Missing or nonbinary candidate labels")
    joined = c.merge(o[KEY + ["status", "aligned_arc_id"]], on=KEY, how="left", validate="many_to_one")
    expected_valid = joined.status.eq(ALIGNED)
    if joined.status.isna().any() or not np.array_equal(joined.label_valid, expected_valid):
        raise ValueError("Candidate validity disagrees with observation alignment")
    expected_label = expected_valid & joined.arc_id.eq(joined.aligned_arc_id)
    if not np.array_equal(joined.label.to_numpy(), expected_label.astype(int).to_numpy()):
        raise ValueError("Labels disagree with aligned ground truth")
    sums = c.groupby(KEY, observed=True).label.sum()
    positives = pd.MultiIndex.from_frame(o[KEY]).map(sums).fillna(0).to_numpy()
    if not np.array_equal(positives, o.gt_in_candidates.astype(int).to_numpy()):
        raise ValueError("Ground-truth candidate coverage disagrees with positive counts")
    counts = c.groupby(KEY, observed=True).size()
    actual = pd.MultiIndex.from_frame(o[KEY]).map(counts).fillna(0).to_numpy()
    if not np.array_equal(actual, o.n_candidates.to_numpy()):
        raise ValueError("Candidate count disagrees with observation table")
    if np.isinf(c[FEATURE_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values")


def prepare(archive, output, partition="train", expected_trips=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()
    manifest_path = output / "manifest.json"
    fingerprint = {"archive_sha256": sha256(archive), "partition": partition,
                   "pipeline_config": asdict(cfg), "features": FEATURE_COLUMNS}
    source_root = Path(__file__).resolve().parents[1]
    fingerprint["pipeline_source_sha256"] = {
        p.name: sha256(p) for p in sorted(source_root.glob("*.py"))}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Cached data differs from current input/configuration; use a new output directory")
        if expected_trips is not None and manifest["n_trips"] != expected_trips:
            raise ValueError("Unexpected cached trip count")
        return output / "tables.joblib"
    root = extract_partition(archive, output / "raw", partition)
    items = discover_trips(root)
    if expected_trips is not None and len(items) != expected_trips:
        raise ValueError(f"Expected {expected_trips} trips, found {len(items)}")
    cs, os, audits = [], [], []
    started = time.perf_counter()
    for trip_id, tp, rp, npth, apth in items:
        if rp is None:
            raise ValueError(f"Missing reference route for {trip_id}")
        nodes, arcs = parse_nodes(npth), parse_arcs(apth)
        net = build_network(nodes, arcs, cfg.io)
        track, route = load_trip_files(tp, rp, cfg.io)
        c, o = process_trip(trip_id, net, build_index(net, cfg.cand), track, route, cfg)
        validate_tables(c, o)
        c[FEATURE_COLUMNS] = c[FEATURE_COLUMNS].astype("float32")
        cs.append(c)
        os.append(o)
        audits.append(dict(trip_id=trip_id, candidate_rows=len(c), **recall_report(o),
                           **audit_ids(nodes, arcs, net, route, cfg), **audit_network(nodes, arcs, net)))
        print(f"{partition} {len(cs)}/{len(items)} {trip_id}: {len(o)} observations, {len(c)} candidates", flush=True)
    c, o = pd.concat(cs, ignore_index=True), pd.concat(os, ignore_index=True)
    validate_tables(c, o)
    joblib.dump((c, o), output / "tables.joblib", compress=3)
    pd.DataFrame(audits).to_csv(output / "trip_audit.csv", index=False)
    o.to_csv(output / "observations.csv.gz", index=False)
    c[FEATURE_COLUMNS].isna().sum().rename("missing_rows").to_csv(output / "feature_missingness.csv")
    manifest = dict(fingerprint=fingerprint, n_trips=len(items), candidate_rows=len(c),
                    preparation_seconds=time.perf_counter() - started, **recall_report(o))
    write_json(manifest_path, manifest)
    return output / "tables.joblib"
