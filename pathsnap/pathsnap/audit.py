"""Read-only dataset audit (new module; does not alter the pipeline).

Answers, per trip and per partition, from the ACTUAL files:
  - file/row counts, track quality (dt, duplicates, invalid coords, jumps)
  - network quality (zero-length/self-loop/duplicate arcs, isolated nodes, components)
  - node-id / arc-id convention evidence (range fit and route node-connectivity under base 0 vs 1)
  - candidate recall, zero-candidate observations, miss reasons
  - WHERE and WHY observations stay unlabeled (leading/trailing/interior runs, distance to route and to network)
Nothing here is hard-coded; every number is computed from the files given.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .align import ALIGNED
from .candidates import build_index
from .config import PipelineConfig
from .dataset import discover_trips, process_trip
from .geo import point_segment_projection
from .io import _is_num, _tokenise, load_trip_files, parse_arcs, parse_nodes
from .labels import recall_report
from .network import build_network

log = logging.getLogger(__name__)


def _components(n_nodes, f, t):
    parent = np.arange(n_nodes)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(f, t):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n_nodes)])
    return roots


def audit_network(nodes, arcs, net):
    f, t = net.arc_from, net.arc_to
    used = np.zeros(len(nodes), bool); used[f] = True; used[t] = True
    roots = _components(len(nodes), f, t)
    comp = pd.Series(roots[used]).value_counts()
    pairs = pd.Series(f.astype(np.int64) * len(nodes) + t)
    return dict(
        n_nodes=len(nodes), n_arcs=len(arcs),
        zero_length_arcs=int((net.length < 0.5).sum()), self_loops=int((f == t).sum()),
        duplicate_arcs=int(pairs.duplicated().sum()), isolated_nodes=int((~used).sum()),
        n_components=int(len(comp)), largest_component_nodes=int(comp.iloc[0]) if len(comp) else 0,
        arc_len_min=float(net.length.min()), arc_len_median=float(np.median(net.length)),
        arc_len_max=float(net.length.max()),
        dup_node_coords=int(nodes.duplicated().sum()))


def audit_track(path, track, xy, n_valid_cols=3):
    raw_t = None
    try:
        rows = _tokenise(path)
        rows = rows[1:] if not _is_num(rows[0][0]) else rows
        raw_t = np.array([float(r[2]) for r in rows])
    except Exception:
        pass
    t = track["t"].to_numpy()
    dt = np.diff(t)
    step = np.hypot(*np.diff(xy, axis=0).T) if len(xy) > 1 else np.array([])
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(dt > 0, step / dt, np.nan)
    lon, lat = track["lon"].to_numpy(), track["lat"].to_numpy()
    return dict(
        n_obs=len(track), duration_s=float(t[-1] - t[0]) if len(t) else 0.0,
        nonmonotone_t=int((np.diff(raw_t) < 0).sum()) if raw_t is not None else -1,
        dup_rows=int(track[["lon", "lat", "t"]].duplicated().sum()),
        dup_timestamps=int(pd.Series(t).duplicated().sum()),
        nan_coords=int((~np.isfinite(lon) | ~np.isfinite(lat)).sum()),
        zero_coords=int(((lon == 0) & (lat == 0)).sum()),
        dt_eq_1s_frac=float((dt == 1).mean()) if len(dt) else np.nan,
        dt_max_s=float(dt.max()) if len(dt) else np.nan,
        n_gaps_gt5s=int((dt > 5).sum()), n_dt_le0=int((dt <= 0).sum()),
        speed_max_mps=float(np.nanmax(v)) if np.isfinite(v).any() else np.nan,
        n_speed_gt50mps=int((np.nan_to_num(v) > 50).sum()),
        path_len_m=float(step.sum()),
        lon_min=float(lon.min()), lon_max=float(lon.max()), lat_min=float(lat.min()), lat_max=float(lat.max()))


def audit_ids(nodes, arcs, net, route_ids, cfg):
    """Evidence for id conventions. Route connectivity is the strongest test of the arc-id base:
    under the right base, consecutive route arcs share a node (to_node[k] == from_node[k+1])."""
    out = {}
    n = len(nodes)
    allnodes = np.concatenate([arcs["from_node"], arcs["to_node"]])
    for b in (0, 1):
        out[f"arcs_nodes_in_range_base{b}"] = float(((allnodes - b >= 0) & (allnodes - b < n)).mean())
    out["arc_node_id_min"], out["arc_node_id_max"] = int(allnodes.min()), int(allnodes.max())
    if route_ids is not None and len(route_ids):
        A = net.n_arcs
        out["route_id_min"], out["route_id_max"] = int(route_ids.min()), int(route_ids.max())
        for b in (0, 1):
            idx = route_ids - b
            ok = (idx >= 0) & (idx < A)
            out[f"route_in_range_base{b}"] = float(ok.mean())
            pair_ok = ok[:-1] & ok[1:]
            if pair_ok.any():
                i0, i1 = idx[:-1][pair_ok], idx[1:][pair_ok]
                out[f"route_connected_base{b}"] = float((net.arc_to[i0] == net.arc_from[i1]).mean())
            else:
                out[f"route_connected_base{b}"] = np.nan
        c0, c1 = out["route_connected_base0"], out["route_connected_base1"]
        cfgb = cfg.io.arc_id_base
        cur, alt = (c0, c1) if cfgb == 0 else (c1, c0)
        out["arc_base_suspect"] = bool(np.nan_to_num(alt) > np.nan_to_num(cur) + 0.2)
        out["route_repeated_arcs"] = int(len(route_ids) - len(set(route_ids.tolist())))
    return out


def _runs(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[cuts + 1]]); ends = np.concatenate([idx[cuts], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def _nearest(net, xy, sel, arc_subset=None):
    ids = np.arange(net.n_arcs) if arc_subset is None else arc_subset
    out = np.empty(len(sel))
    for k, i in enumerate(sel):
        d, _ = point_segment_projection(xy[i, 0], xy[i, 1], net.ax[ids], net.ay[ids], net.bx[ids], net.by[ids])
        out[k] = d.min() if d.size else np.nan
    return out


def audit_trip(trip_id, net, index, track, route_ids, cfg, track_path=None):
    c, o = process_trip(trip_id, net, index, track, route_ids, cfg)
    x, y = net.projector.forward(track["lon"].to_numpy(), track["lat"].to_numpy())
    xy = np.column_stack([x, y])
    row = dict(trip_id=trip_id, n_route_arcs=None if route_ids is None else len(route_ids))
    if track_path is not None:
        row.update(audit_track(track_path, track, xy))
    row["zero_candidate_obs"] = int((o["n_candidates"] == 0).sum())
    runs = []
    ridx = None
    if route_ids is not None:
        in_range = (route_ids - net.arc_base >= 0) & (route_ids - net.arc_base < net.n_arcs)
        ridx = (route_ids - net.arc_base)[in_range]
        rep = recall_report(o)
        row.update({k: v for k, v in rep.items() if k not in ("n_obs",)})
        dists = o["gt_dist_m"].to_numpy()
        row["gt_dist_median_m"] = float(np.nanmedian(dists)) if np.isfinite(dists).any() else np.nan
        row["gt_dist_p95_m"] = float(np.nanpercentile(dists, 95)) if np.isfinite(dists).any() else np.nan
        row["route_len_m"] = float(net.length[ridx].sum())
        row["route_to_path_len_ratio"] = row["route_len_m"] / max(row.get("path_len_m", np.nan), 1e-9)
        unl = (o["status"] != ALIGNED).to_numpy()
        for kind, mask in (("unlabeled", unl), ("zero_candidates", (o["n_candidates"] == 0).to_numpy())):
            for s, e in _runs(mask):
                sel = np.arange(s, e + 1)
                st = o["status"].iloc[sel].value_counts().to_dict()
                runs.append(dict(
                    trip_id=trip_id, kind=kind, start=s, end=e, n=e - s + 1,
                    position="leading" if s == 0 else "trailing" if e == len(o) - 1 else "interior",
                    reasons=st,
                    nearest_route_arc_m_median=float(np.median(_nearest(net, xy, sel, ridx))) if ridx.size else np.nan,
                    nearest_any_arc_m_median=float(np.median(_nearest(net, xy, sel))),
                    max_dt_in_run_s=float(np.nanmax(track["t"].diff().to_numpy()[sel])) if e > 0 else 0.0))
        pos = pd.DataFrame(runs)
        for p in ("leading", "trailing", "interior"):
            row[f"unlabeled_{p}"] = int(pos[(pos.kind == "unlabeled") & (pos.position == p)]["n"].sum()) if len(pos) else 0
    return row, runs


def flag_trip(row, thr):
    f = []
    if row.get("labeled_fraction", 1) < thr["labeled_fraction"]: f.append("low_labeled_fraction")
    if row.get("recall", 1) < thr["recall"]: f.append("low_recall")
    if row.get("zero_candidate_obs", 0) > 0: f.append("zero_candidates")
    if row.get("arc_base_suspect"): f.append("arc_id_base_suspect")
    if row.get("route_connected_base0", 1) < thr["route_connected"] and row.get("route_connected_base1", 1) < thr["route_connected"]:
        f.append("route_not_node_connected")
    if row.get("nonmonotone_t", 0) > 0: f.append("nonmonotone_time")
    if row.get("dup_rows", 0) > 0: f.append("duplicate_rows")
    if row.get("n_speed_gt50mps", 0) > 0: f.append("speed_jumps")
    if row.get("n_gaps_gt5s", 0) > 0: f.append("time_gaps")
    return f


def run_audit(root, out_dir, cfg: PipelineConfig = PipelineConfig(), nodes_path=None, arcs_path=None,
              limit=None, thresholds=None):
    thr = dict(labeled_fraction=0.98, recall=0.99, route_connected=0.99)
    thr.update(thresholds or {})
    items = discover_trips(root, nodes_path, arcs_path)
    if limit:
        items = items[:limit]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    nets, net_rows, rows, runs = {}, [], [], []
    for trip_id, tp, rp, npath, apath in items:
        try:
            key = (str(npath), str(apath))
            if key not in nets:
                nodes, arcs = parse_nodes(npath), parse_arcs(apath)
                net = build_network(nodes, arcs, cfg.io)
                nets[key] = (net, build_index(net, cfg.cand), nodes, arcs)
                net_rows.append(dict(network=npath.name, **audit_network(nodes, arcs, net)))
            net, index, nodes, arcs = nets[key]
            track, route = load_trip_files(tp, rp, cfg.io)
            row, r = audit_trip(trip_id, net, index, track, route, cfg, tp)
            row.update(audit_ids(nodes, arcs, net, route, cfg))
            row.update(files=f"track{'+route' if rp else ''}", network=npath.name,
                       track_bytes=tp.stat().st_size, route_bytes=rp.stat().st_size if rp else 0)
            row["flags"] = ",".join(flag_trip(row, thr))
            rows.append(row); runs += r
        except Exception as e:   # a broken trip must not hide the others
            rows.append(dict(trip_id=trip_id, flags="AUDIT_ERROR", error=f"{type(e).__name__}: {e}"))
    trips = pd.DataFrame(rows)
    trips.to_csv(out / "audit_trips.csv", index=False)
    pd.DataFrame(net_rows).to_csv(out / "audit_networks.csv", index=False)
    pd.DataFrame(runs).to_csv(out / "audit_unlabeled_runs.csv", index=False)
    flagged = trips[trips["flags"].fillna("") != ""]
    lines = [f"# Audit of `{root}`", f"- trips audited: {len(trips)}  | networks: {len(net_rows)}",
             f"- trips flagged: {len(flagged)}", f"- thresholds: {thr}", ""]
    for _, r in flagged.iterrows():
        lines.append(f"- **{r['trip_id']}**: {r['flags']}")
    (out / "audit_summary.md").write_text("\n".join(lines))
    return trips, pd.DataFrame(net_rows), pd.DataFrame(runs)
