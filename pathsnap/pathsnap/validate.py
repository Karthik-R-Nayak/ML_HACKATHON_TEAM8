"""Sanity checks to run on REAL sample files before a full run."""
import numpy as np

from .labels import recall_by_radius, recall_report


def check_id_bases(n_nodes, arcs, route_ids, n_arcs):
    """Heuristics that flag wrong id-base assumptions. Returns list of warning strings."""
    w = []
    amax = max(arcs["from_node"].max(), arcs["to_node"].max())
    amin = min(arcs["from_node"].min(), arcs["to_node"].min())
    if amax >= n_nodes:
        w.append(f"arcs reference node id {amax} but only {n_nodes} nodes: ids likely 1-based or non-sequential")
    if amin == 1 and amax == n_nodes:
        w.append("node ids span 1..N -> use IOConfig(node_id_base=1)")
    if len(route_ids):
        if route_ids.max() >= n_arcs:
            w.append(f"route arc id {route_ids.max()} >= n_arcs {n_arcs}: arc ids likely 1-based or non-sequential")
        if route_ids.min() == 1 and route_ids.max() == n_arcs:
            w.append("route ids span 1..A -> use IOConfig(arc_id_base=1)")
    return w


def trip_summary(trip_id, track, route, obs):
    rep = recall_report(obs)
    rep.update(trip_id=trip_id, n_track_rows=len(track),
               n_route_arcs=None if route is None else len(route),
               n_distinct_route_arcs=None if route is None else int(len(set(route.tolist()))),
               median_gt_dist_m=float(np.nanmedian(obs["gt_dist_m"])) if "gt_dist_m" in obs else None)
    return rep
