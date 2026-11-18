"""Candidate arc generation. Uses ONLY GPS + network (never the route)."""
import numpy as np
import pandas as pd

from .config import CandidateConfig
from .geo import point_segment_projection
from .index import GridIndex
from .network import RoadNetwork


def build_index(net: RoadNetwork, cfg: CandidateConfig = CandidateConfig()) -> GridIndex:
    return GridIndex(net.ax, net.ay, net.bx, net.by, cfg.cell_m)


def generate_candidates(net: RoadNetwork, index: GridIndex, xy: np.ndarray, cfg: CandidateConfig):
    """Returns (candidates DataFrame, n_within_radius per observation)."""
    obs, arc, dist, frac, rank = [], [], [], [], []
    n_within = np.zeros(len(xy), dtype=np.int64)
    for i, (px, py) in enumerate(xy):
        ids = index.query(px, py, cfg.radius_m)
        if ids.size == 0:
            continue
        d, t = point_segment_projection(px, py, net.ax[ids], net.ay[ids], net.bx[ids], net.by[ids])
        keep = d <= cfg.radius_m
        ids, d, t = ids[keep], d[keep], t[keep]
        n_within[i] = ids.size
        order = np.argsort(d, kind="stable")[: cfg.max_candidates]
        k = len(order)
        obs.append(np.full(k, i)); arc.append(ids[order]); dist.append(d[order])
        frac.append(t[order]); rank.append(np.arange(1, k + 1))
    cat = lambda L, dt: np.concatenate(L) if L else np.empty(0, dtype=dt)
    df = pd.DataFrame({
        "obs_idx": cat(obs, int), "arc_idx": cat(arc, int), "dist_m": cat(dist, float),
        "proj_frac": cat(frac, float), "dist_rank": cat(rank, int)})
    return df, n_within
