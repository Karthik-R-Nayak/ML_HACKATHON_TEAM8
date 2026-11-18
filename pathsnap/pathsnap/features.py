"""Inference-time features. Nothing here may depend on the route / labels."""
import numpy as np
import pandas as pd

from .config import FeatureConfig
from .geo import angle_diff_deg, bearing_deg
from .network import RoadNetwork

FEATURE_COLUMNS = [
    "dist_m", "proj_frac", "heading_diff_deg", "gps_speed_mps", "gps_bearing_deg", "arc_bearing_deg",
    "arc_length_m", "from_in_deg", "from_out_deg", "to_in_deg", "to_out_deg",
    "dt_prev_s", "dt_next_s", "gps_heading_valid", "dist_rank", "dist_minus_min_m", "n_candidates",
]


def observation_features(xy: np.ndarray, t: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    n = len(xy)
    nan = np.full(n, np.nan)
    dt_prev, dt_next = nan.copy(), nan.copy()
    if n > 1:
        dt_prev[1:] = np.diff(t)
        dt_next[:-1] = np.diff(t)
    idx = np.arange(n)
    if cfg.heading_mode == "backward":
        a, b = np.maximum(idx - 1, 0), idx
    elif cfg.heading_mode == "central":
        a, b = np.maximum(idx - 1, 0), np.minimum(idx + 1, n - 1)
    else:
        raise ValueError("heading_mode must be 'central' or 'backward'")
    dx, dy = xy[b, 0] - xy[a, 0], xy[b, 1] - xy[a, 1]
    dist = np.hypot(dx, dy)
    dt = t[b] - t[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(dt > 0, dist / dt, np.nan)
    valid = dist >= cfg.min_move_m
    bearing = np.where(valid, bearing_deg(dx, dy), np.nan)
    return pd.DataFrame({
        "obs_idx": idx, "gps_speed_mps": speed, "gps_bearing_deg": bearing,
        "gps_heading_valid": valid.astype(np.int8), "dt_prev_s": dt_prev, "dt_next_s": dt_next})


def build_candidate_features(net: RoadNetwork, cands: pd.DataFrame, obs_feat: pd.DataFrame) -> pd.DataFrame:
    df = cands.merge(obs_feat, on="obs_idx", how="left")
    a = df["arc_idx"].to_numpy()
    df["arc_id"] = net.arc_ids[a]
    df["arc_bearing_deg"] = net.bearing[a]
    df["arc_length_m"] = net.length[a]
    df["heading_diff_deg"] = angle_diff_deg(df["gps_bearing_deg"].to_numpy(), df["arc_bearing_deg"].to_numpy())
    fn, tn = net.arc_from[a], net.arc_to[a]
    df["from_in_deg"], df["from_out_deg"] = net.in_deg[fn], net.out_deg[fn]
    df["to_in_deg"], df["to_out_deg"] = net.in_deg[tn], net.out_deg[tn]
    g = df.groupby("obs_idx")["dist_m"]
    df["dist_minus_min_m"] = df["dist_m"] - g.transform("min")
    df["n_candidates"] = g.transform("size")
    return df
