"""Attach labels (after features) and measure candidate recall / miss rate."""
import numpy as np
import pandas as pd

from .align import ALIGNED, AlignResult
from .config import CandidateConfig
from .network import RoadNetwork

LABEL_COLUMNS = ["label", "label_valid", "aligned_route_pos"]
META_COLUMNS = ["trip_id", "obs_idx", "arc_id", "arc_idx"]


def attach_labels(cand_df: pd.DataFrame, align: AlignResult) -> pd.DataFrame:
    """Adds label columns; never modifies feature columns."""
    df = cand_df.copy()
    o = df["obs_idx"].to_numpy()
    valid = align.status[o] == ALIGNED
    df["label_valid"] = valid
    df["label"] = (valid & (df["arc_idx"].to_numpy() == align.arc_idx[o])).astype(np.int8)
    df["aligned_route_pos"] = align.route_pos[o]
    return df


def observation_table(trip_id, net: RoadNetwork, cand_df, obs_feat, n_within, align: AlignResult,
                      cfg: CandidateConfig) -> pd.DataFrame:
    n = len(obs_feat)
    obs = obs_feat.copy()
    obs.insert(0, "trip_id", trip_id)
    obs["n_candidates"] = np.bincount(cand_df["obs_idx"], minlength=n)
    obs["n_within_radius"] = n_within
    if align is None:
        return obs
    labeled = align.status == ALIGNED
    obs["status"] = align.status
    obs["aligned_arc_id"] = np.where(labeled, net.arc_ids[np.maximum(align.arc_idx, 0)], -1)
    obs["aligned_route_pos"] = align.route_pos
    obs["gt_dist_m"] = align.dist_m
    pos = cand_df[cand_df["obs_idx"].to_numpy() >= 0]
    hit = pos.loc[(pos["arc_idx"].to_numpy() == align.arc_idx[pos["obs_idx"].to_numpy()])
                  & labeled[pos["obs_idx"].to_numpy()]]
    rank = np.full(n, np.nan)
    rank[hit["obs_idx"].to_numpy()] = hit["dist_rank"].to_numpy()
    obs["gt_rank"] = rank
    obs["gt_in_candidates"] = labeled & ~np.isnan(rank)
    reason = np.full(n, "", dtype=object)
    miss = labeled & np.isnan(rank)
    reason[miss & (align.dist_m > cfg.radius_m)] = "radius"
    reason[miss & ~(align.dist_m > cfg.radius_m)] = "cap"
    obs["miss_reason"] = reason
    return obs


def recall_report(obs: pd.DataFrame, ks=(1, 3, 5, 10)) -> dict:
    """Candidate recall / miss rate over label-valid observations."""
    lab = obs[obs["status"] == ALIGNED]
    rep = {"n_obs": int(len(obs)), "n_labeled": int(len(lab)),
           "labeled_fraction": float(len(lab) / max(len(obs), 1))}
    if len(lab) == 0:
        return rep
    rec = float(lab["gt_in_candidates"].mean())
    rep.update(recall=rec, miss_rate=1 - rec,
               miss_radius=int((lab["miss_reason"] == "radius").sum()),
               miss_cap=int((lab["miss_reason"] == "cap").sum()),
               mean_candidates=float(obs["n_candidates"].mean()),
               obs_with_zero_candidates=int((obs["n_candidates"] == 0).sum()))
    for k in ks:
        rep[f"recall@{k}"] = float((lab["gt_rank"] <= k).mean())
    return rep


def recall_by_radius(obs: pd.DataFrame, radii) -> dict:
    """Recall if the radius were r (ignoring the K cap). Valid for r <= alignment search radius."""
    lab = obs[obs["status"] == ALIGNED]
    return {float(r): float((lab["gt_dist_m"] <= r).mean()) for r in radii}
