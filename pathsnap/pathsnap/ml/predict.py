"""Route-free scoring interface for Members 3 and 4."""
import joblib
import numpy as np
import pandas as pd

from ..features import FEATURE_COLUMNS
from .data import KEY


def score_candidates(artifact, candidates):
    """Preserve input order and IDs; labels/reference routes are never required.

    match_probability is a binary classifier posterior, not a generative HMM
    emission likelihood. log_match_score is an optional clipped scoring potential.
    Member 3 must validate its use with transition scores on development trips.
    """
    bundle = joblib.load(artifact) if isinstance(artifact, (str, bytes)) or hasattr(artifact, "__fspath__") else artifact
    columns = bundle["feature_columns"]
    if columns != FEATURE_COLUMNS:
        raise ValueError("Artifact feature schema differs from the pipeline")
    required = KEY + ["arc_id"] + columns
    missing = set(required) - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing inference columns: {sorted(missing)}")
    if candidates.duplicated(KEY + ["arc_id"]).any():
        raise ValueError("Duplicate candidate IDs")
    out = candidates[KEY + ["arc_id"]].copy()
    if candidates.empty:
        out["match_probability"] = pd.Series(dtype=float)
        out["log_match_score"] = pd.Series(dtype=float)
        return out
    x = candidates[columns].to_numpy(dtype=np.float32)
    if np.isinf(x).any():
        raise ValueError("Infinite inference features")
    classes = bundle["pipeline"].classes_
    positive = int(np.flatnonzero(classes == 1)[0])
    prob = bundle["pipeline"].predict_proba(x)[:, positive]
    out["match_probability"] = prob
    out["log_match_score"] = np.log(np.clip(prob, 1e-9, 1.0))
    return out
