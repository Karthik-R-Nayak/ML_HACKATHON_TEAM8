"""Candidate metrics plus observation ranking with explicit coverage denominators."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score

from ..align import ALIGNED
from .data import KEY


def rank_candidates(candidates, scores):
    if len(candidates) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Every candidate must have a finite score")
    ranked = candidates[KEY + ["arc_id", "dist_rank"]].copy()
    ranked["score"] = np.asarray(scores)
    # Ground-truth independent tie breaking, shared by every model.
    ranked = ranked.sort_values(KEY + ["score", "dist_rank", "arc_id"],
                                ascending=[True, True, False, True, True], kind="stable")
    return ranked.drop_duplicates(KEY).rename(columns={"arc_id": "predicted_arc_id"})


def ranking_metrics(candidates, observations, scores):
    predictions = rank_candidates(candidates, scores)
    aligned = observations.loc[observations.status.eq(ALIGNED)].copy()
    if aligned.empty:
        raise ValueError("No valid aligned observations to evaluate")
    scored = aligned.merge(predictions[KEY + ["predicted_arc_id", "score"]], on=KEY, how="left", validate="one_to_one")
    scored["correct"] = scored.predicted_arc_id.eq(scored.aligned_arc_id)
    trip_rows = []
    for trip, d in scored.groupby("trip_id", observed=True):
        hits = d.gt_in_candidates.astype(bool)
        trip_rows.append(dict(trip_id=trip, aligned_observations=len(d), correct=int(d.correct.sum()),
                              top1=float(d.correct.mean()), candidate_miss_rate=float((~hits).mean()),
                              top1_given_candidate=float(d.loc[hits, "correct"].mean()) if hits.any() else np.nan))
    per_trip = pd.DataFrame(trip_rows)
    hits = scored.gt_in_candidates.astype(bool)
    metrics = dict(top1_trip_mean=float(per_trip.top1.mean()), top1_pooled=float(scored.correct.mean()),
                   top1_given_candidate=float(scored.loc[hits, "correct"].mean()) if hits.any() else None,
                   candidate_miss_rate=float((~hits).mean()), aligned_observations=len(scored),
                   unlabeled_observations=int((observations.status != ALIGNED).sum()),
                   zero_candidate_aligned=int((scored.n_candidates == 0).sum()))
    return metrics, per_trip, scored


def evaluate(candidates, observations, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    metrics, per_trip, predictions = ranking_metrics(candidates, observations, p)
    valid = candidates.label_valid.to_numpy(dtype=bool)
    y, prob = candidates.label.to_numpy()[valid], p[valid]
    predicted = (prob >= 0.5).astype(int)
    metrics.update(accuracy=float(accuracy_score(y, predicted)),
                   precision=float(precision_score(y, predicted, zero_division=0)),
                   recall=float(recall_score(y, predicted, zero_division=0)),
                   macro_f1=float(f1_score(y, predicted, average="macro", labels=[0, 1], zero_division=0)),
                   log_loss=float(log_loss(y, np.column_stack([1 - prob, prob]), labels=[0, 1])),
                   brier_score=float(brier_score_loss(y, prob)),
                   candidate_rows_valid=int(valid.sum()),
                   positive_fraction=float(np.mean(y)),
                   confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1]).tolist())
    return metrics, per_trip, predictions
