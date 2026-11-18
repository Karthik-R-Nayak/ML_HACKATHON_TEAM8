"""Regression checks for leakage, metric denominators, and the inference contract."""
import numpy as np
import pandas as pd
import pytest

from pathsnap.features import FEATURE_COLUMNS
from pathsnap.ml.data import validate_tables
from pathsnap.ml.evaluation import evaluate, rank_candidates
from pathsnap.ml.experiment import create_splits, training_rows, remaining_trial_budget
from pathsnap.ml.models import BASELINES, make_model
from pathsnap.ml.predict import score_candidates


@pytest.fixture
def example():
    # Observation 1 is an aligned miss, 2 has zero candidates, 3 is unlabeled.
    c = pd.DataFrame(dict(trip_id=["a"] * 5, obs_idx=[0, 0, 1, 1, 3],
        arc_id=[10, 11, 20, 21, 30], dist_rank=[1, 2, 1, 2, 1],
        label=[0, 1, 0, 0, 0], label_valid=[True, True, True, True, False]))
    o = pd.DataFrame(dict(trip_id=["a"] * 4, obs_idx=[0, 1, 2, 3],
        status=["aligned", "aligned", "aligned", "no_route_arc_nearby"],
        aligned_arc_id=[11, 22, 40, -1], gt_in_candidates=[True, False, False, False],
        n_candidates=[2, 2, 0, 1]))
    return c, o


def test_top1_counts_misses_zero_candidates_and_excludes_unlabeled(example):
    c, o = example
    m, _, _ = evaluate(c, o, np.array([0.2, 0.8, 0.3, 0.1, 0.99]))
    assert m["top1_pooled"] == pytest.approx(1 / 3)
    assert m["top1_given_candidate"] == 1
    assert m["candidate_miss_rate"] == pytest.approx(2 / 3)
    assert m["zero_candidate_aligned"] == 1
    assert m["candidate_rows_valid"] == 4
    assert m["confusion_matrix"] == [[3, 0], [0, 1]]


def test_tie_break_does_not_use_labels(example):
    c, _ = example
    expected = rank_candidates(c, np.ones(len(c)))
    c["label"] = 1 - c["label"]
    pd.testing.assert_frame_equal(expected, rank_candidates(c, np.ones(len(c))))
    assert expected.iloc[0].predicted_arc_id == 10


def test_splits_disjoint_and_every_development_trip_validated_once():
    o = pd.DataFrame({"trip_id": np.repeat([f"t{i:02d}" for i in range(80)], 3)})
    split = create_splits(o)
    development, holdout = set(split["development"]), set(split["holdout"])
    assert len(development) == 64 and len(holdout) == 16
    assert not development & holdout
    seen = []
    for fold in split["folds"]:
        assert not set(fold["train"]) & set(fold["validation"])
        assert not (set(fold["train"]) | set(fold["validation"])) & holdout
        seen.extend(fold["validation"])
    assert len(seen) == len(set(seen)) == 64
    assert set(seen) == development
    assert create_splits(o) == split


@pytest.mark.parametrize("name", list(BASELINES))
def test_all_models_handle_missing_features_and_return_probabilities(name):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 17)).astype("float32")
    x[:10, 2] = np.nan
    x[:, 11] = np.nan  # keep an all-missing feature in a fold
    y = np.tile([0, 1], 40)
    model = make_model(name, jobs=1)
    model.fit(x, y)
    p = model.predict_proba(x[:3])
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
    if name == "hist_gradient_boosting":
        assert model.named_steps["classifier"].early_stopping is False


def test_imputer_and_scaler_fit_training_rows_only():
    x = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    model = make_model("logistic_regression", jobs=1).fit(x, [0, 0, 1, 1])
    assert model.named_steps["imputer"].statistics_[1] == 6
    statistics = model.named_steps["imputer"].statistics_.copy()
    model.predict_proba([[1000000, np.nan]])
    np.testing.assert_array_equal(model.named_steps["imputer"].statistics_, statistics)


def test_inference_uses_only_feature_columns_and_preserves_ids():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, len(FEATURE_COLUMNS))).astype("float32")
    model = make_model("logistic_regression", jobs=1).fit(x, np.tile([0, 1], 20))
    bundle = {"pipeline": model, "feature_columns": FEATURE_COLUMNS}
    c = pd.DataFrame(x[:4], columns=FEATURE_COLUMNS)
    c["trip_id"], c["obs_idx"], c["arc_id"] = "a", [1, 0, 1, 0], [2, 3, 4, 5]
    a = score_candidates(bundle, c)
    c["label"], c["aligned_route_pos"] = 999, -999
    b = score_candidates(bundle, c)
    pd.testing.assert_frame_equal(a, b)
    assert a.arc_id.tolist() == [2, 3, 4, 5]
    assert score_candidates(bundle, c.iloc[:0]).empty


def test_validation_rejects_inconsistent_ground_truth(example):
    c, o = example
    c[FEATURE_COLUMNS] = 0.0
    # Feature assignment includes dist_rank, which is not needed by this check.
    validate_tables(c, o)
    c.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="Labels disagree"):
        validate_tables(c, o)


def test_knn_training_budget_is_grouped_reproducible_and_label_independent():
    c = pd.DataFrame({"trip_id": np.repeat(["a", "b"], 240),
                      "obs_idx": np.tile(np.repeat(np.arange(120), 2), 2),
                      "label_valid": True, "label": np.tile([0, 1], 240)})
    first = training_rows(c, "knn", 42)
    second = training_rows(c.assign(label=1 - c.label), "knn", 42)
    assert len(first) == 400
    assert first.groupby(["trip_id", "obs_idx"]).size().eq(2).all()
    assert first.groupby("trip_id").obs_idx.nunique().eq(100).all()
    pd.testing.assert_frame_equal(first[["trip_id", "obs_idx"]], second[["trip_id", "obs_idx"]])
    assert len(training_rows(c, "random_forest", 42)) == len(c)


def test_queued_trial_does_not_reduce_completed_trial_budget():
    import optuna
    study = optuna.create_study()
    study.enqueue_trial({"C": 1.0})
    assert remaining_trial_budget(study, 12) == 12
    study.add_trial(optuna.trial.create_trial(value=0.5))
    assert remaining_trial_budget(study, 12) == 11
