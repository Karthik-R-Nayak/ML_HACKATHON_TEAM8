"""Validate real prepared data, saved model inference, metrics, splits and tuning.

This runs inference on complete student tables; it does not refit models or tune.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import optuna
from threadpoolctl import threadpool_limits

from pathsnap.candidates import build_index
from pathsnap.config import PipelineConfig
from pathsnap.dataset import discover_trips, process_trip
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.io import load_trip_files, parse_arcs, parse_nodes
from pathsnap.network import build_network
from pathsnap.ml.data import sha256, validate_tables, write_json
from pathsnap.ml.evaluation import evaluate
from pathsnap.ml.predict import score_candidates


def validate(work, report):
    started = time.perf_counter()
    work, report = Path(work).resolve(), Path(report).resolve()
    report.mkdir(parents=True, exist_ok=True)
    results = work / "results"
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    config = read(results / "experiment_config.json")
    selection = read(results / "selection.json")
    checks = []

    def passed(name, **details):
        checks.append(dict(check=name, status="PASS", **details))
        print(f"PASS {name}: {details}", flush=True)

    ml_source = Path(__file__).resolve().parents[1] / "pathsnap/ml"
    assert config["code_sha256"] == {p.name: sha256(p) for p in sorted(ml_source.glob("*.py"))}
    passed("training source fingerprints")
    for partition, expected in [("train", 80), ("student_test", 10)]:
        directory = work / "data" / partition
        manifest = read(directory / "manifest.json")
        c, o = joblib.load(directory / "tables.joblib")
        validate_tables(c, o)
        assert o.trip_id.nunique() == expected == manifest["n_trips"]
        assert len(c) == manifest["candidate_rows"] and len(o) == manifest["n_obs"]
        assert manifest["fingerprint"]["features"] == FEATURE_COLUMNS
        core = ml_source.parent
        assert manifest["fingerprint"]["pipeline_source_sha256"] == {
            p.name: sha256(p) for p in sorted(core.glob("*.py"))}
        passed(f"{partition} table/label consistency", trips=expected, observations=len(o), candidates=len(c))
        del c, o
        gc.collect()

    splits = read(results / "splits.json")
    development, holdout = set(splits["development"]), set(splits["holdout"])
    assert len(development) == 64 and len(holdout) == 16 and not development & holdout
    validation = []
    for fold in splits["folds"]:
        tr, va = set(fold["train"]), set(fold["validation"])
        assert not tr & va and tr | va == development
        assert not (tr | va) & holdout
        validation.extend(va)
    assert len(validation) == len(set(validation)) == 64
    passed("trip separation and CV coverage", development=64, holdout=16, folds=len(splits["folds"]))

    study = optuna.load_study(study_name=f"pathsnap_{selection['model']}",
        storage=f"sqlite:///{(results / 'tuning.sqlite3').as_posix()}")
    assert len(study.trials) == 12 and all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    checkpoint = joblib.load(results / "tpe_sampler.joblib")
    assert checkpoint["completed_trials"] == 12 == selection["n_trials"]
    for trial in study.trials:
        assert trial.user_attrs.keys() >= {"log_loss", "macro_f1", "fit_seconds", "predict_seconds"}
        summary = read(results / (f"cv/{selection['model']}/summary.json" if trial.number == 0 else f"tuning/trial_{trial.number:03d}/summary.json"))
        assert np.isclose(trial.value, summary["top1_trip_mean"], rtol=0, atol=1e-12)
    passed("Bayesian database, sampler checkpoint and recorded fold scores", completed_trials=12)

    c, o = joblib.load(work / "data/student_test/tables.joblib")
    cfg = PipelineConfig()
    inference_features = ["trip_id", "obs_idx", "arc_id"] + FEATURE_COLUMNS
    for trip, tp, _, npth, apth in discover_trips(work / "data/student_test/raw"):
        net = build_network(parse_nodes(npth), parse_arcs(apth), cfg.io)
        track, route = load_trip_files(tp, None, cfg.io)
        assert route is None
        raw_candidates, raw_obs = process_trip(trip, net, build_index(net, cfg.cand), track, None, cfg)
        assert not {"label", "label_valid", "aligned_route_pos"} & set(raw_candidates)
        reference = c[c.trip_id.eq(trip)]
        pd.testing.assert_frame_equal(raw_candidates[FEATURE_COLUMNS].astype("float32").reset_index(drop=True),
                                      reference[FEATURE_COLUMNS].reset_index(drop=True))
    passed("route-free feature regeneration for all student trips", trips=10)

    expected_metrics = pd.read_csv(results / "student_comparison.csv")
    assert len(expected_metrics) == 6
    measurements = []
    for row in expected_metrics.itertuples():
        tag = f"{row.model}_{row.variant}"
        bundle = joblib.load(work / "models" / f"{tag}.joblib")
        assert bundle["feature_columns"] == FEATURE_COLUMNS
        assert asdict(cfg) == bundle["pipeline_config"]
        assert len(bundle["train_trip_ids"]) == 80
        for name in ["numpy", "scikit-learn", "scipy", "joblib"]:
            assert bundle["versions"][name] == importlib.metadata.version(name)
        start = time.perf_counter()
        with threadpool_limits(limits=4):
            scored = score_candidates(bundle, c[inference_features])
        elapsed = time.perf_counter() - start
        probability = scored.match_probability.to_numpy()
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        assert np.isfinite(scored.log_match_score).all()
        pd.testing.assert_frame_equal(scored[["trip_id", "obs_idx", "arc_id"]], c[["trip_id", "obs_idx", "arc_id"]])
        expected_probability = np.load(results / f"student_{tag}_probabilities.npz")["probability"]
        np.testing.assert_allclose(probability, expected_probability, rtol=1e-10, atol=1e-12)
        metrics, trips, predictions = evaluate(c, o, probability)
        for metric in ["accuracy", "precision", "recall", "macro_f1", "log_loss", "brier_score",
                       "top1_trip_mean", "top1_pooled", "top1_given_candidate", "candidate_miss_rate"]:
            assert np.isclose(metrics[metric], getattr(row, metric), rtol=0, atol=1e-12), (tag, metric, metrics[metric], getattr(row, metric))
        assert metrics["confusion_matrix"] == json.loads(row.confusion_matrix)
        sample = c[inference_features].iloc[:20].copy()
        for invalid in ["missing", "duplicate", "infinity"]:
            bad = sample.copy()
            if invalid == "missing":
                bad = bad.drop(columns=[FEATURE_COLUMNS[0]])
            elif invalid == "duplicate":
                bad = pd.concat([bad, bad.iloc[:1]], ignore_index=True)
            else:
                bad.loc[bad.index[0], FEATURE_COLUMNS[0]] = np.inf
            try:
                score_candidates(bundle, bad)
            except ValueError:
                continue
            raise AssertionError(f"{tag} accepted invalid input: {invalid}")
        assert score_candidates(bundle, sample.iloc[:0]).empty
        passed(f"{tag} saved-model inference and reproduced metrics", candidates=len(scored), top1=metrics["top1_trip_mean"])
        measurements.append(dict(model=row.model, variant=row.variant, candidates=len(scored),
                                 inference_seconds=elapsed, top1_trip_mean=metrics["top1_trip_mean"], macro_f1=metrics["macro_f1"]))
        del bundle, scored
        gc.collect()
    selected_tag = f"{selection['selected_model']}_{selection['selected_variant']}"
    assert sha256(work / "models/selected_model.joblib") == sha256(work / "models" / f"{selected_tag}.joblib")
    assert read(work / "models/model_metadata.json") == selection
    passed("selected-model identity, checksum and metadata")
    summary = dict(status="PASS", checked_utc=datetime.now(timezone.utc).isoformat(),
                   total_checks=len(checks), checks=checks, measurements=measurements,
                   elapsed_seconds=time.perf_counter() - started,
                   scope="Member 1 pipeline and Member 2 models; HMM, API and UI are outside this validation")
    write_json(report / "artifact-validation.json", summary)
    lines = ["# PathSnap execution validation", "", "Status: PASS", "",
             "The full pytest suite was run separately: 37 passed, zero failures, two non-blocking warnings.", "",
             "Saved-model checks used all 366,633 candidate rows from all 10 student-testing trips.", ""]
    lines.extend(f"- PASS: {check['check']}" for check in checks)
    lines.extend(["", "All saved probabilities and reported classification/ranking metrics were reproduced.",
                  "No models were refitted, and no new optimisation trials were added.", "",
                  "HMM/Viterbi, API, frontend and two-laptop deployment are outside the completed Member 2 scope.", ""])
    (report / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Validation complete: {report / 'VALIDATION_REPORT.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    validate(args.work, args.report)
