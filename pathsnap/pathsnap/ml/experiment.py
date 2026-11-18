"""Trip-grouped comparison, Bayesian tuning, and frozen evaluation partitions."""
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from ..features import FEATURE_COLUMNS
from .data import prepare, sha256, write_json
from .evaluation import evaluate, ranking_metrics
from .models import BASELINES, make_model, suggest_params


def training_rows(candidates, name, seed):
    valid = candidates[candidates.label_valid].copy()
    if name != "knn":
        return valid
    # Bound exact-neighbour inference cost; selection is training-only and label independent.
    rng = np.random.default_rng(seed)
    observations = valid[["trip_id", "obs_idx"]].drop_duplicates()
    selected = []
    for _, trip in observations.groupby("trip_id", sort=True):
        indices = rng.choice(len(trip), size=min(100, len(trip)), replace=False)
        selected.append(trip.iloc[indices])
    chosen = pd.concat(selected, ignore_index=True)
    return valid.merge(chosen, on=["trip_id", "obs_idx"], how="inner", validate="many_to_one")


def subset(tables, trips):
    c, o = tables
    return (c[c.trip_id.isin(trips)].reset_index(drop=True),
            o[o.trip_id.isin(trips)].reset_index(drop=True))


def create_splits(observations, seed=42, folds=3):
    trips = np.array(sorted(observations.trip_id.unique()))
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(trips, groups=trips))
    development, holdout = trips[train], trips[test]
    cv = []
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, va in splitter.split(development, groups=development):
        cv.append({"train": development[tr].tolist(), "validation": development[va].tolist()})
    return {"seed": seed, "development": development.tolist(), "holdout": holdout.tolist(), "folds": cv}


def positive_probability(model, x):
    positive = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(x)[:, positive]


def fit_evaluate(name, params, train, validation, seed, jobs):
    tc, _ = train
    vc, vo = validation
    fitting = training_rows(tc, name, seed)
    y = fitting["label"].to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training partition must contain both binary classes")
    model = make_model(name, params, seed, jobs)
    with threadpool_limits(limits=jobs):
        start = time.perf_counter()
        model.fit(fitting[FEATURE_COLUMNS].to_numpy(dtype=np.float32), y)
        fit_seconds = time.perf_counter() - start
        xval = vc[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        start = time.perf_counter()
        p = positive_probability(model, xval)
        prediction_seconds = time.perf_counter() - start
    metrics, trips, predictions = evaluate(vc, vo, p)
    metrics.update(fit_seconds=fit_seconds, predict_seconds=prediction_seconds,
                   predict_ms_per_1000_candidates=prediction_seconds * 1e6 / max(len(vc), 1),
                   training_candidate_rows=len(fitting), evaluated_candidate_rows=len(vc),
                   available_training_candidate_rows=int(tc.label_valid.sum()),
                   training_observations=int(fitting[["trip_id", "obs_idx"]].drop_duplicates().shape[0]),
                   training_observation_cap_per_trip=100 if name == "knn" else None)
    return model, metrics, trips, predictions, p


def cross_validate(name, params, tables, splits, output, seed, jobs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"name": name, "params": params, "splits": splits, "seed": seed, "jobs": jobs}
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(f"Conflicting experiment cache: {output}")
    write_json(identity_path, identity)
    folds, trip_rows = [], []
    for i, fold in enumerate(splits["folds"]):
        path = output / f"fold_{i}.json"
        trip_path = output / f"fold_{i}_trips.csv"
        if path.exists() and trip_path.exists():
            metrics = json.loads(path.read_text())
            per_trip = pd.read_csv(trip_path, dtype={"trip_id": str})
        else:
            train, validation = subset(tables, fold["train"]), subset(tables, fold["validation"])
            model, metrics, per_trip, _, _ = fit_evaluate(name, params, train, validation, seed, jobs)
            del model, train, validation
            gc.collect()
            write_json(path, metrics)
            per_trip.to_csv(trip_path, index=False)
        folds.append(metrics)
        trip_rows.append(per_trip)
        print(f"{name} {output.name} fold {i + 1}/{len(splits['folds'])}: "
              f"trip Top-1={metrics['top1_trip_mean']:.4f}, F1={metrics['macro_f1']:.4f}, "
              f"fit={metrics['fit_seconds']:.1f}s predict={metrics['predict_seconds']:.1f}s", flush=True)
    per_trip = pd.concat(trip_rows, ignore_index=True)
    if per_trip.trip_id.duplicated().any() or set(per_trip.trip_id) != set(splits["development"]):
        raise ValueError("CV must evaluate every development trip exactly once")
    summary = dict(model=name, params=params, top1_trip_mean=float(per_trip.top1.mean()),
                   top1_pooled=float(per_trip.correct.sum() / per_trip.aligned_observations.sum()),
                   top1_fold_std=float(np.std([m["top1_trip_mean"] for m in folds], ddof=1)),
                   macro_f1=float(np.mean([m["macro_f1"] for m in folds])),
                   log_loss=float(np.average([m["log_loss"] for m in folds],
                                             weights=[m["candidate_rows_valid"] for m in folds])),
                   fit_seconds=float(sum(m["fit_seconds"] for m in folds)),
                   predict_seconds=float(sum(m["predict_seconds"] for m in folds)),
                   macro_f1_fold_std=float(np.std([m["macro_f1"] for m in folds], ddof=1)))
    write_json(output / "summary.json", summary)
    per_trip.to_csv(output / "per_trip.csv", index=False)
    return summary


def baseline_search_params(name):
    """Enqueue the baseline within the declared tuning space for fair comparison."""
    keys = {
        "logistic_regression": ["C"], "knn": ["n_neighbors", "weights", "p"],
        "decision_tree": ["max_depth", "min_samples_leaf", "ccp_alpha"],
        "random_forest": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
        "hist_gradient_boosting": ["learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"],
    }
    defaults = dict(BASELINES[name], p=2, ccp_alpha=0.0)
    return {key: defaults[key] for key in keys[name]}


def remaining_trial_budget(study, target):
    """Queued trials have not run and must not reduce the completed-trial budget."""
    return max(0, target - sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials))


def run(archive, work, phase="all", seed=42, folds=3, trials=20, jobs=4, tune_model=None):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    output = work / "results"
    output.mkdir(exist_ok=True)
    table_path = prepare(archive, work / "data" / "train", "train", 80)
    tables = joblib.load(table_path)
    data_manifest = json.loads((table_path.parent / "manifest.json").read_text())
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    config = dict(seed=seed, folds=folds, trials=trials, jobs=jobs, baselines=BASELINES,
                  knn_training_observation_cap_per_trip=100,
                  data_fingerprint=data_manifest["fingerprint"],
                  code_sha256={p.name: sha256(p) for p in source_files})
    config_path = output / "experiment_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Experiment configuration changed; choose a new work directory")
    write_json(config_path, config)
    split_path = output / "splits.json"
    splits = create_splits(tables[1], seed, folds)
    if split_path.exists() and json.loads(split_path.read_text()) != splits:
        raise ValueError("Persisted splits differ")
    write_json(split_path, splits)
    versions = {p: importlib.metadata.version(p) for p in
                ["numpy", "pandas", "scipy", "scikit-learn", "optuna", "joblib", "matplotlib", "threadpoolctl"]}
    write_json(output / "environment.json", dict(python=platform.python_version(), packages=versions,
               platform=platform.platform(), cpu_count=os.cpu_count(),
               source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               recorded_utc=datetime.now(timezone.utc).isoformat()))
    if phase in ("all", "compare"):
        summaries = []
        for name in BASELINES:
            summaries.append(cross_validate(name, {}, tables, splits, output / "cv" / name, seed, jobs))
            pd.DataFrame(summaries).to_csv(output / "baseline_cv.csv", index=False)
        winner = sorted(summaries, key=lambda r: (-r["top1_trip_mean"], r["log_loss"], r["model"]))[0]
        write_json(output / "baseline_selection.json", winner)
    if phase == "compare":
        return
    strongest_baseline = json.loads((output / "baseline_selection.json").read_text())
    choice_path = output / "tuning_choice.json"
    previous_choice = json.loads(choice_path.read_text()) if choice_path.exists() else None
    name = tune_model or (previous_choice["model"] if previous_choice else strongest_baseline["model"])
    if previous_choice and previous_choice["model"] != name:
        raise ValueError("Tuning family is already frozen")
    write_json(choice_path, dict(model=name, strongest_baseline=strongest_baseline["model"],
               rule="Explicit promising family when supplied; otherwise highest CV Top-1 family"))
    winner = json.loads((output / "cv" / name / "summary.json").read_text())
    if phase in ("all", "tune"):
        # Serial seeded TPE: five startup trials, then Bayesian-guided trials.
        sampler_path = output / "tpe_sampler.joblib"
        checkpoint = joblib.load(sampler_path) if sampler_path.exists() else None
        sampler = checkpoint["sampler"] if checkpoint else optuna.samplers.TPESampler(seed=seed, n_startup_trials=5)
        study = optuna.create_study(study_name=f"pathsnap_{name}", direction="maximize", sampler=sampler,
                                    storage=f"sqlite:///{(output / 'tuning.sqlite3').as_posix()}", load_if_exists=True)
        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        if checkpoint and checkpoint["completed_trials"] != completed:
            raise ValueError("Sampler checkpoint and completed trial count disagree")
        if not study.trials:
            study.enqueue_trial(baseline_search_params(name))

        def objective(trial):
            params = suggest_params(trial, name)
            # Trial 0 uses the already measured baseline; it is not refitted.
            baseline = baseline_search_params(name)
            if params == baseline:
                result = winner
            else:
                result = cross_validate(name, params, tables, splits, output / "tuning" / f"trial_{trial.number:03d}", seed, jobs)
            for metric in ["macro_f1", "log_loss", "fit_seconds", "predict_seconds"]:
                trial.set_user_attr(metric, result[metric])
            return result["top1_trip_mean"]

        start = time.perf_counter()
        remaining = remaining_trial_budget(study, trials)

        def save_trial(study, trial):
            study.trials_dataframe().to_csv(output / "tuning_trials.csv", index=False)
            joblib.dump(dict(sampler=study.sampler,
                        completed_trials=sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)), sampler_path)

        study.optimize(objective, n_trials=remaining, n_jobs=1,
                       callbacks=[save_trial])
        best = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                      key=lambda t: (-t.value, t.user_attrs["log_loss"], t.number))[0]
        # Freeze selection using CV only, before evaluating holdout/student data.
        improved = best.value > strongest_baseline["top1_trip_mean"] or (
            best.value == strongest_baseline["top1_trip_mean"] and best.user_attrs["log_loss"] < strongest_baseline["log_loss"])
        selection = dict(model=name, baseline_cv=winner, best_trial=best.number,
                         strongest_baseline_cv=strongest_baseline,
                         selected_model=name if improved else strongest_baseline["model"],
                         best_params=best.params, tuned_cv_top1=best.value,
                         tuned_cv_macro_f1=best.user_attrs["macro_f1"], tuned_cv_log_loss=best.user_attrs["log_loss"],
                         selected_variant="tuned" if improved else "untuned",
                         n_trials=sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
                         tuning_session_seconds=time.perf_counter() - start,
                         selection_rule="highest trip-mean observation Top-1; tie break lower candidate log loss")
        write_json(output / "selection.json", selection)
    if phase == "tune":
        return
    selection = json.loads((output / "selection.json").read_text())
    # Holdout is never passed to the Bayesian objective.
    holdout_rows = []
    comparisons = [(name, "untuned", {}), (name, "tuned", selection["best_params"])]
    if selection["selected_model"] != name:
        comparisons.append((selection["selected_model"], "strongest_untuned", {}))
    for compare_name, variant, params in comparisons:
        model, metrics, trip, predictions, _ = fit_evaluate(compare_name, params,
            subset(tables, splits["development"]), subset(tables, splits["holdout"]), seed, jobs)
        holdout_rows.append(dict(model=compare_name, variant=variant, **metrics))
        trip.to_csv(output / f"holdout_{variant}_trips.csv", index=False)
        predictions.to_csv(output / f"holdout_{variant}_predictions.csv.gz", index=False)
        print(f"Frozen holdout {variant}: {metrics['top1_trip_mean']:.4f}", flush=True)
        del model
        gc.collect()
    pd.DataFrame(holdout_rows).to_csv(output / "holdout_comparison.csv", index=False)
    # Build student tables only AFTER configurations and selected variant are fixed.
    student = joblib.load(prepare(archive, work / "data" / "student_test", "student_test", 10))
    artifacts = work / "models"
    artifacts.mkdir(exist_ok=True)
    rows = []
    configs = [(n, "untuned", {}) for n in BASELINES] + [(name, "tuned", selection["best_params"])]
    for model_name, variant, params in configs:
        model, metrics, trip, predictions, p = fit_evaluate(model_name, params, tables, student, seed, jobs)
        tag = f"{model_name}_{variant}"
        bundle = dict(pipeline=model, feature_columns=FEATURE_COLUMNS, model=model_name, variant=variant,
                      params=dict(BASELINES[model_name], **params), seed=seed,
                      training_candidate_rows=metrics["training_candidate_rows"],
                      training_observations=metrics["training_observations"],
                      training_observation_cap_per_trip=metrics["training_observation_cap_per_trip"],
                      pipeline_config=data_manifest["fingerprint"]["pipeline_config"],
                      train_trip_ids=sorted(tables[1].trip_id.unique()), versions=versions,
                      score_semantics="binary candidate-match posterior; not an HMM emission likelihood")
        joblib.dump(bundle, artifacts / f"{tag}.joblib", compress=3)
        if model_name == selection["selected_model"] and variant == selection["selected_variant"]:
            joblib.dump(bundle, artifacts / "selected_model.joblib", compress=3)
        trip.to_csv(output / f"student_{tag}_trips.csv", index=False)
        predictions.to_csv(output / f"student_{tag}_predictions.csv.gz", index=False)
        np.savez_compressed(output / f"student_{tag}_probabilities.npz", probability=p)
        rows.append(dict(model=model_name, variant=variant, **metrics))
        pd.DataFrame(rows).to_csv(output / "student_comparison.csv", index=False)
        print(f"Student {tag}: trip Top-1={metrics['top1_trip_mean']:.4f}, F1={metrics['macro_f1']:.4f}", flush=True)
        del model, bundle
        gc.collect()
    for partition, part in [("holdout", subset(tables, splits["holdout"])), ("student", student)]:
        c, o = part
        m, trips, _ = ranking_metrics(c, o, -c.dist_m.to_numpy())
        write_json(output / f"{partition}_nearest_road_baseline.json", m)
        trips.to_csv(output / f"{partition}_nearest_road_trips.csv", index=False)
    write_json(artifacts / "model_metadata.json", selection)
    print(f"Completed. Selected artifact: {artifacts / 'selected_model.joblib'}", flush=True)
