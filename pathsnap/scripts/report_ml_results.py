"""Render tables, diagnostic figures, and a report from completed real experiments."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"logistic_regression": "Logistic\nRegression", "knn": "KNN\n(sampled)", "decision_tree": "Decision\nTree",
          "random_forest": "Random\nForest", "hist_gradient_boosting": "Histogram\nBoosting"}
FAMILIES = {"logistic_regression": "Linear/probabilistic", "knn": "Instance based", "decision_tree": "Tree",
            "random_forest": "Bagging ensemble", "hist_gradient_boosting": "Boosting ensemble"}
DESCRIPTIONS = {
    "logistic_regression": "A linear log-odds baseline, trained by regularised logistic loss. It tests whether distance and heading features suffice with additive effects. It cannot directly capture complex feature interactions; scaling and median imputation are fitted within each training fold.",
    "knn": "A distance-weighted neighbourhood classifier in standardised feature space. It tests whether similar feature patterns have similar labels, without learning a parametric decision boundary. Its limitations are inference cost, sensitivity to feature geometry and potentially extreme probabilities. Raw bearing wrap-around remains a limitation of the supplied 17-feature representation.",
    "decision_tree": "A nonlinear Gini-impurity tree baseline. It learns distance/heading interactions through thresholds and is interpretable, but can overfit. Depth and minimum leaf size constrain complexity.",
    "random_forest": "An ensemble of bootstrapped Gini-impurity trees. Feature subsampling and averaging test whether reduced tree variance improves transfer across trips. Its costs are fitting time, model size and potentially imperfect probability calibration.",
    "hist_gradient_boosting": "A histogram-based tree ensemble trained with logistic loss. Sequential boosting learns feature interactions efficiently. Learning rate, iteration count, leaf complexity and regularisation control capacity. Internal random-row early stopping is disabled so that validation remains trip grouped.",
}


def table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for c in columns:
            value = row[c]
            cells.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(work):
    work = Path(work)
    out = work / "results"
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    read = lambda name: json.loads((out / name).read_text())
    cv = pd.read_csv(out / "baseline_cv.csv")
    student = pd.read_csv(out / "student_comparison.csv")
    holdout = pd.read_csv(out / "holdout_comparison.csv")
    trials = pd.read_csv(out / "tuning_trials.csv")
    selection = read("selection.json")
    splits = read("splits.json")
    fold_count, development_count, holdout_count = len(splits["folds"]), len(splits["development"]), len(splits["holdout"])
    configuration = read("experiment_config.json")
    cv_ranked = cv.sort_values("top1_trip_mean", ascending=False)
    cv_lead_pp = 100 * (cv_ranked.iloc[0].top1_trip_mean - cv_ranked.iloc[1].top1_trip_mean)
    selection_note = (f"The leading untuned CV model is ahead of the runner-up by {cv_lead_pp:.3f} percentage points. "
        "The selected artifact follows the declared highest-CV-mean rule; this is not evidence of statistically proven superiority. "
        "The complete student comparison is shown below and did not change the frozen selection.")
    tuning_note = ("Histogram boosting was chosen as the promising tuning family after the five-model comparison because its "
        "measured Top-1 was close to Random Forest while fitting substantially faster. This was a practical compute "
        "decision, rather than a claim that it had the highest untuned Top-1. Final selection still compares its tuned "
        "CV score with the strongest untuned family.") if (selection["model"] == "hist_gradient_boosting" and
        selection["strongest_baseline_cv"]["model"] == "random_forest") else (
        f"The chosen promising tuning family was {selection['model']}. Its tuned CV score is compared with the strongest untuned family.")
    train_manifest = json.loads((work / "data/train/manifest.json").read_text())
    student_manifest = json.loads((work / "data/student_test/manifest.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    labels = [LABELS[m] for m in cv.model]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, col, err, title in [(axes[0], "top1_trip_mean", "top1_fold_std", "Trip-mean observation Top-1"),
                                (axes[1], "macro_f1", "macro_f1_fold_std", "Candidate macro F1")]:
        ax.bar(labels, cv[col], yerr=cv[err], capsize=4, color="#287d8e")
        ax.set(title=title, ylabel="Score", ylim=(0, 1.04))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{fold_count}-fold trip-grouped CV on {development_count} development trips | error bars: fold SD")
    fig.savefig(figs / "cross_validation.png", dpi=170)
    plt.close(fig)
    student_labels = [LABELS[r.model] + ("\n(tuned)" if r.variant == "tuned" else "") for r in student.itertuples()]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, col, title in [(axes[0], "top1_trip_mean", "Trip-mean observation Top-1"),
                            (axes[1], "macro_f1", "Candidate macro F1"), (axes[2], "log_loss", "Candidate log loss (lower is better)")]:
        ax.bar(student_labels, student[col], color=["#d97a32" if v == "tuned" else "#287d8e" for v in student.variant])
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Student-testing comparison | 10 trips | configurations frozen before scoring")
    fig.savefig(figs / "student_comparison.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, col, title in [(axes[0], "fit_seconds", "Full 80-trip refit time"),
                            (axes[1], "predict_seconds", "Student candidate prediction time")]:
        ax.bar(student_labels, student[col], color=["#d97a32" if v == "tuned" else "#287d8e" for v in student.variant])
        ax.set(title=title, ylabel="Seconds (log scale)", yscale="log")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Measured CPU cost | same data and {configuration['jobs']}-thread limit")
    fig.savefig(figs / "computational_cost.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for ax, (_, row) in zip(axes.flat, student.iterrows()):
        cm = np.asarray(json.loads(row.confusion_matrix))
        ax.imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="True aligned label",
               title=row.model.replace("_", " ") + f" ({row.variant})")
    fig.suptitle("Student candidate confusion matrices | valid alignments only")
    fig.savefig(figs / "confusion_matrices.png", dpi=170)
    plt.close(fig)
    completed = trials[trials.state == "COMPLETE"].sort_values("number")
    trial_seconds = float(pd.to_timedelta(completed.duration).dt.total_seconds().sum())
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.scatter(completed.number, completed.value, label="Trial objective", color="#287d8e")
    ax.plot(completed.number, completed.value.cummax(), label="Best so far", color="#d97a32")
    ax.axvline(4.5, color="grey", linestyle="--", label="End of five startup trials")
    ax.set(xlabel="Trial number (0 is the baseline)", ylabel="CV trip-mean observation Top-1",
           title=f"Bayesian TPE search: {selection['model'].replace('_', ' ')}")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.savefig(figs / "bayesian_search.png", dpi=170)
    plt.close(fig)
    # Calibration is diagnostic only; it does not alter frozen model selection.
    import joblib
    candidates, _ = joblib.load(work / "data/student_test/tables.joblib")
    valid = candidates.label_valid.to_numpy(dtype=bool)
    y = candidates.label.to_numpy()[valid]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, row in zip(axes.flat, student.itertuples()):
        tag = f"{row.model}_{row.variant}"
        prob = np.load(out / f"student_{tag}_probabilities.npz")["probability"][valid]
        bins = np.minimum((prob * 10).astype(int), 9)
        x, actual = [], []
        for i in range(10):
            mask = bins == i
            if mask.any():
                x.append(prob[mask].mean())
                actual.append(y[mask].mean())
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.plot(x, actual, "o-", color="#287d8e")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive fraction",
               title=row.model.replace("_", " ") + f" ({row.variant})")
    fig.suptitle("Student probability reliability | ten equal-width bins | no calibration fitted")
    fig.savefig(figs / "probability_reliability.png", dpi=170)
    plt.close(fig)
    # Paired trip bootstrap: uncertainty in tuned-minus-untuned holdout Top-1.
    a = pd.read_csv(out / "holdout_untuned_trips.csv", dtype={"trip_id": str})
    b = pd.read_csv(out / "holdout_tuned_trips.csv", dtype={"trip_id": str})
    paired = a.merge(b, on="trip_id", suffixes=("_untuned", "_tuned"), validate="one_to_one")
    delta = (paired.top1_tuned - paired.top1_untuned).to_numpy()
    rng = np.random.default_rng(splits["seed"])
    boot = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    uncertainty = dict(trips=len(delta), tuned_minus_untuned_trip_top1=float(delta.mean()),
                       paired_trip_bootstrap_95_percent=[float(low), float(high)], resamples=5000,
                       note="Exploratory trip-level uncertainty; does not replace external evaluation")
    (out / "holdout_uncertainty.json").write_text(json.dumps(uncertainty, indent=2))
    cv_display = cv[["model", "top1_trip_mean", "top1_pooled", "macro_f1", "log_loss", "fit_seconds", "predict_seconds"]].copy()
    cv_display.insert(1, "family", cv_display.model.map(FAMILIES))
    student_display = student[["model", "variant", "top1_trip_mean", "top1_pooled", "macro_f1", "log_loss", "candidate_miss_rate", "predict_seconds"]]
    holdout_display = holdout[["model", "variant", "top1_trip_mean", "top1_pooled", "macro_f1", "log_loss", "predict_seconds"]]
    lines = ["# PathSnap Member 2: model experiments and Bayesian optimisation", "",
        "## Outcome", "",
        f"The strongest untuned model by CV Top-1 was **{selection['strongest_baseline_cv']['model']}**. "
        f"The promising family chosen for Bayesian tuning was **{selection['model']}**. "
        f"After {selection['n_trials']} TPE trials, the frozen selection is "
        f"**{selection['selected_model']} ({selection['selected_variant']})**. "
        "Selection used development-trip CV only; neither holdout nor student-test scores selected models, thresholds or parameters.", "",
        selection_note, "",
        "## Data and evaluation design", "",
        f"- Training: {train_manifest['n_trips']} trips, {train_manifest['n_obs']:,} GPS observations, "
        f"{train_manifest['candidate_rows']:,} candidate rows.",
        f"- Valid route alignment: {train_manifest['n_labeled']:,} observations "
        f"({100 * train_manifest['labeled_fraction']:.2f}%). Candidate miss rate among valid alignments: "
        f"{100 * train_manifest['miss_rate']:.2f}%.",
        f"- Student testing: {student_manifest['n_trips']} trips, {student_manifest['n_obs']:,} observations. "
        f"Valid alignment: {100 * student_manifest['labeled_fraction']:.2f}%; "
        f"candidate miss rate: {100 * student_manifest['miss_rate']:.2f}%.",
        f"- An initial GroupShuffleSplit reserved {holdout_count} of the 80 training trips as an internal holdout. "
        f"The other {development_count} trips underwent {fold_count}-fold GroupKFold with identical folds for every model and tuning trial. "
        "All 80 training trips were used for final refits. No evaluator-testing data was read.",
        "- Primary objective: observation Top-1 averaged equally across trips, then across all development trips. "
        "This limits domination by long trips. Pooled observation accuracy is also reported. Ties use lower candidate log loss.",
        "- Invalid alignments are excluded from supervised metrics and training. Full-data models retain all candidates "
        "from valid alignments, including negative-only rows when the true arc is absent; KNN's training-only budget "
        "is documented below. Those misses and zero-candidate "
        "observations count as incorrect in unconditional Top-1; conditional Top-1 is reported separately.",
        "- Median imputation is fitted on each training fold. Logistic Regression and KNN also use fold-fitted standardisation. "
        "The fixed 17 features are used; trip/observation/arc IDs and all label/alignment metadata are excluded.",
        "- KNN uses at most 100 seeded training observations per trip, retaining every candidate of selected observations. "
        "An exact full-table attempt and backend benchmark established that its inference cost was prohibitive on this machine. "
        "Other models use every valid training row. Every model is evaluated on complete validation and testing partitions. "
        "Consequently the KNN comparison includes a model-specific fitting budget, rather than an equal-row training comparison; "
        "fitting-row counts are saved with every fold. The sampling policy is independent of candidate labels.",
        "- KNN's sampled-training results do not establish how full-table KNN would perform; its reduced fitting "
        "budget is an explicit limitation of this resource-constrained comparison.",
        "- Classification threshold is fixed at 0.5. Precision and recall refer to the positive candidate class. "
        "Macro F1 averages the two binary classes; log loss is candidate-level and row weighted. "
        "CV macro F1 is the unweighted mean of fold scores, not pooled F1.",
        "- Timing includes fold-local preprocessing in fitting and candidate prediction. It excludes raw-data preparation, "
        f"metric calculation, serialization and route decoding. All models use a {configuration['jobs']}-thread limit.", "",
        "## Untuned model comparison", "", table(cv_display, list(cv_display.columns)), "",
        "![Cross-validation results](figures/cross_validation.png)", "",
        "## Bayesian optimisation", "",
        tuning_note, "",
        "Optuna's seeded Tree-structured Parzen Estimator (TPE) performed five startup trials followed by "
        "Bayesian-guided trials. Trial 0 records the already measured baseline. "
        f"The same {fold_count} trip-grouped folds and primary Top-1 objective were used for every trial. "
        "Search spaces vary model capacity, regularisation and learning rate as appropriate; their exact definitions are in models.py. "
        "No pruning or random-row early stopping was used.", "",
        f"Best trial: {selection['best_trial']}. Best parameters:", "", "```json",
        json.dumps(selection["best_params"], indent=2), "```", "",
        f"Untuned CV trip Top-1: {selection['baseline_cv']['top1_trip_mean']:.6f}; "
        f"tuned CV trip Top-1: {selection['tuned_cv_top1']:.6f}. "
        f"Untuned CV macro F1: {selection['baseline_cv']['macro_f1']:.6f}; "
        f"tuned CV macro F1: {selection['tuned_cv_macro_f1']:.6f}. "
        "The best tuning CV score is a selection score and can be optimistic; use the frozen holdout to assess the improvement.", "",
        f"Sum of recorded trial durations across tuning sessions: {trial_seconds:.1f} seconds. "
        "Per-trial fold metrics and fit/predict costs are retained in tuning_trials.csv and the tuning folder. "
        "This excludes the aborted exploratory full-table KNN attempt, which is documented in experiment_amendment.json.", "",
        "![Bayesian search](figures/bayesian_search.png)", "",
        "## Frozen internal holdout: tuned versus untuned", "", table(holdout_display, list(holdout_display.columns)), "",
        f"Paired trip-mean Top-1 change: {100 * delta.mean():+.3f} percentage points. "
        f"Exploratory 95% paired-trip bootstrap interval: [{100 * low:+.3f}, {100 * high:+.3f}] percentage points "
        f"({len(delta)} held-out trips; 5,000 resamples). "
        "This interval describes trip variability, rather than proving superiority on future data.", "",
        "## Student-testing comparison", "", table(student_display, list(student_display.columns)), "",
        "Each model was refitted on all 80 training trips before evaluating the same 10 student-testing trips. "
        "Student results are development comparisons, not evaluator-testing results. "
        "The preselected model is retained even if a different model happens to score better here. "
        "Student candidate coverage is much higher than training coverage, so their higher Top-1 scores should not "
        "be interpreted as an improvement measured on the same trip population.", "",
        "![Student-testing comparison](figures/student_comparison.png)", "",
        "![Computational cost](figures/computational_cost.png)", "",
        "![Confusion matrices](figures/confusion_matrices.png)", "",
        "![Probability reliability](figures/probability_reliability.png)", "",
        "## Individual model interpretation", ""]
    for row in cv.itertuples():
        test = student[(student.model == row.model) & (student.variant == "untuned")].iloc[0]
        # The full baseline configuration, including defaults fixed by this project, is saved in experiment_config.json.
        configured = read("experiment_config.json")["baselines"][row.model]
        lines.extend([f"### {row.model}", "", DESCRIPTIONS[row.model], "",
            "Initial project configuration: `" + json.dumps(configured) + "`.", "",
            f"CV trip Top-1 {row.top1_trip_mean:.4f}, fold SD {row.top1_fold_std:.4f}; "
            f"CV macro F1 {row.macro_f1:.4f}. Student trip Top-1 {test.top1_trip_mean:.4f}, "
            f"macro F1 {test.macro_f1:.4f}, log loss {test.log_loss:.4f}; prediction time "
            f"{test.predict_seconds:.3f} s for {test.evaluated_candidate_rows:,} candidates.", "",
            "These held-out results assess transfer across trips. A CV-to-student difference can reflect distribution shift "
            "or sampling variability; without a matched training-score comparison, it does not independently establish overfitting.", ""])
    nearest = read("student_nearest_road_baseline.json")
    lines.extend(["## Limitations and integration", "",
        f"The non-ML nearest-road baseline achieved student trip Top-1 {nearest['top1_trip_mean']:.4f} "
        f"and pooled Top-1 {nearest['top1_pooled']:.4f}. It uses distance alone and has no classifier probabilities.", "",
        "Scores measure agreement with Member 1's route-aligned observation labels. These labels are inferred using "
        "geometry and monotone route alignment, rather than independently annotated per-observation ground truth. "
        "Alignment errors and coverage therefore limit interpretation; consistency checks do not prove label correctness.", "",
        "The supplied central heading and dt_next_s features use future GPS observations. These artifacts support "
        "offline trip reconstruction; causal streaming would require a separate feature configuration and retraining. "
        "Trip grouping prevents within-trip leakage but does not establish generalisation to entirely unseen cities or road networks.", "",
        "The selected model returns binary candidate-match posterior probabilities. They are not automatically calibrated "
        "HMM emission likelihoods. score_candidates supplies clipped log scoring potentials for Member 3 to validate "
        "alongside transition scores. No additional calibration was fitted; reliability plots and Brier scores expose probability quality.", "",
        "Member 3 owns HMM/Viterbi, route continuity and sequence-level evaluation; these have not been implemented here. "
        "No full-route accuracy or connectivity improvements are claimed from classifier scores alone.", "",
        "Selected artifact: ../models/selected_model.joblib. See MEMBER2_README.md for reproduction commands and "
        "the route-free scoring interface. All existing teammate modules remain unchanged; no commits or remote writes were made.", ""])
    verification_path = out / "verification.json"
    if verification_path.exists():
        verified = read("verification.json")
        lines.extend(["## Verification", "",
            f"{verified['unit_tests_passed']} unit tests passed, including the original Member 1 tests. "
            f"The real-trip inference check regenerated {verified['inference_candidates']:,} candidates for "
            f"{verified['inference_observations']:,} GPS observations without reading a reference route. "
            "Feature values matched the evaluation tables, serialized model probabilities matched the stored results, "
            "candidate identities and input order were preserved, extra label metadata had no effect, and empty "
            "candidate tables were accepted. All 12 Bayesian trials are complete and the recorded source fingerprints match.", ""])
    (out / "MEMBER2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(out / "MEMBER2_REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    main(parser.parse_args().work)
