# Member 2: five-model comparison and Bayesian optimisation

This adds candidate-match classification on top of Member 1's existing pipeline.
It does not modify feature generation, route alignment, candidate generation or teammates' existing source files.

## Run

From the repository root, with Python 3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r pathsnap/requirements-member2.txt
python -m pytest pathsnap/tests -q --basetemp ../pytest-member2-new
python -u pathsnap/scripts/train_models.py --archive "C:/path/to/Student folder.zip" --work ../experiments/member2 --folds 3 --trials 12 --jobs 4 --tune-model hist_gradient_boosting
python pathsnap/scripts/report_ml_results.py --work ../experiments/member2
python pathsnap/scripts/validate_ml_artifacts.py --work ../experiments/member2 --report ../experiments/member2/validation
```

Replace `C:/path/to/Student folder.zip` with your dataset archive location.
Use an unused pytest temporary directory: pytest can clear its explicitly selected base directory.
The actual executed environment for this workspace is `../.venv` relative to the repository root.

The runner also accepts `--phase compare`, `--phase tune`, and `--phase final`.
Run these in that order to inspect intermediate outputs. Completed CV fold outputs
are cached, so comparison can be resumed after interruption. Changing source,
inputs, seed or configuration requires a new work directory. TPE sampler state is
checkpointed after each completed trial for deterministic clean-boundary resumption.
A checkpoint mismatch fails explicitly; the saved trial log is the authoritative record.

## Experiment design

- Models: Logistic Regression, distance-weighted KNN, Decision Tree, Random Forest,
  and HistGradientBoostingClassifier.
- Fixed feature schema: the exact 17 columns exported by `pathsnap.features.FEATURE_COLUMNS`.
- Candidate labels: binary match/non-match; validity comes from Member 1's route alignment.
- Invalid alignments are excluded from fitting and supervised evaluation. Valid
  alignments with absent ground-truth candidates retain their all-negative candidate
  rows in training. KNN alone uses a seeded training-only sample of at most 100 valid
  observations per trip, retaining every candidate for each chosen observation.
  Its exact brute-force neighbour search otherwise has prohibitive full-table cost.
  Other models use all valid training rows. All validation and testing observations
  are retained for every model; the differing KNN fitting budget is reported explicitly.
- Seed 42 reserves 16 training trips as a frozen internal holdout. The remaining 64
  trips use identical three-fold GroupKFold splits for all models and tuning trials.
- Median imputation and, for Logistic Regression/KNN, standardisation are fitted
  within each training fold. All-missing columns retain the fixed feature width.
- Internal random-row early stopping is disabled for histogram boosting.
- Primary model-selection objective: observation Top-1 averaged equally over trips.
  Ties use lower candidate log loss. These choices are fixed before results are read.
- Aligned candidate misses and zero-candidate observations count as Top-1 failures.
  Conditional Top-1 given candidate coverage and pooled observation Top-1 are also reported.
- Candidate accuracy, positive precision/recall, binary macro F1, log loss, Brier score,
  confusion matrices, fit time and candidate-prediction time are recorded. The
  classification threshold is fixed at 0.5; it is not tuned against held-out data.
- The executed promising family receives 12 Optuna TPE trials: baseline trial 0,
  four additional startup trials, then seven Bayesian-guided trials. Every trial uses
  the same CV splits. The baseline CV measurements are reused rather than refitted.
- `--tune-model hist_gradient_boosting` can select a promising family explicitly after
  comparing measured quality and compute cost. Without this option, the highest CV
  Top-1 family is tuned. Final selection compares the tuned candidate with the
  strongest baseline across all five families, using CV Top-1 and the log-loss tie break.
- Chosen parameters and selected tuned/untuned variant are frozen using CV only.
  Both variants of the tuned family are compared on the untouched internal holdout,
  along with the strongest baseline when it is the final selection, then every baseline
  and the tuned candidate are refitted on all 80 training trips and compared on the
  10 student-testing trips. Student testing never enters Bayesian optimisation.
- Evaluator-testing data is neither accepted nor loaded by the runner.

## Search spaces

The selected family determines the space; fixed baseline parameters remain in effect
unless explicitly searched. Exact definitions live in `pathsnap/ml/models.py`.

| Family | Search space |
| --- | --- |
| Logistic Regression | C: log-uniform 0.001–100 |
| KNN | neighbours: odd integers 5–51; weights: uniform/distance; Minkowski p: 1/2 |
| Decision Tree | depth: 5–24; minimum leaf rows: log-integer 5–100; pruning alpha: uniform 0–0.001 |
| Random Forest | trees: 80/120/160/200; depth: 10–26; minimum leaf rows: log-integer 3–60; feature fraction: sqrt/0.5/1.0 |
| Histogram Boosting | learning rate: log-uniform 0.03–0.2; iterations: 80–240 in steps of 40; leaves: 15–63 in steps of 8; minimum leaf rows: log-integer 15–120; L2: log-uniform 0.01–20 |

These ranges test capacity and regularisation while keeping full-dataset CPU experiments
practical. Search spaces are declared before trial results and do not use evaluator data.

## Results and artifacts

The executed work directory is `../experiments/member2` relative to the repository
root. The earlier full-table KNN attempt is retained outside it as an exploratory run;
`results/experiment_amendment.json` documents the switch and reuse of unchanged completed
model measurements. The `--work` directory contains:

- `data/train` and `data/student_test`: validated tables, per-trip audits, missingness,
  original raw files from the respective allowed partition, observation CSVs and
  source/data/configuration fingerprints. `tables.joblib` stores `(candidate_df, observation_df)`.
- `results/splits.json`: explicit trip membership for every split.
- `results/experiment_config.json` and `environment.json`: model configurations,
  code/input hashes, package versions, seed and execution details.
- `results/baseline_cv.csv`, `cv/`: fold metrics, per-trip scores and baseline comparison.
- `results/tuning.sqlite3`, `tuning_trials.csv`, `tuning/`: all Bayesian trial records and fold results.
- `results/selection.json`: frozen selection and best parameters.
- `results/tuning_choice.json`: explicitly frozen tuning family.
- `results/holdout_comparison.csv`, `student_comparison.csv`: measured results.
- `results/student_*_probabilities.npz`: probabilities in the exact row order of
  the candidate table in `data/student_test/tables.joblib`.
- `results/*_predictions.csv.gz`: observation predictions for valid aligned observations;
  includes absent predictions for zero-candidate observations.
- `results/MEMBER2_REPORT.md` and `results/figures/`: report tables and real-result plots.
- `validation/`: pytest XML, saved-model/data checks, cached CLI resume logs and a testing summary.
- `models/selected_model.joblib`: chosen pipeline, preprocessing, feature order,
  pipeline configuration, training trip IDs and package versions.
- `models/*_untuned.joblib`, `models/*_tuned.joblib`: all baseline pipelines and the tuned candidate.

Per-trip CSVs give each trip equal visibility. Global candidate miss rates are
observation weighted. CV macro F1 is the mean of fold macro F1; log loss is weighted
by the number of valid candidate rows. Error bars show fold SD, not confidence intervals.
Model selection scores from tuning are optimistic; the separate holdout is the check
on generalisation before the final all-training-trip refit.

## Interface for Members 3 and 4

From the repository root, add `pathsnap/` to Python's module path:

```python
from pathlib import Path
import sys
sys.path.insert(0, "pathsnap")
from pathsnap.ml.predict import score_candidates

# candidate_df is Member 1's inference output, produced with route_arc_ids=None.
scores = score_candidates(Path("../experiments/member2/models/selected_model.joblib"), candidate_df)
# columns: trip_id, obs_idx, arc_id, match_probability, log_match_score
```

The function preserves input row order, observation identities and external arc IDs.
It requires no routes, labels or alignment metadata. Extra metadata is ignored.
Empty candidate tables return an empty scored table. Member 3 must use the separate
observation table to handle observations with no candidates.

`match_probability` is a binary candidate-match posterior, not a calibrated generative
HMM emission likelihood. `log_match_score` is `log(clip(p, 1e-9, 1))`, an optional
discriminative scoring potential. Member 3 must validate its combination with transition
scores on development data. No calibration or route decoder is silently applied.

The current features use central headings and next-observation time differences;
the saved model supports offline trajectories. Streaming inference needs a separate
causal feature configuration and retraining. Evaluation labels themselves are inferred
from route alignment, so measured classifier scores are agreement with that alignment.
Route continuity, HMM/Viterbi and sequence-level evaluation remain Member 3's work.

## Ownership and repository handling

New implementation source is limited to `pathsnap/ml/`, the four new scripts, this README,
`requirements-member2.txt` and `tests/test_ml.py`. Existing Member 1 source remains unchanged.
Generated experiments and models live outside the repository checkout by default.
Nothing is committed, pushed or written to GitHub by these scripts.

The checked-in [results snapshot](../member2_results/README.md) contains the report,
plots, compact metrics, configurations and validation evidence. Large generated
feature tables and trained model binaries remain outside Git.
