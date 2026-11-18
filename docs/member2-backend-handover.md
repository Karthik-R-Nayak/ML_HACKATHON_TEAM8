# Member 2 backend handover: completed local scoring adapter

## Status

The actual Member 4 contracts were supplied in `important-do-not-wait-for-other.zip`.
Their backend, frontend, tests and documentation were imported without their virtual
environment, Git metadata or caches. No supplied files were overwritten. The typed
Member 2 adapter is implemented and the original model was not retrained.

Member 3's interface is available; its decoder implementation is not. Compatibility
is verified through an actual PathSnapDecodeRequest containing scored real-trip
candidates, observations and road topology. No HMM/Viterbi execution or reconstructed
route is claimed. See [Laptop 1/Laptop 2 demo instructions](laptop1-demo.md).

## Selected model

- Algorithm/variant: untuned Random Forest.
- Local artifact, relative to repository root:
  `../experiments/member2/models/selected_model.joblib`.
- SHA-256: `b5533a8709b43a90e06376991b659edf848d188079cebcccee502ba203476eb4`.
- Class labels: `[0, 1]`; the positive match label is `1`. The original scorer
  locates its actual position in `pipeline.classes_` rather than assuming an index.
- Pipeline: fitted `SimpleImputer(strategy="median", keep_empty_features=True)`
  followed by `RandomForestClassifier`. No scaler is used for this selected model.
- Project parameters: 120 trees, maximum depth 18, minimum leaf size 10,
  `max_features="sqrt"`, seed 42 and four inference jobs.
- Metadata: the artifact bundle includes feature order, versions, pipeline settings,
  model name/variant, parameters and training trip IDs. The accompanying
  `model_metadata.json` records selection results; its `best_params` describe the
  separately tuned boosting candidate, not Random Forest's actual parameters.

The original model is ignored by Git and must be transferred separately. The local
workspace's `PathSnap_Member2_Backend_Model.zip` contains this exact model, its
metadata/checksum, ordered feature schema and minimal inference dependencies.
This ZIP is a local handoff artifact; it has not been uploaded anywhere.

## Exact ordered feature schema

```text
dist_m
proj_frac
heading_diff_deg
gps_speed_mps
gps_bearing_deg
arc_bearing_deg
arc_length_m
from_in_deg
from_out_deg
to_in_deg
to_out_deg
dt_prev_s
dt_next_s
gps_heading_valid
dist_rank
dist_minus_min_m
n_candidates
```

The authoritative definition is `pathsnap.features.FEATURE_COLUMNS`. The original
scorer selects these columns in that order, converts them to float32, and invokes
the saved fitted pipeline. Numeric NaNs use its fitted imputation statistics;
infinities are rejected. There are no categorical encoders or independently fitted
preprocessing steps. Input column order itself may differ because selection is explicit.

## Available original interface and example

From the repository root, in the existing project environment:

```python
from pathlib import Path
import sys
sys.path.insert(0, "pathsnap")
from pathsnap.ml.predict import score_candidates

# candidate_df is actual Member 1 output, generated without a reference route.
scores = score_candidates(
    Path("../experiments/member2/models/selected_model.joblib"),
    candidate_df,
)
```

The existing function signature is `score_candidates(artifact, candidates)`.
Required columns are `trip_id`, `obs_idx`, `arc_id`, plus the 17 ordered features.
It preserves input row order/index and returns those three IDs together with
`match_probability` and `log_match_score`. Empty input is supported when the schema
is present; duplicate identity triples, missing required columns and infinite
features raise ValueError.

## Actual typed adapter

Implementation: `backend/app/pathsnap/member2_scorer.py`.
Exports: `Member2Scorer` and `load_member2_scorer(artifact_dir: Path) -> PathSnapMLScorer`.
It imports Member 4's actual protocol and dataclasses rather than defining replacements.

From the repository root:

```python
from pathlib import Path
import sys
sys.path[:0] = ["backend", "pathsnap"]
from app.pathsnap.member2_scorer import load_member2_scorer
from app.pathsnap.pipeline import PATHSNAP_PIPELINE
from app.pathsnap.interfaces import PathSnapModelRequest, PathSnapDecodeRequest

scorer = load_member2_scorer(Path("backend/artifacts/member2"))
# payload is the actual app.contracts.PathSnapPreprocessRequest; omit route labels.
prepared = PATHSNAP_PIPELINE.prepare_payload(payload)
scored = scorer.score_candidates(PathSnapModelRequest(
    trip_id=prepared.trip_id,
    candidate_features=prepared.candidate_features,
    feature_columns=prepared.candidate_feature_columns,
))
# This constructs the real downstream interface; it does not run a missing decoder.
decode_request = PathSnapDecodeRequest(
    trip_id=prepared.trip_id,
    candidate_scores=scored.candidate_scores,
    observations=prepared.observations,
    network=prepared.network,
)
```

The output is the actual PathSnapModelResult. Its candidate_scores copies every input
column and index, preserving trip_id/obs_idx/arc_idx/arc_id and extra backend IDs,
then adds float64 emission_probability. No rows are dropped or reordered. Empty
batches preserve schema. Missing features/keys, duplicate candidate triples, invalid
trip identity, nonnumeric features and infinities fail explicitly; NaNs use the saved
imputer. Request.feature_columns must match the exact feature tuple. DataFrame
columns may be supplied in any order; the original scorer enforces model order.

The loader uses manifest.json's artifact_file, verifies its checksum and schema,
and checks the fitted Random Forest, bundled preprocessing and dependency versions.
If no manifest is provided, compatible .joblib bundles are discovered by content;
ambiguous different Random Forest artifacts are rejected, rather than silently selected.
There is no algorithm fallback. A renamed valid file is covered by testing.

Local ready-to-transfer directory: backend/artifacts/member2, containing the selected
model, manifest.json and selection_metadata.json. Do not treat selection_metadata's
boosting best_params as Random Forest parameters; the actual bundle is authoritative.
The original model file and its checksum remain unchanged.

## Probability meaning

Scores are binary candidate-match posteriors. They are not automatically calibrated
generative HMM emission likelihoods. No calibration or decoder is silently added.
Member 3 owns their use with transition potentials and route reconstruction.
The current feature configuration uses central headings and `dt_next_s`, so it is
intended for offline trajectories rather than causal streaming.

## Dependencies and checks

The full existing requirements are `pathsnap/requirements-member2.txt`.
The transfer bundle additionally lists the exact installed versions of the minimal
inference dependencies: Python, numpy, pandas, scipy, scikit-learn, joblib and
threadpoolctl. Model metadata can be checked before loading on Member 4's machine.

Existing checks, run from the repository root in PowerShell:

```powershell
& "..\.venv\Scripts\python.exe" -m pytest pathsnap/tests -q --basetemp "../.runtime-temp/pytest-backend-preparation"
& "..\.venv\Scripts\python.exe" pathsnap/scripts/validate_ml_artifacts.py --work "../experiments/member2" --report "../experiments/member2/validation"
```

The historical completed validation contains 37 passing pytest tests, 13 real-data/
artifact checks, and nine CLI checks; all six saved model configurations reproduced
their student-test probabilities and metrics. This does **not** validate a backend
adapter that has not been implemented, nor an unavailable HMM integration.
New preparation-only checks and their exact scope are saved separately under
`../experiments/member2/backend_handover/`.

## Verification and remaining boundary

See `../experiments/member2/backend_handover/backend-integration-tests.xml` for the
actual final test count. Tests include artifact discovery, missing/invalid/wrong
models, manifest order/class/path/checksum checks, NaNs and column ordering,
identity/index preservation, repeated inference within numerical tolerance,
real saved-probability reproduction, and the true preprocessingÃ¢â€ â€™modelÃ¢â€ â€™decoder-request
composition. Existing Member 1, Member 2 and supplied Member 4 tests are included.

The standalone Laptop 1 API uses these actual types and the new scorer. It is launched
with `python -m uvicorn laptop1_demo.server:app --host 0.0.0.0 --port 8000` from the
repository root. Member 4's supplied app/main.py remains unchanged and preprocessing-only;
starting app.main:app alone does not register the scorer. The demo provides additive
ML-scoring routes, with honest HMM-not-connected capability reporting.

No real Member 3 decoder exists in the ZIP. Member 4 must register it when supplied,
then run full route reconstruction/evaluation. Physical Laptop 2 connectivity still
requires a request from that machine; same-host LAN probes cannot establish that.
No evaluator labels were inspected and no model was retrained.

All changes remain local and uncommitted. No remote branches, pull requests, merges,
commits, uploads or pushes were made during this integration request.


## Final measured validation

The combined suite completed with **84 passed, 10 subtests passed**, and two
non-blocking dependency/pandas warnings. The typed adapter was additionally run
on all ten real student trips, regenerating backend features and comparing all
366,633 candidate probabilities against the original saved outputs. The maximum
absolute difference was 4.440892098500626e-16; every output remained float64.
All 41 imported Member 4 files match their original ZIP hashes.

PowerShell live HTTP checks passed for health/trip listing at both 127.0.0.1:8000
and 10.0.13.39:8000, and a real-trip request produced 2,503 observations and 23,160
scored candidates. Physical Laptop 2 reachability is still pending. See
laptop1-demo.md for the client and firewall diagnostics.

For Member 4 inference-only setup, install backend/requirements-member2.txt in a
Python 3.13 environment. Keep both the supplied backend package and the original
pathsnap/pathsnap/ source package available; do not copy the ZIP's Python 3.15 .venv.

Additional local files include backend/app/pathsnap/member2_scorer.py,
backend/tests/test_member2_scorer.py, backend/requirements-member2.txt,
backend/artifacts/member2/manifest.json and selection_metadata.json, and the unchanged
selected model binary (ignored by Git). The laptop1_demo/ composition and client
are separate from Member 4's untouched source. Imported files are inventoried by
member4-import-manifest.json under the experiment's backend_handover directory.

Member 2 backend integration is complete locally. No changes have been committed or pushed to GitHub.


The supplied backend's training_matrix helper excludes candidate-miss groups,
whereas the historical Member 2 fit retained valid all-negative candidate rows.
This integration performs inference only: it neither switches training policies nor
rebuilds the model. Backend and original inference features were checked against
the same prepared real student data on every trip, and matched after the original
float32 input conversion. Any future retraining should explicitly reconcile that
training-policy difference rather than silently reusing historical metrics.
