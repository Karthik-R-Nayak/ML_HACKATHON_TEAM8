# PathSnap preprocessing integration

## Imported package

All ten supplied files were copied together to `backend/app/pathsnap/preprocessing/` so their relative imports continue to work:

`labels.py`, `align.py`, `dataset.py`, `config.py`, `candidates.py`, `features.py`, `io.py`, `geo.py`, `index.py`, and `network.py`.

Imports use `app.pathsnap.preprocessing.*`; the nested `io.py` cannot shadow Python's top-level `io` module. The FastAPI bridge calls the imported implementation through `PathSnapPreprocessingAdapter`, `PathSnapCandidateGenerationAdapter`, and `PathSnapFeatureExtractionAdapter`.

## API contract

`POST /v1/pathsnap/preprocess` accepts:

```json
{
  "trip_id": "trip-123",
  "observations": [{"lon": -122.4, "lat": 37.7, "t": 1730000000.0}],
  "nodes": [{"lon": -122.4, "lat": 37.7}, {"lon": -122.399, "lat": 37.7}],
  "arcs": [{"from_node": 0, "to_node": 1}],
  "node_id_base": 0,
  "arc_id_base": 0,
  "route_arc_ids": null
}
```

Timestamps are Unix seconds and observations should be chronological. Node and arc IDs follow the declared bases. Each candidate response retains `arc_id` (source ID), `arc_idx` (internal row index), and `obs_idx` (the GPS observation relationship). Observation rows preserve `gps_lon`, `gps_lat`, and optional input `orig_row` as `source_obs_idx`. `candidate_feature_columns` and `observation_feature_columns` declare stable ordering. The route is optional and used only for label alignment/evaluation. Normal inference sends `route_arc_ids: null`, performs no route-file access, sets `labeling_performed: false`, and reports `is_prediction: false`.

The flow is GPS/network validation → geographic projection and network construction → grid spatial index → candidates → central-heading observation features → candidate features → structured response. No Member 2 model or Member 3 HMM is included.

## Training and dataset isolation

`training_matrix()` groups rows by `(trip_id, obs_idx)` and includes a group only when its labels are valid and at least one candidate is positive. This prevents candidate misses from becoming all-negative training groups. The observation-level `recall_report()` continues to include label-valid misses, including the radius/candidate-cap miss breakdown; no ground-truth candidate is injected.

`discover_dataset_splits(root)` requires separate `train/` and `student_test/` directories and only uses those exact partitions. It optionally discovers an existing `evaluator_test/`, `evaluator-testing/`, or `evaluator_testing/` directory. It never invents trip IDs or draws random test examples from training. Use `run_training_pipeline(root, ...)` for training feature generation; it targets only `root/train/`. Run student/evaluator reports by explicitly passing the corresponding test directory to the general preprocessing function. Evaluator data has no path into `run_training_pipeline`.

The available `Downloads/Student folder.zip` archive was inspected without extracting its contents into the project: it contains 80 `train/` trip directories and 10 `student_test/` trip directories, each with `.track`, `.nodes`, `.arcs`, and `.route` files. No evaluator-test folder was present. The routes in student-testing data are not read by the inference endpoint.

Training and inference use `FeatureConfig(heading_mode="central")`. Preserve `FEATURE_COLUMNS` exactly when a model adapter is added; training and inference must share this feature configuration.

## Integration tests

`backend/tests/test_pathsnap_integration.py` covers imports, network/candidate/feature stages, original arc identity, feature ordering, candidate-miss handling, dataset separation, central heading, route-free operation, and the FastAPI endpoint. Test geometry is synthetic and is labelled in the test helper; it is not a scientific result.

## Member 2 — trained candidate scorer

### Expected files

- Implementation: `backend/app/pathsnap/member2_scorer.py`
- Export: `Member2Scorer` implementing `PathSnapMLScorer`, plus `load_member2_scorer(artifact_dir: Path) -> PathSnapMLScorer`
- Artifact directory: `backend/artifacts/member2/`; include a `manifest.json` naming the artifact file, model/version, positive class, and the exact ordered `feature_columns`. The model file format and required dependency versions must be declared by Member 2; no format or model is assumed here.
- Tests: `backend/tests/test_member2_scorer.py`, using explicit synthetic inputs and checking row/key preservation and scores in `[0,1]`.

### Training input

Use `run_training_pipeline(dataset_root, ...)` to read only `dataset_root/train/`. Pass its candidate table to:

```python
training_matrix(cand_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]
```

The return values are `(X, y, groups)`: `X` has the 17 columns in `FEATURE_COLUMNS`, in that exact order; `y` is integer 0/1 candidate-match labels; `groups` contains trip IDs for group-aware cross-validation. Candidate rows are retained only for valid `(trip_id, obs_idx)` groups with at least one positive. Fit preprocessing/imputation only on training folds and persist it with the model. Do not train, select hyperparameters, or fit transforms on student/evaluator test trips.

### Inference signature and types

The adapter protocol is declared in `backend/app/adapters/ml_model.py`:

```python
score_candidates(request: PathSnapModelRequest) -> PathSnapModelResult
```

`PathSnapModelRequest` is in `backend/app/pathsnap/interfaces.py` and contains:

- `trip_id: str`
- `candidate_features: pandas.DataFrame`, one row per generated candidate, carrying `trip_id`, `obs_idx`, `arc_idx`, `arc_id`, and all ordered feature columns. Feature values are numeric pandas columns; unavailable heading/speed values may be `NaN` as in Member 1's output.
- `feature_columns: tuple[str, ...]`, exactly `tuple(FEATURE_COLUMNS)`.

Return `PathSnapModelResult(candidate_scores: pandas.DataFrame)`. It must have exactly one row per input candidate, in the same order and with unchanged `trip_id`, `obs_idx`, `arc_idx`, and `arc_id` values, plus `emission_probability: float64` in `[0,1]`. Probability is for the positive class “candidate matches route label”. No candidates means an empty result with the same columns. Do not add or synthesize candidates, and do not choose a final path here.

## Member 3 — HMM/Viterbi route decoder

### Expected files

- Implementation: `backend/app/pathsnap/member3_decoder.py`
- Export: `Member3ViterbiDecoder` implementing `PathSnapRouteDecoder`, plus `load_member3_decoder(artifact_dir: Path) -> PathSnapRouteDecoder`
- Artifact directory: `backend/artifacts/member3/`; include a `manifest.json` with decoder/transition parameter version, parameter filenames, and documented start/transition/emission conventions. Member 3 owns the parameter format; no transition values or decoder are supplied here.
- Tests: `backend/tests/test_member3_decoder.py`, with explicit synthetic network/candidate fixtures and edge cases for empty candidate observations and impossible transitions.

### Decode signature and types

The adapter protocol is declared in `backend/app/adapters/hmm_viterbi.py`:

```python
decode_path(request: PathSnapDecodeRequest) -> PathSnapDecodeResult
```

`PathSnapDecodeRequest` contains:

- `trip_id: str`
- `candidate_scores: pandas.DataFrame` from Member 2, with `trip_id`, `obs_idx`, `arc_idx`, `arc_id`, and `emission_probability` (`float64` in `[0,1]`). Rows are candidate states and may be ragged across observations.
- `observations: pandas.DataFrame` with one row per GPS observation, ordered by `obs_idx`; includes observation features, `gps_lon`, `gps_lat`, `gps_timestamp_s`, and optional `source_obs_idx`.
- `network: RoadNetwork`, the in-process topology built by Member 1. Use `arc_idx` to index its internal arrays and preserve `arc_id` as the external/source identifier.
- `score_column: str`, default `"emission_probability"`.

Return `PathSnapDecodeResult(decoded_path: pandas.DataFrame, path_log_probability: float | None)`. `decoded_path` has exactly one row per input observation and these columns: `trip_id: str`, `obs_idx: int64`, nullable `arc_idx: Int64`, nullable `arc_id: Int64`, and `decode_status: str`. Use `decoded`, `no_candidates`, or `no_valid_path`; missing states have null arc IDs. A valid path has a finite aggregate log probability; if no valid path exists, use `None`. The selected arc must be one of the candidates for that observation. Do not use ground-truth route IDs or route files during inference.

## Composition point

`PathSnapPreprocessingPipeline.prepare_payload(payload) -> PathSnapPreparedTrip` is the internal, non-HTTP composition boundary. It preserves Member 1's candidate and observation DataFrames, ordered feature-column tuples, and `RoadNetwork` object without serializing the road graph. Once both members' modules arrive, the runtime can compose:

```python
prepared = PATHSNAP_PIPELINE.prepare_payload(payload)  # route_arc_ids omitted
scored = scorer.score_candidates(PathSnapModelRequest(
    trip_id=prepared.trip_id,
    candidate_features=prepared.candidate_features,
    feature_columns=prepared.candidate_feature_columns,
))
decoded = decoder.decode_path(PathSnapDecodeRequest(
    trip_id=prepared.trip_id,
    candidate_scores=scored.candidate_scores,
    observations=prepared.observations,
    network=prepared.network,
))
```

The existing `/v1/pathsnap/preprocess` endpoint remains preprocessing-only. A prediction/matched-route endpoint should be added only after both real implementations are registered; it must return the decoder's actual output and must never substitute fixture scores or paths.
