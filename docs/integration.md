# Adapter integration guide

Each component is an independent implementation of a `typing.Protocol` in `backend/app/adapters/`. Keep dependencies for a component inside its module/package and translate its native inputs/outputs at the adapter edge. Never return synthetic values from a production adapter to hide missing upstream work.

## Stable stage contracts

| Stage | Interface | Input | Output | Required checks |
|---|---|---|---|---|
| Data preprocessing | `Preprocessor.preprocess(PreprocessRequest) -> PreprocessResult` | raw sequence, declared alphabet | normalized sequence, source positions, warnings | non-empty; positions count equals output length; retain source mapping |
| Candidate generation | `CandidateGenerator.generate(CandidateRequest) -> CandidateResult` | normalized sequence and positions | unique IDs; half-open spans; candidate kind | `0 <= start < end <= sequence length`; deterministic order |
| Feature extraction | `FeatureExtractor.extract(FeatureRequest) -> FeatureResult` | sequence and candidates | ordered feature names and candidate keyed numeric vectors | finite values; exact dimensions; document feature order/version |
| ML model | `SequenceModel.predict(ModelRequest) -> ModelResult` | feature names and vectors | model identity, ordered labels, per-symbol score rows | one row per sequence symbol; one probability per label; each score in `[0,1]` |
| HMM/Viterbi | `SequenceDecoder.decode(DecodeRequest) -> DecodeResult` | score matrix and labels | decoded label path and log probability | path length equals symbols; labels known; finite log probability |

`Contract` models forbid unknown fields. Sequence input is whitespace-stripped and uppercased. Coordinates are zero-based and end-exclusive. Input size is capped at 100,000 symbols; candidates and vectors are capped at 10,000, feature width at 4,096, and labels at 256. Expand limits only with explicit resource sizing and API review.

## Handoff procedure

1. Implement the protocol and expose stable `name` and `version` values.
2. Add isolated tests using fixtures whose names and values explicitly say `fixture`; test malformed shapes and edge cases as well as a valid sample.
3. Register the adapter in the deployment's pipeline factory. The application calls stages in the order shown above.
4. Keep `ModelResult.scores` aligned to sequence positions. If the model scores candidates instead, add a documented aggregation/expansion step in that adapter; do not change endpoint semantics silently.
5. Validate the response and report dependency/configuration failures clearly. Do not send sequence payloads to logs.
6. Run backend unit tests and the frontend build before handoff.

`Pipeline` validates cross-stage correspondence and maps missing integrations to HTTP 503 (`adapter_not_configured`). A malformed adapter result is HTTP 502 (`adapter_contract_violation`). FastAPI validates malformed client requests as HTTP 422. Production adapter registration is intentionally not guessed: deployment wiring must import the real implementations and pass them into `create_app(pipeline=...)`.

## Testing fixtures

`backend/app/fixtures.py` contains deterministic synthetic implementations solely for demonstrating UI/API transport. They generate `FIXTURE_A`/`FIXTURE_B`, constant scores, and synthetic model metadata. The response has `is_fixture: true`, a warning, and `TEST FIXTURE — NOT A PREDICTION`. Never use these for model evaluation, demos presented as scientific predictions, or customer traffic. The environment switch is rejected in production.
