"""Typed Member 2 bridge. Probability semantics remain candidate-match posterior."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_complex_dtype
from sklearn.ensemble import RandomForestClassifier

from app.adapters.ml_model import PathSnapMLScorer
from app.pathsnap.interfaces import CANDIDATE_KEY_COLUMNS, PathSnapModelRequest, PathSnapModelResult
from app.pathsnap.preprocessing.features import FEATURE_COLUMNS as BACKEND_FEATURE_COLUMNS

# Resolve the original package from the shared checkout, never a user-specific path.
original_package = Path(__file__).resolve().parents[3] / "pathsnap"
if (original_package / "pathsnap/ml/predict.py").is_file():
    sys.path.insert(0, str(original_package))
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.ml.predict import score_candidates as original_score_candidates

if tuple(FEATURE_COLUMNS) != tuple(BACKEND_FEATURE_COLUMNS):
    raise ImportError("Backend features differ from the original Member 2 feature schema")


def _validate_bundle(bundle):
    if not isinstance(bundle, dict) or bundle.get("model") != "random_forest" or bundle.get("variant") != "untuned":
        raise ValueError("Artifact must be the selected untuned Random Forest bundle")
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Artifact feature ordering is incompatible")
    pipeline = bundle.get("pipeline")
    classifier = getattr(pipeline, "named_steps", {}).get("classifier")
    if not isinstance(classifier, RandomForestClassifier):
        raise ValueError("Artifact classifier is not RandomForestClassifier")
    if not hasattr(classifier, "classes_") or set(classifier.classes_) != {0, 1} or classifier.n_features_in_ != 17:
        raise ValueError("Expected a fitted binary model with 17 features and classes 0/1")
    if "imputer" not in pipeline.named_steps or not hasattr(pipeline.named_steps["imputer"], "statistics_"):
        raise ValueError("Fitted preprocessing is missing")
    for package in ("scikit-learn", "numpy", "scipy", "joblib"):
        expected = bundle.get("versions", {}).get(package)
        if expected and importlib.metadata.version(package) != expected:
            raise ValueError(f"Incompatible {package}: artifact requires {expected}")


class Member2Scorer(PathSnapMLScorer):
    name = "random_forest"

    def __init__(self, bundle: dict, version: str = "member2-rf-untuned-v1"):
        _validate_bundle(bundle)
        self._bundle = bundle
        self.version = version

    def score_candidates(self, request: PathSnapModelRequest) -> PathSnapModelResult:
        if not isinstance(request, PathSnapModelRequest):
            raise TypeError("Expected PathSnapModelRequest")
        if request.feature_columns != tuple(FEATURE_COLUMNS):
            raise ValueError("feature_columns must exactly match the ordered 17-feature contract")
        frame = request.candidate_features
        if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
            raise ValueError("candidate_features must be a DataFrame with unique columns")
        required = list(CANDIDATE_KEY_COLUMNS) + FEATURE_COLUMNS
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing candidate columns: {sorted(missing)}")
        if frame[list(CANDIDATE_KEY_COLUMNS)].isna().any().any():
            raise ValueError("Candidate identity keys cannot be null")
        if not frame.trip_id.eq(request.trip_id).all():
            raise ValueError("Candidate trip_id differs from request.trip_id")
        for column in FEATURE_COLUMNS:
            if not is_numeric_dtype(frame[column]) or is_complex_dtype(frame[column]):
                raise ValueError(f"Feature {column} must be real numeric; NaN is allowed")
        for column in ("obs_idx", "arc_idx", "arc_id"):
            try:
                values = frame[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid identity column: {column}") from exc
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{column} must contain finite integer identities")
        if (frame.obs_idx < 0).any() or (frame.arc_idx < 0).any():
            raise ValueError("Internal observation/arc indices cannot be negative")
        scores = original_score_candidates(self._bundle, frame)
        probability = scores.match_probability.to_numpy(dtype=np.float64)
        if len(probability) != len(frame) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Model produced invalid candidate probabilities")
        result = frame.copy(deep=True)
        result["emission_probability"] = probability
        return PathSnapModelResult(candidate_scores=result)


def load_member2_scorer(artifact_dir: Path) -> PathSnapMLScorer:
    directory = Path(artifact_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {directory}")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        raise ValueError("Artifact manifest must be a JSON object")
    if manifest:
        if manifest.get("feature_columns") != FEATURE_COLUMNS or manifest.get("positive_class") != 1:
            raise ValueError("Manifest feature schema or positive class is incompatible")
        if manifest.get("model_name") != "random_forest" or manifest.get("variant") != "untuned":
            raise ValueError("Manifest does not describe the selected untuned Random Forest")
        filename = manifest.get("artifact_file")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Manifest must name artifact_file")
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Artifact filename escapes artifact_dir")
        paths = [path]
    else:
        paths = sorted(directory.glob("*.joblib"))
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError("Selected model artifact is missing")
    valid, errors = [], []
    for path in paths:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if manifest.get("sha256") and digest != manifest["sha256"]:
                raise ValueError("Artifact checksum mismatch")
            bundle = joblib.load(path)
            _validate_bundle(bundle)
            valid.append((digest, bundle))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    if not valid:
        raise ValueError("No compatible selected Random Forest artifact: " + "; ".join(errors))
    if len({digest for digest, _ in valid}) != 1:
        raise ValueError("Ambiguous Random Forest artifacts; provide an explicit manifest")
    return Member2Scorer(valid[0][1], manifest.get("model_version", "member2-rf-untuned-v1"))
