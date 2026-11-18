"""Feature extraction contract.

Input: normalized sequence plus candidate spans. Output: fixed ordered feature_names
and exactly one finite numeric vector of matching dimension per candidate, keyed by
candidate_id. Feature names/order/version must remain stable for a trained model.
Implement FeatureExtractor.extract; do not call the model here.
"""
from typing import Protocol
from app.contracts import FeatureRequest, FeatureResult


class PathSnapFeatureExtractionAdapter:
    """Extract route-independent GPS and candidate features in stable column order."""
    name = "pathsnap-observation-candidate-features"
    version = "member1-import-v1"

    def extract_for_track(self, net, candidates, xy, timestamps, feature_config):
        from app.pathsnap.preprocessing.features import (
            FEATURE_COLUMNS, build_candidate_features, observation_features,
        )

        observations = observation_features(xy, timestamps, feature_config)
        candidate_features = build_candidate_features(net, candidates, observations)
        return candidate_features, observations, FEATURE_COLUMNS


class FeatureExtractor(Protocol):
    name: str
    version: str

    def extract(self, request: FeatureRequest) -> FeatureResult: ...
