"""Typed transport records for the future PathSnap model and route decoder.

These are contracts only. They deliberately contain no scorer, model weights,
transition parameters, or predicted values.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from app.pathsnap.preprocessing.network import RoadNetwork


CANDIDATE_KEY_COLUMNS = ("trip_id", "obs_idx", "arc_idx", "arc_id")
EMISSION_COLUMN = "emission_probability"
DECODED_PATH_COLUMNS = ("trip_id", "obs_idx", "arc_idx", "arc_id", "decode_status")


@dataclass(frozen=True)
class PathSnapModelRequest:
    """Rows from Member 1 candidate features plus the fixed feature ordering."""

    trip_id: str
    candidate_features: pd.DataFrame
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class PathSnapPreparedTrip:
    """Internal Member 1 output before JSON serialization.

    Keep the network object and DataFrames inside the service process for the
    Member 2/3 adapters; do not serialize the road graph into the HTTP response.
    """

    trip_id: str
    candidate_features: pd.DataFrame
    observations: pd.DataFrame
    network: "RoadNetwork"
    candidate_feature_columns: tuple[str, ...]
    observation_feature_columns: tuple[str, ...]
    labeling_performed: bool
    route_arc_ids: tuple[int, ...] | None

    @property
    def sequence_length(self) -> int:
        return len(self.observations)

    def to_preprocess_response(self) -> dict:
        """Serialize preprocessing only, preserving the current HTTP contract."""
        import json

        return {
            "trip_id": self.trip_id,
            "sequence_length": self.sequence_length,
            "candidate_count": len(self.candidate_features),
            "candidate_feature_columns": list(self.candidate_feature_columns),
            "observation_feature_columns": list(self.observation_feature_columns),
            "candidates": json.loads(self.candidate_features.to_json(orient="records", double_precision=15)),
            "observations": json.loads(self.observations.to_json(orient="records", double_precision=15)),
            "labeling_performed": self.labeling_performed,
            "route_arc_ids": None if self.route_arc_ids is None else list(self.route_arc_ids),
            "network": {"node_count": len(self.network.node_xy), "arc_count": self.network.n_arcs,
                        "arc_id_base": self.network.arc_base},
            "feature_config": {"heading_mode": "central"},
            "is_prediction": False,
            "notice": "Preprocessing features only; no ML prediction or route reconstruction was performed.",
        }


@dataclass(frozen=True)
class PathSnapModelResult:
    """One positive-class probability per candidate, in the same row order.

    ``candidate_scores`` contains the four columns in CANDIDATE_KEY_COLUMNS plus
    ``emission_probability``. The keys must be unchanged from the request.
    """

    candidate_scores: pd.DataFrame


@dataclass(frozen=True)
class PathSnapDecodeRequest:
    """Scored candidate states, observation sequence and source road topology."""

    trip_id: str
    candidate_scores: pd.DataFrame
    observations: pd.DataFrame
    network: "RoadNetwork"
    score_column: str = EMISSION_COLUMN


@dataclass(frozen=True)
class PathSnapDecodeResult:
    """One selected state per input observation, plus the trip path log score.

    ``decoded_path`` contains DECODED_PATH_COLUMNS. ``arc_idx`` and ``arc_id``
    are nullable for ``no_candidates`` or ``no_valid_path`` rows. The aggregate
    log score is finite when a valid path exists, otherwise None.
    """

    decoded_path: pd.DataFrame
    path_log_probability: float | None
