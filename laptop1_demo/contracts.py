"""Stable, validated data contracts shared by all pipeline stages."""
from typing import Literal
from math import isfinite
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SequenceInput(Contract):
    sequence: str = Field(min_length=1, max_length=100_000)
    sequence_id: str | None = Field(default=None, max_length=128)
    alphabet: Literal["any", "dna", "rna", "protein"] = "any"

    @field_validator("sequence")
    @classmethod
    def normalize_sequence(cls, value: str) -> str:
        normalized = "".join(value.split()).upper()
        if not normalized:
            raise ValueError("sequence must contain at least one non-whitespace symbol")
        return normalized


class PreprocessRequest(Contract):
    sequence: str
    alphabet: Literal["any", "dna", "rna", "protein"]


class PreprocessResult(Contract):
    sequence: str = Field(min_length=1)
    positions: list[int]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def positions_match_sequence(self):
        if len(self.positions) != len(self.sequence):
            raise ValueError("positions length must match sequence length")
        return self


class Candidate(Contract):
    candidate_id: str = Field(min_length=1, max_length=128)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    kind: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end <= self.start:
            raise ValueError("candidate end must be greater than start")
        return self


class CandidateRequest(Contract):
    sequence: str
    positions: list[int]


class CandidateResult(Contract):
    candidates: list[Candidate] = Field(max_length=10_000)


class FeatureRequest(Contract):
    sequence: str
    candidates: list[Candidate]


class FeatureVector(Contract):
    candidate_id: str
    values: list[float] = Field(min_length=1, max_length=4096)

    @field_validator("values")
    @classmethod
    def finite_values(cls, values: list[float]) -> list[float]:
        if any(not isfinite(value) for value in values):
            raise ValueError("feature values must be finite")
        return values


class FeatureResult(Contract):
    feature_names: list[str] = Field(min_length=1, max_length=4096)
    vectors: list[FeatureVector] = Field(max_length=10_000)

    @model_validator(mode="after")
    def dimensions_match(self):
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if any(len(vector.values) != len(self.feature_names) for vector in self.vectors):
            raise ValueError("every vector must match feature_names dimensions")
        return self


class ModelRequest(Contract):
    features: FeatureResult


class ModelResult(Contract):
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    labels: list[str] = Field(min_length=1, max_length=256)
    scores: list[list[float]] = Field(min_length=1)

    @field_validator("labels")
    @classmethod
    def unique_labels(cls, labels: list[str]) -> list[str]:
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")
        return labels

    @model_validator(mode="after")
    def scores_match_labels(self):
        if any(len(row) != len(self.labels) for row in self.scores):
            raise ValueError("each score row must contain one value per label")
        if any(not isfinite(score) or not 0 <= score <= 1 for row in self.scores for score in row):
            raise ValueError("scores must be probabilities in [0, 1]")
        return self


class DecodeRequest(Contract):
    scores: list[list[float]] = Field(min_length=1)
    labels: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dimensions(self):
        if any(len(row) != len(self.labels) for row in self.scores):
            raise ValueError("score dimensions must match labels")
        return self


class DecodeResult(Contract):
    path: list[str] = Field(min_length=1)
    log_probability: float

    @field_validator("log_probability")
    @classmethod
    def finite_log_probability(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("log_probability must be finite")
        return value


class AnalysisResponse(Contract):
    sequence_id: str | None
    sequence_length: int
    normalized_sequence: str
    candidates: list[Candidate]
    feature_names: list[str]
    model_name: str
    model_version: str
    labels: list[str]
    scores: list[list[float]]
    decoded_path: list[str]
    warnings: list[str]
    is_fixture: bool = False
    notice: str | None = None


class GPSObservation(Contract):
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)
    t: float = Field(description="Unix timestamp in seconds; observations should be chronological")
    orig_row: int | None = Field(default=None, ge=0,
        description="Optional input row identifier retained when observations have been reordered")

    @field_validator("t")
    @classmethod
    def finite_timestamp(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("timestamp must be finite")
        return value


class RoadNodeInput(Contract):
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)


class RoadArcInput(Contract):
    from_node: int
    to_node: int


class PathSnapPreprocessRequest(Contract):
    trip_id: str = Field(min_length=1, max_length=128)
    observations: list[GPSObservation] = Field(min_length=1, max_length=100_000)
    nodes: list[RoadNodeInput] = Field(min_length=2, max_length=1_000_000)
    arcs: list[RoadArcInput] = Field(min_length=1, max_length=2_000_000)
    node_id_base: int = 0
    arc_id_base: int = 0
    route_arc_ids: list[int] | None = Field(default=None, max_length=2_000_000,
        description="Optional route labels for evaluation/training only; omit for normal inference.")


class PathSnapPreprocessResponse(Contract):
    trip_id: str
    sequence_length: int
    candidate_count: int
    candidate_feature_columns: list[str]
    observation_feature_columns: list[str]
    candidates: list[dict]
    observations: list[dict]
    labeling_performed: bool
    route_arc_ids: list[int] | None
    network: dict
    feature_config: dict
    is_prediction: bool
    notice: str
