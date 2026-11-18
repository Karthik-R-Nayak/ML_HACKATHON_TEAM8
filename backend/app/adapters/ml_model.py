"""ML model adapter contract.

Input: feature names and vectors. Output: versioned model identity, ordered class
labels, and one probability row per sequence position (or candidate, as established
by the integration profile), with values in [0,1]. Current pipeline contract uses
one row per normalized symbol. Return aligned output deterministically. Implement
SequenceModel.predict; keep loading/inference details inside the implementation.
"""
from typing import Protocol
from app.contracts import ModelRequest, ModelResult
from app.pathsnap.interfaces import PathSnapModelRequest, PathSnapModelResult


class SequenceModel(Protocol):
    name: str
    version: str

    def predict(self, request: ModelRequest) -> ModelResult: ...


class PathSnapMLScorer(Protocol):
    """Member 2 boundary for binary candidate scoring.

    ``candidate_features`` has one row per candidate and includes the identity
    columns trip_id/obs_idx/arc_idx/arc_id plus exactly the ordered feature
    columns. Return one ``emission_probability`` in [0,1] per row, preserving
    key values and order. The positive class means this candidate matches the
    observation's route label. This is a per-candidate score, not a decoded path.
    """

    name: str
    version: str

    def score_candidates(self, request: PathSnapModelRequest) -> PathSnapModelResult: ...
