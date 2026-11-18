"""Candidate generation contract.

Input: normalized sequence and source positions. Output: unique candidate IDs,
half-open [start,end) spans in normalized-sequence coordinates, and kind. Keep
span bounds within sequence length and return deterministic ordering. Implement
CandidateGenerator.generate; do not depend on HTTP request objects.
"""
from typing import Protocol
from app.contracts import CandidateRequest, CandidateResult


class PathSnapCandidateGenerationAdapter:
    """Generate nearest road-segment candidates with Member 1's spatial index."""
    name = "pathsnap-grid-candidate-generator"
    version = "member1-import-v1"

    def generate_for_track(self, net, xy, candidate_config):
        from app.pathsnap.preprocessing.candidates import build_index, generate_candidates

        index = build_index(net, candidate_config)
        candidates, n_within_radius = generate_candidates(net, index, xy, candidate_config)
        return index, candidates, n_within_radius


class CandidateGenerator(Protocol):
    name: str
    version: str

    def generate(self, request: CandidateRequest) -> CandidateResult: ...
