"""HMM/Viterbi decoder contract.

Input: T-by-K score matrix and K ordered labels. Output: exactly T labels selected
from that set and a finite path log-probability. Decoder owns transition/emission
parameter versioning and must document start/transition conventions. Implement
SequenceDecoder.decode; validate shapes before indexing.
"""
from typing import Protocol
from app.contracts import DecodeRequest, DecodeResult
from app.pathsnap.interfaces import PathSnapDecodeRequest, PathSnapDecodeResult


class SequenceDecoder(Protocol):
    name: str
    version: str

    def decode(self, request: DecodeRequest) -> DecodeResult: ...


class PathSnapRouteDecoder(Protocol):
    """Member 3 boundary for variable candidate states and road-network paths.

    Consume Member 2's per-candidate emission probabilities together with all
    observations and the RoadNetwork topology. Return exactly one row per input
    obs_idx, choosing a scored candidate or marking an explicit missing/no-path
    state. Use the candidate's arc_idx and original arc_id without remapping.
    Do not use ground-truth route_arc_ids.
    """

    name: str
    version: str

    def decode_path(self, request: PathSnapDecodeRequest) -> PathSnapDecodeResult: ...
