"""Data preprocessing adapter contract.

Input: PreprocessRequest with raw sequence and declared alphabet. Output: same
length normalized sequence, one source position per output symbol, optional warnings.
Preserve positional provenance; reject unsupported alphabets/symbols explicitly.
Implement Preprocessor.preprocess; no API, storage, or model-specific dependencies.
"""
from typing import Protocol
from app.contracts import PreprocessRequest, PreprocessResult


class PathSnapPreprocessingAdapter:
    """Build the projected road network and project ordered GPS observations.

    Input tables use lon/lat nodes and integer from_node/to_node arcs; GPS track
    has lon, lat, and epoch seconds. Output retains the RoadNetwork's original
    ``arc_ids`` and stable internal ``arc_idx`` mapping.
    """
    name = "pathsnap-geographic-preprocessing"
    version = "member1-import-v1"

    def prepare(self, nodes, arcs, track, io_config):
        import numpy as np
        from app.pathsnap.preprocessing.network import build_network

        net = build_network(nodes, arcs, io_config)
        x, y = net.projector.forward(track["lon"].to_numpy(), track["lat"].to_numpy())
        return net, np.column_stack([x, y]), track["t"].to_numpy()


class Preprocessor(Protocol):
    name: str
    version: str

    def preprocess(self, request: PreprocessRequest) -> PreprocessResult: ...
