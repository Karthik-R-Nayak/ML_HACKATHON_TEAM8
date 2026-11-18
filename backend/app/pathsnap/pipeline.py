"""Composition bridge from GPS/network JSON to Member 1's preprocessing modules."""
import numpy as np
import pandas as pd

from app.adapters.preprocessing import PathSnapPreprocessingAdapter
from app.adapters.candidate_generation import PathSnapCandidateGenerationAdapter
from app.adapters.feature_extraction import PathSnapFeatureExtractionAdapter
from app.pathsnap.preprocessing.align import align_route
from app.pathsnap.preprocessing.config import (
    AlignConfig, CandidateConfig, FeatureConfig, IOConfig,
)
from app.pathsnap.preprocessing.features import FEATURE_COLUMNS
from app.pathsnap.preprocessing.labels import attach_labels, observation_table
from app.pathsnap.interfaces import PathSnapPreparedTrip


class PathSnapPreprocessingPipeline:
    """Run projection → road index → candidates → observation/candidate features.

    The route is optional and used only for label generation/evaluation. Normal
    inference passes ``route_arc_ids=None`` and never loads a ``.route`` file.
    """
    def __init__(self):
        self.preprocessor = PathSnapPreprocessingAdapter()
        self.candidate_generator = PathSnapCandidateGenerationAdapter()
        self.feature_extractor = PathSnapFeatureExtractionAdapter()

    def run(self, *, trip_id, nodes, arcs, track, node_id_base=0, arc_id_base=0,
            route_arc_ids=None):
        return self.prepare(
            trip_id=trip_id, nodes=nodes, arcs=arcs, track=track,
            node_id_base=node_id_base, arc_id_base=arc_id_base,
            route_arc_ids=route_arc_ids,
        ).to_preprocess_response()

    def prepare(self, *, trip_id, nodes, arcs, track, node_id_base=0, arc_id_base=0,
                route_arc_ids=None) -> PathSnapPreparedTrip:
        """Return internal DataFrames/network for scorer and decoder composition."""
        io_cfg = IOConfig(node_id_base=node_id_base, arc_id_base=arc_id_base)
        candidate_cfg = CandidateConfig()
        feature_cfg = FeatureConfig(heading_mode="central")
        align_cfg = AlignConfig()

        net, xy, timestamps = self.preprocessor.prepare(nodes, arcs, track, io_cfg)
        _, raw_candidates, n_within_radius = self.candidate_generator.generate_for_track(
            net, xy, candidate_cfg,
        )
        candidate_frame, observation_features, feature_columns = self.feature_extractor.extract_for_track(
            net, raw_candidates, xy, timestamps, feature_cfg,
        )
        candidate_frame.insert(0, "trip_id", trip_id)

        aligned = None
        if route_arc_ids is not None:
            route_idx = net.route_to_idx(np.asarray(route_arc_ids, dtype=np.int64))
            aligned = align_route(xy, route_idx, net, align_cfg)
            candidate_frame = attach_labels(candidate_frame, aligned)

        observation_frame = observation_table(
            trip_id, net, raw_candidates, observation_features, n_within_radius,
            aligned, candidate_cfg,
        )
        observation_frame["gps_lon"] = track["lon"].to_numpy()
        observation_frame["gps_lat"] = track["lat"].to_numpy()
        observation_frame["gps_timestamp_s"] = track["t"].to_numpy()
        if "orig_row" in track.columns:
            observation_frame["source_obs_idx"] = track["orig_row"].to_numpy()
        return PathSnapPreparedTrip(
            trip_id=trip_id,
            candidate_features=candidate_frame,
            observations=observation_frame,
            network=net,
            candidate_feature_columns=tuple(feature_columns),
            observation_feature_columns=tuple(observation_features.columns),
            labeling_performed=route_arc_ids is not None,
            route_arc_ids=None if route_arc_ids is None else tuple(int(value) for value in route_arc_ids),
        )

    def run_payload(self, payload):
        return self.prepare_payload(payload).to_preprocess_response()

    def prepare_payload(self, payload) -> PathSnapPreparedTrip:
        nodes = pd.DataFrame([item.model_dump() for item in payload.nodes], columns=["lon", "lat"])
        arcs = pd.DataFrame([item.model_dump() for item in payload.arcs], columns=["from_node", "to_node"])
        track = pd.DataFrame([item.model_dump(exclude_none=True) for item in payload.observations])
        return self.prepare(
            trip_id=payload.trip_id, nodes=nodes, arcs=arcs, track=track,
            node_id_base=payload.node_id_base, arc_id_base=payload.arc_id_base,
            route_arc_ids=payload.route_arc_ids,
        )


PATHSNAP_PIPELINE = PathSnapPreprocessingPipeline()
