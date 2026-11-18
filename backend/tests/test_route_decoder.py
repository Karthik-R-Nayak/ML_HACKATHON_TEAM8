"""Unit and integration tests for Member 3 HMM / Viterbi route decoder."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.adapters.hmm_viterbi import PathSnapRouteDecoder
from app.pathsnap.route_decoder import Member3RouteDecoder, load_member3_decoder
from app.pathsnap.member2_scorer import load_member2_scorer
from app.pathsnap.interfaces import (
    DECODED_PATH_COLUMNS,
    PathSnapModelRequest,
    PathSnapDecodeRequest,
    PathSnapDecodeResult,
)
from app.pathsnap.pipeline import PATHSNAP_PIPELINE
from pathsnap.io import parse_nodes, parse_arcs, load_trip_files

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT.parent / "experiments/member2/data/student_test/raw/00000052"


@pytest.fixture(scope="module")
def scorer():
    return load_member2_scorer(ROOT / "backend/artifacts/member2")


@pytest.fixture(scope="module")
def decoder():
    return load_member3_decoder()


@pytest.fixture(scope="module")
def trip_52_prep_and_scored(scorer):
    nodes = parse_nodes(RAW_DIR / "00000052.nodes")
    arcs = parse_arcs(RAW_DIR / "00000052.arcs")
    track, route = load_trip_files(RAW_DIR / "00000052.track", RAW_DIR / "00000052.route")

    prep = PATHSNAP_PIPELINE.prepare(
        trip_id="00000052", nodes=nodes, arcs=arcs, track=track, route_arc_ids=route
    )
    model_req = PathSnapModelRequest(
        trip_id="00000052",
        candidate_features=prep.candidate_features,
        feature_columns=prep.candidate_feature_columns,
    )
    scored = scorer.score_candidates(model_req)
    return prep, scored


def test_decoder_implements_protocol(decoder):
    assert hasattr(decoder, "decode_path")
    assert callable(decoder.decode_path)
    assert hasattr(decoder, "name") and decoder.name == "hmm_viterbi"
    assert hasattr(decoder, "version") and "member3" in decoder.version


def test_decode_path_real_trip_success(decoder, trip_52_prep_and_scored):
    prep, scored = trip_52_prep_and_scored
    req = PathSnapDecodeRequest(
        trip_id="00000052",
        candidate_scores=scored.candidate_scores,
        observations=prep.observations,
        network=prep.network,
    )
    result = decoder.decode_path(req)

    assert isinstance(result, PathSnapDecodeResult)
    assert isinstance(result.decoded_path, pd.DataFrame)
    assert tuple(result.decoded_path.columns) == DECODED_PATH_COLUMNS
    assert len(result.decoded_path) == len(prep.observations)
    assert result.path_log_probability is not None
    assert np.isfinite(result.path_log_probability)

    # Validate high accuracy on trip 00000052 (> 95%)
    aligned = prep.observations[prep.observations.status == "aligned"]
    merged = aligned.merge(result.decoded_path, on="obs_idx")
    acc = (merged["arc_id"] == merged["aligned_arc_id"]).mean()
    assert acc >= 0.95

    # Validate route continuity
    net = prep.network
    arc_idx_vals = result.decoded_path["arc_idx"].to_numpy()
    valid_mask = pd.notna(arc_idx_vals)[:-1] & pd.notna(arc_idx_vals)[1:]
    u = arc_idx_vals[:-1][valid_mask].astype(int)
    v = arc_idx_vals[1:][valid_mask].astype(int)
    connected = (u == v) | (net.arc_to[u] == net.arc_from[v])
    continuity = connected.mean()
    assert continuity >= 0.95


def test_decode_empty_trip(decoder, trip_52_prep_and_scored):
    prep, scored = trip_52_prep_and_scored
    req = PathSnapDecodeRequest(
        trip_id="empty_trip",
        candidate_scores=scored.candidate_scores.iloc[:0],
        observations=prep.observations.iloc[:0],
        network=prep.network,
    )
    result = decoder.decode_path(req)
    assert len(result.decoded_path) == 0
    assert tuple(result.decoded_path.columns) == DECODED_PATH_COLUMNS
    assert result.path_log_probability is None


def test_decode_with_missing_candidates(decoder, trip_52_prep_and_scored):
    prep, scored = trip_52_prep_and_scored
    # Drop all candidates for observation 5
    filtered_cands = scored.candidate_scores[scored.candidate_scores.obs_idx != 5].copy()
    req = PathSnapDecodeRequest(
        trip_id="00000052",
        candidate_scores=filtered_cands,
        observations=prep.observations,
        network=prep.network,
    )
    result = decoder.decode_path(req)
    assert len(result.decoded_path) == len(prep.observations)
    row_5 = result.decoded_path[result.decoded_path.obs_idx == 5].iloc[0]
    assert row_5["decode_status"] == "no_candidates"
    assert row_5["arc_id"] is None
    assert row_5["arc_idx"] is None


def test_decode_with_large_time_gap(decoder, trip_52_prep_and_scored):
    prep, scored = trip_52_prep_and_scored
    obs = prep.observations.copy()
    # Inject a large 2-minute time gap at observation 50
    obs.loc[obs.obs_idx >= 50, "gps_timestamp_s"] += 120.0
    req = PathSnapDecodeRequest(
        trip_id="00000052",
        candidate_scores=scored.candidate_scores,
        observations=obs,
        network=prep.network,
    )
    result = decoder.decode_path(req)
    assert len(result.decoded_path) == len(prep.observations)
    assert result.path_log_probability is not None
