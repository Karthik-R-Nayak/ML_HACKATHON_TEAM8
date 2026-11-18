"""Unit and integration tests for Member 3 HMM/Viterbi Route Decoder."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.pathsnap.member3_decoder import Member3ViterbiDecoder, load_member3_decoder
from app.pathsnap.member2_scorer import load_member2_scorer
from app.pathsnap.interfaces import (
    DECODED_PATH_COLUMNS,
    PathSnapModelRequest,
    PathSnapDecodeRequest,
    PathSnapDecodeResult,
)
from app.pathsnap.pipeline import PATHSNAP_PIPELINE
from pathsnap.io import parse_nodes, parse_arcs, parse_track

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "backend/artifacts/member3"
TEST_TRIP_DIR = ROOT.parent / "experiments/member2/data/student_test/raw/00000052"


@pytest.fixture(scope="module")
def decoder():
    return load_member3_decoder(ARTIFACT_DIR)


@pytest.fixture(scope="module")
def scorer():
    return load_member2_scorer(ROOT / "backend/artifacts/member2")


@pytest.fixture(scope="module")
def prepared_trip():
    nodes = parse_nodes(TEST_TRIP_DIR / "00000052.nodes")
    arcs = parse_arcs(TEST_TRIP_DIR / "00000052.arcs")
    track = parse_track(TEST_TRIP_DIR / "00000052.track")
    return PATHSNAP_PIPELINE.prepare(
        trip_id="00000052",
        nodes=nodes,
        arcs=arcs,
        track=track,
        route_arc_ids=None,
    )


def test_decoder_loader():
    dec = load_member3_decoder(ARTIFACT_DIR)
    assert isinstance(dec, Member3ViterbiDecoder)
    assert dec.name == "viterbi"
    assert dec.version == "member3-viterbi-v1"
    assert dec.beta_m == 10.0
    assert dec.disconnect_penalty == 50.0


def test_decode_real_trip(decoder, scorer, prepared_trip):
    scored = scorer.score_candidates(
        PathSnapModelRequest(
            trip_id=prepared_trip.trip_id,
            candidate_features=prepared_trip.candidate_features,
            feature_columns=prepared_trip.candidate_feature_columns,
        )
    )

    req = PathSnapDecodeRequest(
        trip_id=prepared_trip.trip_id,
        candidate_scores=scored.candidate_scores,
        observations=prepared_trip.observations,
        network=prepared_trip.network,
    )

    result = decoder.decode_path(req)
    assert isinstance(result, PathSnapDecodeResult)
    assert result.path_log_probability is not None
    assert np.isfinite(result.path_log_probability)

    df = result.decoded_path
    assert list(df.columns) == list(DECODED_PATH_COLUMNS)
    assert len(df) == len(prepared_trip.observations)
    assert (df["obs_idx"].to_numpy() == np.arange(len(prepared_trip.observations))).all()
    assert (df["trip_id"] == "00000052").all()

    # All matched in this test trip
    assert (df["decode_status"] == "matched").all()
    assert not df["arc_idx"].isna().any()
    assert not df["arc_id"].isna().any()


def test_empty_observations(decoder, prepared_trip):
    empty_obs = prepared_trip.observations.iloc[:0].copy()
    req = PathSnapDecodeRequest(
        trip_id="empty_trip",
        candidate_scores=prepared_trip.candidate_features.iloc[:0].copy(),
        observations=empty_obs,
        network=prepared_trip.network,
    )
    result = decoder.decode_path(req)
    assert isinstance(result, PathSnapDecodeResult)
    assert len(result.decoded_path) == 0
    assert list(result.decoded_path.columns) == list(DECODED_PATH_COLUMNS)
    assert result.path_log_probability is None


def test_empty_candidates_handling(decoder, prepared_trip):
    obs = prepared_trip.observations.iloc[:5].copy()
    empty_cand = pd.DataFrame(columns=[
        "trip_id", "obs_idx", "arc_idx", "arc_id", "emission_probability"
    ])
    req = PathSnapDecodeRequest(
        trip_id="00000052",
        candidate_scores=empty_cand,
        observations=obs,
        network=prepared_trip.network,
    )
    result = decoder.decode_path(req)
    assert len(result.decoded_path) == 5
    assert (result.decoded_path["decode_status"] == "no_candidates").all()
    assert result.decoded_path["arc_idx"].isna().all()
    assert result.decoded_path["arc_id"].isna().all()
    assert result.path_log_probability is None


def test_invalid_requests(decoder, prepared_trip):
    # Invalid request type
    with pytest.raises(TypeError):
        decoder.decode_path("not a request")

    # Invalid observations type
    with pytest.raises(TypeError):
        decoder.decode_path(
            PathSnapDecodeRequest(
                trip_id="t",
                candidate_scores=pd.DataFrame(),
                observations="not df",
                network=prepared_trip.network,
            )
        )

    # Missing obs_idx in observations
    bad_obs = prepared_trip.observations.drop(columns="obs_idx")
    with pytest.raises(ValueError):
        decoder.decode_path(
            PathSnapDecodeRequest(
                trip_id="00000052",
                candidate_scores=prepared_trip.candidate_features.iloc[:2],
                observations=bad_obs,
                network=prepared_trip.network,
            )
        )
