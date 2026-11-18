"""Real-artifact HTTP integration tests. Never substitute a fixture classifier."""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from laptop1_demo.server import create_app, DEFAULT_WORK, load_selected
from pathsnap.io import parse_track, parse_nodes, parse_arcs


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as client:
        yield client


def payload():
    root = DEFAULT_WORK / 'data/student_test/raw/00000000'
    return dict(
        trip_id="api-check",
        observations=parse_track(root / '00000000.track').iloc[:3][['lon', 'lat', 't']].to_dict('records'),
        nodes=parse_nodes(root / '00000000.nodes').to_dict('records'),
        arcs=parse_arcs(root / '00000000.arcs').to_dict('records')
    )


def test_real_model_health_and_truthful_readiness(client):
    health = client.get('/healthz').json()
    assert health['model_loaded'] is True
    assert health['hmm_connected'] is True
    assert health['decoder'] == 'viterbi_directed'

    capabilities = client.get('/v1/pathsnap/capabilities').json()
    assert capabilities['ml_model'] == 'ready'
    assert capabilities['hmm_viterbi'] == 'ready'
    assert capabilities['full_route_pipeline_ready'] is True
    assert len(client.get('/v1/pathsnap/trips').json()['trips']) == 10


def test_full_real_trip_matches_original_inference(client):
    response = client.post('/v1/pathsnap/trips/00000000/score')
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['observation_count'] == 2503 and body['candidate_count'] == 23160
    assert body['probability_dtype'] == 'float64'
    c, _ = joblib.load(DEFAULT_WORK / 'data/student_test/tables.joblib')
    mask = c.trip_id.eq('00000000\\00000000')
    expected = np.load(DEFAULT_WORK / 'results/student_random_forest_untuned_probabilities.npz')['probability'][mask]
    actual = np.array([r['emission_probability'] for r in body['candidates']])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert np.isfinite(actual).all() and ((actual >= 0) & (actual <= 1)).all()
    result = pd.DataFrame(body['candidates'])
    for key in ['obs_idx', 'arc_id', 'arc_idx']:
        np.testing.assert_array_equal(result[key], c.loc[mask, key].to_numpy())
    assert not body['is_fixture'] and not body['labeling_performed']
    assert not {'label', 'label_valid', 'aligned_route_pos'} & set(result)


def test_full_route_matching_stored_trip(client):
    response = client.post('/v1/pathsnap/trips/00000052/match')
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['trip_id'] == '00000052'
    assert data['observation_count'] == 262
    assert data['matched_count'] == 262
    assert data['path_log_probability'] is not None
    assert 'metrics' in data
    assert data['metrics']['hmm_accuracy'] == 1.0  # 100% on trip 00000052
    assert data['metrics']['hmm_continuity'] > 0.95
    assert 'map_geometry' in data
    assert len(data['map_geometry']['gps_points']) == 262
    assert len(data['map_geometry']['matched_arcs']) > 0


def test_reference_routes_cannot_influence_scores(client):
    p = payload()
    a = client.post('/v1/pathsnap/score', json=p)
    p['route_arc_ids'] = [999999999]
    b = client.post('/v1/pathsnap/score', json=p)
    assert a.status_code == b.status_code == 200
    left, right = pd.DataFrame(a.json()['candidates']), pd.DataFrame(b.json()['candidates'])
    pd.testing.assert_frame_equal(left.drop(columns='emission_probability'), right.drop(columns='emission_probability'))
    np.testing.assert_allclose(left.emission_probability, right.emission_probability, rtol=1e-10, atol=1e-12)
    assert b.json()['reference_route_ignored'] is True


def test_validation_and_unknown_trip(client):
    assert client.post('/v1/pathsnap/trips/not-a-trip/score').status_code == 404
    assert client.post('/v1/pathsnap/trips/not-a-trip/match').status_code == 404
    p = payload()
    p['observations'][0]['t'] = 'nan'
    assert client.post('/v1/pathsnap/score', json=p).status_code == 422
    p = payload()
    p['arcs'][0]['from_node'] = 999999999
    assert client.post('/v1/pathsnap/score', json=p).status_code == 422


def test_ui_endpoint(client):
    res = client.get('/ui')
    assert res.status_code == 200
    assert 'PathSnap' in res.text
    assert 'Evaluator' in res.text


def test_missing_or_wrong_model_fails_explicitly(tmp_path):
    with pytest.raises(ValueError, match='not found'):
        load_selected(tmp_path / 'missing.joblib')
    bad = tmp_path / 'wrong.joblib'
    joblib.dump(dict(model='logistic_regression', variant='untuned'), bad)
    with pytest.raises(ValueError, match='Random Forest'):
        load_selected(bad)
