from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from app.pathsnap.member2_scorer import load_member2_scorer, Member2Scorer
from app.pathsnap.interfaces import PathSnapModelRequest, PathSnapModelResult, PathSnapDecodeRequest
from app.pathsnap.preprocessing.features import FEATURE_COLUMNS
from app.pathsnap.pipeline import PATHSNAP_PIPELINE
from pathsnap.ml.predict import score_candidates as original_score
from pathsnap.io import parse_track, parse_nodes, parse_arcs

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent/'experiments/member2'


@pytest.fixture(scope='module')
def scorer():
    return load_member2_scorer(ROOT/'backend/artifacts/member2')


@pytest.fixture(scope='module')
def table():
    c, _ = joblib.load(WORK/'data/student_test/tables.joblib')
    return c[c.trip_id.eq('00000000\\00000000')].copy()


def request(table):
    return PathSnapModelRequest(trip_id='00000000\\00000000', candidate_features=table, feature_columns=tuple(FEATURE_COLUMNS))


def test_real_saved_probabilities_and_complete_identity_preservation(scorer,table):
    incoming=table.copy()
    incoming['backend_candidate_id']=np.arange(len(incoming))
    incoming.index=np.arange(len(incoming))*2+9
    result=scorer.score_candidates(request(incoming))
    assert isinstance(result,PathSnapModelResult)
    pd.testing.assert_frame_equal(result.candidate_scores.drop(columns='emission_probability'),incoming)
    assert result.candidate_scores.emission_probability.dtype==np.dtype('float64')
    expected=np.load(WORK/'results/student_random_forest_untuned_probabilities.npz')['probability'][:len(table)]
    np.testing.assert_allclose(result.candidate_scores.emission_probability,expected,rtol=1e-10,atol=1e-12)
    assert result.candidate_scores.emission_probability.between(0,1).all()


def test_column_order_nan_and_repeated_inference(scorer,table):
    small=table.iloc[:30].copy();small.loc[small.index[:3],'heading_diff_deg']=np.nan
    bundle=joblib.load(ROOT/'backend/artifacts/member2/selected_model.joblib')
    expected=original_score(bundle,small).match_probability
    for _ in range(2):
        actual=scorer.score_candidates(request(small[small.columns[::-1]])).candidate_scores.emission_probability
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)


def test_empty_batch(scorer,table):
    result=scorer.score_candidates(request(table.iloc[:0])).candidate_scores
    assert result.empty and result.emission_probability.dtype==np.dtype('float64')
    assert list(result.columns)==list(table.columns)+['emission_probability']


@pytest.mark.parametrize('case',['missing_feature','missing_identity','wrong_trip','duplicate','infinite','text','null_key','wrong_order'])
def test_invalid_requests(scorer,table,case):
    small=table.iloc[:3].copy()
    columns=tuple(FEATURE_COLUMNS)
    if case=='missing_feature':small=small.drop(columns='dist_m')
    if case=='missing_identity':small=small.drop(columns='arc_idx')
    if case=='wrong_trip':small['trip_id']='other'
    if case=='duplicate':small=pd.concat([small,small.iloc[:1]])
    if case=='infinite':small.loc[small.index[0],'dist_m']=np.inf
    if case=='text':small['dist_m']='bad'
    if case=='null_key':small['arc_idx']=np.nan
    if case=='wrong_order':columns=columns[::-1]
    with pytest.raises(ValueError):scorer.score_candidates(PathSnapModelRequest(request(small).trip_id,small,columns))


def test_missing_invalid_wrong_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):load_member2_scorer(tmp_path/'absent')
    with pytest.raises(FileNotFoundError):load_member2_scorer(tmp_path)
    bad=tmp_path/'bad.joblib';bad.write_bytes(b'not a joblib file')
    with pytest.raises(ValueError):load_member2_scorer(tmp_path)
    joblib.dump(dict(model='logistic_regression',variant='untuned'),bad)
    with pytest.raises(ValueError,match='Random Forest'):load_member2_scorer(tmp_path)


def test_discovery_without_assumed_filename(tmp_path):
    # A renamed artifact is discovered by its actual bundle metadata.
    import shutil
    shutil.copy2(ROOT/'backend/artifacts/member2/selected_model.joblib',tmp_path/'team-model-renamed.joblib')
    assert load_member2_scorer(tmp_path).name=='random_forest'


@pytest.mark.parametrize('change', ['feature_order','positive_class','missing_filename','escape','checksum'])
def test_invalid_manifests(tmp_path,change):
    import json,shutil
    manifest=json.loads((ROOT/'backend/artifacts/member2/manifest.json').read_text())
    if change=='feature_order':manifest['feature_columns']=manifest['feature_columns'][::-1]
    if change=='positive_class':manifest['positive_class']=0
    if change=='missing_filename':del manifest['artifact_file']
    if change=='escape':manifest['artifact_file']='../outside.joblib'
    if change=='checksum':
        shutil.copy2(ROOT/'backend/artifacts/member2/selected_model.joblib',tmp_path/'selected_model.joblib')
        manifest['sha256']='wrong'
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):load_member2_scorer(tmp_path)


def test_real_backend_preparation_to_model_to_decoder_request(scorer):
    root=WORK/'data/student_test/raw/00000000'
    prepared=PATHSNAP_PIPELINE.prepare(trip_id='real-http-trip',nodes=parse_nodes(root/'00000000.nodes'),
        arcs=parse_arcs(root/'00000000.arcs'),track=parse_track(root/'00000000.track'),route_arc_ids=None)
    scored=scorer.score_candidates(PathSnapModelRequest(prepared.trip_id,prepared.candidate_features,prepared.candidate_feature_columns))
    downstream=PathSnapDecodeRequest(trip_id=prepared.trip_id,candidate_scores=scored.candidate_scores,
        observations=prepared.observations,network=prepared.network)
    assert downstream.score_column=='emission_probability'
    assert len(downstream.candidate_scores)==23160 and len(downstream.observations)==2503
    assert not prepared.labeling_performed
    expected=original_score(scorer._bundle,prepared.candidate_features).match_probability
    np.testing.assert_allclose(downstream.candidate_scores.emission_probability,expected,rtol=1e-10,atol=1e-12)
    # No decoder exists in the supplied backend: constructing its real input is
    # compatibility validation, not a claim that a route has been reconstructed.
