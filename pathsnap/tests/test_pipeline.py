import numpy as np
import pandas as pd
import pytest

from conftest import simulate
from pathsnap.align import ALIGNED, align_route
from pathsnap.candidates import generate_candidates
from pathsnap.config import CandidateConfig, FeatureConfig, PipelineConfig
from pathsnap.dataset import group_folds, process_trip, run_pipeline, training_matrix
from pathsnap.features import FEATURE_COLUMNS, observation_features
from pathsnap.labels import LABEL_COLUMNS, META_COLUMNS, recall_by_radius, recall_report


def nodes_path(w, spec):
    return [w["nid"](i, j) for i, j in spec]


L_ROUTE = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2), (5, 2)]   # east, north, east


def sim(w, spec=L_ROUTE, **kw):
    net = w["net"]
    return simulate(nodes_path(w, spec), w["look"], net.projector, w["xy_of"], **kw)


def test_features_values(world):
    """Eastbound constant-speed track: bearing 90, speed 10, heading diff 0 vs eastbound arc, 180 vs twin."""
    net = world["net"]
    x = np.arange(10) * 50.0
    xy = np.column_stack([x, np.full(10, 3.0)]) + net.node_xy[0]
    t = np.arange(10) * 5.0
    of = observation_features(xy, t, FeatureConfig())
    assert np.allclose(of["gps_speed_mps"], 10) and np.allclose(of["gps_bearing_deg"], 90, atol=0.5)
    assert np.isnan(of["dt_prev_s"][0]) and of["dt_next_s"][0] == 5 and of["dt_prev_s"][1] == 5
    cands, _ = generate_candidates(net, world["index"], xy, CandidateConfig())
    from pathsnap.features import build_candidate_features
    df = build_candidate_features(net, cands, of)
    row = lambda a: df[(df.obs_idx == 4) & (df.arc_idx == world["look"][a])].iloc[0]
    assert row((0, 1))["heading_diff_deg"] < 1 and row((1, 0))["heading_diff_deg"] > 179
    assert row((0, 1))["proj_frac"] == pytest.approx(200 / 200 * 0 + (xy[4, 0] - net.node_xy[0, 0]) / 200, abs=0.01)
    assert row((0, 1))["from_out_deg"] == 2 and set(FEATURE_COLUMNS) <= set(df.columns)


def test_stationary_heading_nan(world):
    xy = np.tile(world["net"].node_xy[0], (5, 1)) + np.random.default_rng(0).normal(0, 0.3, (5, 2))
    of = observation_features(xy, np.arange(5) * 1.0)
    assert of["gps_bearing_deg"].isna().all() and (of["gps_heading_valid"] == 0).all()


def test_candidate_recall_and_alignment_accuracy(world):
    track, true_arc, route = sim(world)
    cand, obs = process_trip("t1", world["net"], world["index"], track, route, world["cfg"])
    rep = recall_report(obs)
    assert rep["labeled_fraction"] > 0.95
    assert rep["recall"] >= 0.99 and rep["miss_rate"] <= 0.01
    # alignment vs simulated truth
    lab = obs[obs.status == ALIGNED]
    acc = (lab.aligned_arc_id.to_numpy() == true_arc[lab.obs_idx.to_numpy()]).mean()
    assert acc >= 0.9, acc
    assert (np.diff(lab.aligned_route_pos.to_numpy()) >= 0).all()          # temporal order preserved
    # exactly one positive per labeled observation that has the GT among candidates
    pos = cand[cand.label == 1].groupby("obs_idx").size()
    assert (pos == 1).all()
    assert recall_by_radius(obs, [100, 150])[150] >= recall_by_radius(obs, [100, 150])[100]


def test_track_and_route_lengths_unrelated(world):
    """Different sampling rates give different track lengths for the same route; labels still consistent."""
    t1, _, route = sim(world, dt=5.0)
    t2, tr2, _ = sim(world, dt=2.0, seed=3)
    assert len(t1) != len(t2) and len(t1) != len(route)
    _, obs = process_trip("t2", world["net"], world["index"], t2, route, world["cfg"])
    lab = obs[obs.status == ALIGNED]
    assert (lab.aligned_arc_id.to_numpy() == tr2[lab.obs_idx.to_numpy()]).mean() >= 0.9


def test_loop_revisits_same_arc_in_order(world):
    """Route goes out and comes back over the same arcs in reverse + a U-turn; positions must be monotone."""
    spec = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (1, 0), (2, 0), (3, 0)]   # (1,0)->(2,0) used twice
    track, true_arc, route = sim(world, spec)
    assert len(set(route.tolist())) < len(route)
    _, obs = process_trip("loop", world["net"], world["index"], track, route, world["cfg"])
    lab = obs[obs.status == ALIGNED]
    assert (np.diff(lab.aligned_route_pos.to_numpy()) >= 0).all()
    assert (lab.aligned_arc_id.to_numpy() == true_arc[lab.obs_idx.to_numpy()]).mean() >= 0.85


def test_outliers_left_unlabeled(world):
    track, _, route = sim(world)
    track = track.copy()
    far_lon, far_lat = world["net"].projector.inverse(np.array([5000.0]), np.array([5000.0]))
    track.loc[5, ["lon", "lat"]] = [far_lon[0], far_lat[0]]
    _, obs = process_trip("o", world["net"], world["index"], track, route, world["cfg"])
    assert obs.loc[5, "status"] != ALIGNED and obs.loc[5, "aligned_arc_id"] == -1


def test_miss_reasons_small_radius(world):
    track, _, route = sim(world, noise=12.0)
    cfg = PipelineConfig(cand=CandidateConfig(radius_m=5.0))
    _, obs = process_trip("m", world["net"], world["index"], track, route, cfg)
    rep = recall_report(obs)
    assert rep["miss_rate"] > 0.3 and rep["miss_radius"] > 0
    cfg2 = PipelineConfig(cand=CandidateConfig(radius_m=100.0, max_candidates=1))
    _, obs2 = process_trip("m", world["net"], world["index"], track, route, cfg2)
    assert recall_report(obs2)["miss_cap"] > 0 or recall_report(obs2)["recall"] > 0.5


def test_no_label_leakage(world):
    """Features must be identical with the true route, a wrong route, and no route at all."""
    track, _, route = sim(world)
    w = world
    wrong = np.array(route[::-1])
    c_true, _ = process_trip("x", w["net"], w["index"], track, route, w["cfg"])
    c_wrong, _ = process_trip("x", w["net"], w["index"], track, wrong, w["cfg"])
    c_none, o_none = process_trip("x", w["net"], w["index"], track, None, w["cfg"])
    for other in (c_wrong, c_none):
        pd.testing.assert_frame_equal(c_true[FEATURE_COLUMNS + ["obs_idx", "arc_id"]].reset_index(drop=True),
                                      other[FEATURE_COLUMNS + ["obs_idx", "arc_id"]].reset_index(drop=True))
    assert "label" not in c_none.columns and "status" not in o_none.columns
    assert not set(FEATURE_COLUMNS) & set(LABEL_COLUMNS + META_COLUMNS)
    assert not any("label" in c or "route" in c or "aligned" in c or "gt_" in c for c in FEATURE_COLUMNS)
    X, y, g = training_matrix(c_true)
    assert list(X.columns) == FEATURE_COLUMNS and set(g) == {"x"}


def test_group_folds_no_trip_overlap():
    trips = [f"t{i}" for i in range(23)]
    seen = []
    for tr, va in group_folds(trips, 5, seed=1):
        assert not set(tr) & set(va) and set(tr) | set(va) == set(trips)
        seen += list(va)
    assert sorted(seen) == sorted(trips)


def test_end_to_end_files(world, tmp_path):
    w = world
    for i, seed in enumerate([0, 1, 2]):
        track, _, route = sim(w, seed=seed)
        track = track.assign(t=lambda d: d.t.round(0))
        track[["lon", "lat", "t"]].to_csv(tmp_path / f"trip{i}.track", header=False, index=False)
        if i < 2:   # trip2 has no route -> inference-only
            (tmp_path / f"trip{i}.route").write_text("\n".join(map(str, route)))
    w["nodes"].to_csv(tmp_path / "city.nodes", header=False, index=False, sep=" ")
    w["arcs"].to_csv(tmp_path / "city.arcs", header=False, index=False, sep=" ")
    cand, obs, rep = run_pipeline(tmp_path, tmp_path / "out")
    assert set(cand.trip_id) == {"trip0", "trip1", "trip2"}
    assert (tmp_path / "out" / "candidates.csv.gz").exists()
    unl = cand[cand.trip_id == "trip2"]
    assert len(unl) > 0 and not unl["label_valid"].fillna(False).astype(bool).any()   # no route -> never labeled
    assert rep["recall"] >= 0.99
