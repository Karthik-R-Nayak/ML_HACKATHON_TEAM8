import numpy as np
import pandas as pd
import pytest

from pathsnap import io
from pathsnap.geo import LocalProjector, angle_diff_deg, bearing_deg, haversine_m, point_segment_projection


def test_projection_matches_haversine():
    p = LocalProjector(77.59, 12.97)
    lon = np.array([77.59, 77.60, 77.62, 77.57]); lat = np.array([12.97, 12.99, 12.95, 12.98])
    x, y = p.forward(lon, lat)
    d_proj = np.hypot(x[1:] - x[0], y[1:] - y[0])
    assert np.allclose(d_proj, haversine_m(77.59, 12.97, lon[1:], lat[1:]), rtol=2e-3)
    pair = np.hypot(x[2] - x[1], y[2] - y[1])
    assert pair == pytest.approx(haversine_m(lon[1], lat[1], lon[2], lat[2]), rel=5e-3)
    lo2, la2 = p.inverse(x, y)
    assert np.allclose(lo2, lon, atol=1e-9) and np.allclose(la2, lat, atol=1e-9)


def test_bearing_and_angle_diff():
    assert bearing_deg(0, 1) == 0 and bearing_deg(1, 0) == 90 and bearing_deg(0, -1) == 180 and bearing_deg(-1, 0) == 270
    assert angle_diff_deg(350, 10) == 20 and angle_diff_deg(0, 180) == 180 and angle_diff_deg(90, 90) == 0


def test_point_segment():
    d, t = point_segment_projection(5, 3, 0, 0, 10, 0)
    assert d == 3 and t == 0.5
    d, t = point_segment_projection(-4, 3, 0, 0, 10, 0)
    assert d == 5 and t == 0.0                       # clamped to endpoint
    d, t = point_segment_projection(1, 1, 2, 2, 2, 2)  # zero-length segment
    assert t == 0 and d == pytest.approx(np.sqrt(2))


def test_parsers_variants(tmp_path):
    (tmp_path / "a.nodes").write_text("lon lat\n77.5 12.9\n77.6 12.95\n")
    assert len(io.parse_nodes(tmp_path / "a.nodes")) == 2          # header + whitespace
    (tmp_path / "b.nodes").write_text("77.5,12.9\n77.6,12.95\n")
    assert len(io.parse_nodes(tmp_path / "b.nodes")) == 2          # comma, no header
    (tmp_path / "a.arcs").write_text("0 1\n1 0\n")
    assert io.parse_arcs(tmp_path / "a.arcs")["to_node"].tolist() == [1, 0]
    (tmp_path / "a.route").write_text("3\n4 5\n3\n")
    assert io.parse_route(tmp_path / "a.route").tolist() == [3, 4, 5, 3]
    (tmp_path / "a.track").write_text("77.5,12.9,2020-01-01 00:00:00\n77.5,12.9,2020-01-01 00:00:10\n")
    assert np.diff(io.parse_track(tmp_path / "a.track")["t"]).tolist() == [10.0]
    (tmp_path / "b.track").write_text("77.5 12.9 1700000000000\n77.5 12.9 1700000005000\n")
    assert np.diff(io.parse_track(tmp_path / "b.track")["t"]).tolist() == [5.0]   # ms auto-detected


def test_unsorted_track_sorted_and_swapped_cols_rejected(tmp_path):
    (tmp_path / "c.track").write_text("77.5 12.9 20\n77.5 12.9 10\n")
    assert io.parse_track(tmp_path / "c.track")["t"].tolist() == [10.0, 20.0]
    (tmp_path / "d.nodes").write_text("12.9 177.5\n")
    with pytest.raises(ValueError, match="swapped"):
        io.parse_nodes(tmp_path / "d.nodes")
