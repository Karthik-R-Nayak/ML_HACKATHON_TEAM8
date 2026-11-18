import numpy as np
import pandas as pd
import pytest

from pathsnap.config import IOConfig
from pathsnap.geo import point_segment_projection
from pathsnap.index import GridIndex
from pathsnap.network import build_network


def test_degrees_and_bearing(world):
    net = world["net"]
    corner = 0                       # node (0,0): 2 out, 2 in
    centre = world["nid"](2, 2)      # 4 out, 4 in
    assert (net.out_deg[corner], net.in_deg[corner]) == (2, 2)
    assert (net.out_deg[centre], net.in_deg[centre]) == (4, 4)
    a = world["look"][(0, 1)]        # eastbound
    assert net.bearing[a] == pytest.approx(90, abs=0.5) and net.length[a] == pytest.approx(200, abs=1)
    assert net.arc_ids[a] == a


def test_bad_node_reference_raises(world):
    arcs = pd.DataFrame({"from_node": [0], "to_node": [999]})
    with pytest.raises(ValueError, match="node id convention"):
        build_network(world["nodes"], arcs)
    # 1-based ids work with the right base
    arcs1 = world["arcs"] + 1
    net = build_network(world["nodes"], arcs1, IOConfig(node_id_base=1))
    assert np.allclose(net.length, world["net"].length)


def test_route_id_range_check(world):
    with pytest.raises(ValueError, match="arc_id_base"):
        world["net"].route_to_idx([10 ** 6])


def test_index_matches_bruteforce():
    rng = np.random.default_rng(1)
    n = 400
    a = rng.uniform(0, 3000, (n, 2)); b = a + rng.normal(0, 300, (n, 2))   # includes long diagonals
    idx = GridIndex(a[:, 0], a[:, 1], b[:, 0], b[:, 1], cell_m=100)
    for _ in range(200):
        p = rng.uniform(-200, 3200, 2); r = rng.uniform(10, 250)
        d, _ = point_segment_projection(p[0], p[1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        truth = set(np.flatnonzero(d <= r))
        got = idx.query(p[0], p[1], r)
        assert truth <= set(got.tolist())          # no false negatives
