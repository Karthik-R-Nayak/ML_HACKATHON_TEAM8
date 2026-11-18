import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest

from pathsnap.candidates import build_index
from pathsnap.config import PipelineConfig
from pathsnap.geo import LocalProjector
from pathsnap.network import build_network

LON0, LAT0, SP, NG = 77.59, 12.97, 200.0, 6   # Bengaluru, 200 m spacing, 6x6 lattice


def make_grid():
    """Bidirectional lattice. Returns nodes_df, arcs_df, (proj, lookup dict (a,b)->arc id)."""
    proj = LocalProjector(LON0, LAT0)
    xs, ys = np.meshgrid(np.arange(NG) * SP, np.arange(NG) * SP)
    lon, lat = proj.inverse(xs.ravel(), ys.ravel())
    nodes = pd.DataFrame({"lon": lon, "lat": lat})
    nid = lambda i, j: j * NG + i
    arcs, look = [], {}
    for j in range(NG):
        for i in range(NG):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < NG and j + dj < NG:
                    for a, b in ((nid(i, j), nid(i + di, j + dj)), (nid(i + di, j + dj), nid(i, j))):
                        look[(a, b)] = len(arcs); arcs.append((a, b))
    return nodes, pd.DataFrame(arcs, columns=["from_node", "to_node"]), (proj, look, nid)


def simulate(route_nodes, look, proj, nid_xy, speed=10.0, dt=5.0, noise=8.0, seed=0):
    """Drive along node path; sample every dt s. Returns track df (lon,lat,t), true arc id per sample, route ids."""
    rng = np.random.default_rng(seed)
    pts = np.array([nid_xy(n) for n in route_nodes])
    seg_len = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    s = np.arange(0, cum[-1], speed * dt)
    seg = np.minimum(np.searchsorted(cum, s, side="right") - 1, len(seg_len) - 1)
    f = (s - cum[seg]) / seg_len[seg]
    xy = pts[seg] + (pts[seg + 1] - pts[seg]) * f[:, None] + rng.normal(0, noise, (len(s), 2))
    lon, lat = proj.inverse(xy[:, 0], xy[:, 1])
    route = [look[(route_nodes[k], route_nodes[k + 1])] for k in range(len(seg_len))]
    track = pd.DataFrame({"lon": lon, "lat": lat, "t": 1.7e9 + np.arange(len(s)) * dt})
    return track, np.array(route)[seg], np.array(route)


@pytest.fixture(scope="session")
def world():
    nodes, arcs, (proj, look, nid) = make_grid()
    net = build_network(nodes, arcs)
    idx = build_index(net)
    # net projector is centred on bbox middle, not LON0; simulation must use net.projector
    xy_of = lambda n: (net.node_xy[n, 0], net.node_xy[n, 1])
    return dict(nodes=nodes, arcs=arcs, net=net, index=idx, look=look, nid=nid, xy_of=xy_of, cfg=PipelineConfig())
