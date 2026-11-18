import numpy as np
import pandas as pd

from conftest import make_grid, simulate
from pathsnap.audit import _runs, audit_ids, audit_network, run_audit
from pathsnap.config import IOConfig, PipelineConfig
from pathsnap.network import build_network


def _write(root, world, mutate=None, arc_base=0):
    net = world["net"]
    spec = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2), (5, 2)]
    track, _, route = simulate([world["nid"](a, b) for a, b in spec], world["look"], net.projector,
                               world["xy_of"], dt=1.0, speed=10.0, noise=5.0)
    track["t"] = track["t"].round(0)
    if mutate:
        track = mutate(track)
    track[["lon", "lat", "t"]].to_csv(root / "t0.track", header=False, index=False, sep=" ")
    (root / "t0.route").write_text("\n".join(map(str, route + arc_base)))
    world["nodes"].to_csv(root / "net.nodes", header=False, index=False, sep=" ")
    world["arcs"].to_csv(root / "net.arcs", header=False, index=False, sep=" ")


def test_runs():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert _runs(m) == [(1, 2), (5, 5), (7, 9)] and _runs(np.zeros(4, bool)) == []


def test_clean_trip_not_flagged(world, tmp_path):
    _write(tmp_path, world)
    trips, nets, runs = run_audit(tmp_path, tmp_path / "rep")
    r = trips.iloc[0]
    assert r["flags"] in ("", None) or pd.isna(r["flags"]) or r["flags"] == ""
    assert r["route_connected_base0"] == 1.0 and not r["arc_base_suspect"]
    assert nets.iloc[0]["n_components"] == 1 and nets.iloc[0]["isolated_nodes"] == 0
    assert (tmp_path / "rep" / "audit_summary.md").exists()


def test_arc_base_mismatch_detected(world, tmp_path):
    """Routes written 1-based but audited with base 0: connectivity under base1 is perfect, base0 is not."""
    _write(tmp_path, world, arc_base=1)
    trips, _, _ = run_audit(tmp_path, tmp_path / "rep", PipelineConfig(io=IOConfig(arc_id_base=0)))
    r = trips.iloc[0]
    assert r["route_connected_base1"] == 1.0 and r["route_connected_base0"] < 0.5
    assert bool(r["arc_base_suspect"]) and "arc_id_base_suspect" in r["flags"]


def test_far_outlier_explained_in_runs(world, tmp_path):
    far = world["net"].projector.inverse(np.array([9000.0]), np.array([9000.0]))

    def mutate(tr):
        tr = tr.copy(); tr.loc[10, ["lon", "lat"]] = [far[0][0], far[1][0]]
        return pd.concat([tr, tr.iloc[[3]]]).sort_values("t", kind="stable").reset_index(drop=True)  # + dup row
    _write(tmp_path, world, mutate)
    trips, _, runs = run_audit(tmp_path, tmp_path / "rep")
    r = trips.iloc[0]
    assert r["zero_candidate_obs"] >= 1 and "zero_candidates" in r["flags"] and "duplicate_rows" in r["flags"]
    assert (runs.kind == "zero_candidates").any()
    z = runs[(runs.kind == "zero_candidates") & (runs.nearest_any_arc_m_median > 1000)]
    assert len(z) == 1 and z.iloc[0]["position"] == "interior" and z.iloc[0]["n"] == 1   # index shifted by dup row


def test_network_defects_counted(world):
    nodes, arcs = world["nodes"], world["arcs"].copy()
    arcs.loc[len(arcs)] = [5, 5]                    # self loop
    arcs.loc[len(arcs)] = list(arcs.iloc[0])         # duplicate
    net = build_network(nodes, arcs)
    rep = audit_network(nodes, arcs, net)
    assert rep["self_loops"] == 1 and rep["duplicate_arcs"] == 1 and rep["zero_length_arcs"] == 1
    assert rep["n_components"] == 1
