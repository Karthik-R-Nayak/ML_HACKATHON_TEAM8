"""Write a small synthetic dataset (same assumed formats) to try the scripts without real data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from conftest import make_grid, simulate
from pathsnap.network import build_network

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
nodes, arcs, (proj, look, nid) = make_grid()
net = build_network(nodes, arcs)
nodes.to_csv(out / "city.nodes", header=False, index=False, sep=" ")
arcs.to_csv(out / "city.arcs", header=False, index=False, sep=" ")
spec = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (4, 2), (5, 2)]
for i in range(3):
    tr, _, route = simulate([nid(a, b) for a, b in spec], look, net.projector, lambda n: tuple(net.node_xy[n]), seed=i)
    tr["t"] = tr["t"].round(0)
    tr[["lon", "lat", "t"]].to_csv(out / f"trip{i}.track", header=False, index=False)
    (out / f"trip{i}.route").write_text("\n".join(map(str, route)))
