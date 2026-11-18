"""Road segments from nodes + arcs."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import IOConfig
from .geo import LocalProjector, bearing_deg


@dataclass
class RoadNetwork:
    projector: LocalProjector
    node_xy: np.ndarray
    arc_from: np.ndarray      # node row indices
    arc_to: np.ndarray
    arc_ids: np.ndarray       # external arc ids
    ax: np.ndarray
    ay: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    length: np.ndarray
    bearing: np.ndarray       # NaN for (near) zero-length arcs
    in_deg: np.ndarray        # per node
    out_deg: np.ndarray
    arc_base: int = 0

    @property
    def n_arcs(self):
        return len(self.arc_ids)

    def route_to_idx(self, route_ids, strict=True):
        idx = np.asarray(route_ids, dtype=np.int64) - self.arc_base
        bad = (idx < 0) | (idx >= self.n_arcs)
        if bad.any() and strict:
            raise ValueError(
                f"{int(bad.sum())} route arc ids outside [{self.arc_base}, {self.arc_base + self.n_arcs - 1}] "
                f"(e.g. {np.asarray(route_ids)[bad][:5].tolist()}); check arc_id_base")
        return idx[~bad] if not strict else idx


def build_network(nodes: pd.DataFrame, arcs: pd.DataFrame, io: IOConfig = IOConfig(),
                  projector: LocalProjector = None) -> RoadNetwork:
    proj = projector or LocalProjector.from_lonlat(nodes["lon"], nodes["lat"])
    x, y = proj.forward(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    node_xy = np.column_stack([x, y])
    f = arcs["from_node"].to_numpy() - io.node_id_base
    t = arcs["to_node"].to_numpy() - io.node_id_base
    n = len(node_xy)
    bad = (f < 0) | (f >= n) | (t < 0) | (t >= n)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} arcs reference nodes outside [0,{n-1}] after node_id_base="
                         f"{io.node_id_base}; check the node id convention")
    ax, ay, bx, by = node_xy[f, 0], node_xy[f, 1], node_xy[t, 0], node_xy[t, 1]
    dx, dy = bx - ax, by - ay
    length = np.hypot(dx, dy)
    bearing = np.where(length > 0.5, bearing_deg(dx, dy), np.nan)
    return RoadNetwork(
        projector=proj, node_xy=node_xy, arc_from=f, arc_to=t,
        arc_ids=np.arange(len(f), dtype=np.int64) + io.arc_id_base,
        ax=ax, ay=ay, bx=bx, by=by, length=length, bearing=bearing,
        in_deg=np.bincount(t, minlength=n), out_deg=np.bincount(f, minlength=n),
        arc_base=io.arc_id_base)
