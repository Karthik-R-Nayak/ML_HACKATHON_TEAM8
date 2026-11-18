"""Route <-> GPS alignment used ONLY to create labels (never for features).

Monotone Viterbi over route positions; see module docstring in README of the design notes:
 - states for obs i: route positions j whose arc lies within search_radius
 - route index must be non-decreasing; progress may regress <= back_tol on same arc
 - penalty on |route progress - straight-line GPS displacement| beyond a slack
 - obs without feasible state are left unlabeled rather than forced.
"""
from dataclasses import dataclass

import numpy as np

from .config import AlignConfig
from .geo import point_segment_projection
from .network import RoadNetwork

ALIGNED, NO_ROUTE_NEARBY, INFEASIBLE = "aligned", "no_route_arc_nearby", "dropped_infeasible"


@dataclass
class AlignResult:
    status: np.ndarray        # per obs, str
    route_pos: np.ndarray     # index into the route sequence, -1 if unlabeled
    arc_idx: np.ndarray       # network arc index, -1 if unlabeled
    dist_m: np.ndarray        # distance to aligned arc, NaN if unlabeled


def align_route(xy: np.ndarray, route_arc_idx: np.ndarray, net: RoadNetwork,
                cfg: AlignConfig = AlignConfig()) -> AlignResult:
    n = len(xy)
    status = np.full(n, NO_ROUTE_NEARBY, dtype=object)
    out = AlignResult(status, np.full(n, -1), np.full(n, -1), np.full(n, np.nan))
    route = np.asarray(route_arc_idx, dtype=np.int64)
    if n == 0 or len(route) == 0:
        return out
    ax, ay, bx, by = net.ax[route], net.ay[route], net.bx[route], net.by[route]
    L = net.length[route]
    cum = np.concatenate([[0.0], np.cumsum(L)[:-1]])

    layers = []  # dicts: i, js, d, s, cost, back
    prev = None
    for i in range(n):
        d, t = point_segment_projection(xy[i, 0], xy[i, 1], ax, ay, bx, by)
        js = np.flatnonzero(d <= cfg.search_radius_m)
        if js.size == 0:
            continue
        dj, sj = d[js], cum[js] + t[js] * L[js]
        emis = 0.5 * (dj / cfg.sigma_m) ** 2
        if prev is None:
            cost, back = emis, np.full(js.size, -1)
        else:
            disp = float(np.hypot(*(xy[i] - xy[prev["i"]])))
            ds = sj[None, :] - prev["s"][:, None]
            tol = cfg.tol_abs_m + cfg.tol_rel * disp
            pen = cfg.lam_per_m * np.minimum(np.maximum(0.0, np.abs(ds - disp) - tol), cfg.skip_cap_m)
            infeasible = (ds < -cfg.back_tol_m) | (js[None, :] < prev["js"][:, None])
            tot = prev["cost"][:, None] + np.where(infeasible, np.inf, pen)
            back = np.argmin(tot, axis=0)
            cost = emis + tot[back, np.arange(js.size)]
            if not np.isfinite(cost).any():
                status[i] = INFEASIBLE
                continue
        layer = dict(i=i, js=js, d=dj, s=sj, cost=cost, back=back)
        layers.append(layer)
        prev = layer
    if not layers:
        return out
    k = int(np.argmin(layers[-1]["cost"]))
    for layer in reversed(layers):
        i, j = layer["i"], int(layer["js"][k])
        status[i], out.route_pos[i], out.arc_idx[i], out.dist_m[i] = ALIGNED, j, route[j], layer["d"][k]
        k = int(layer["back"][k])
    return out
