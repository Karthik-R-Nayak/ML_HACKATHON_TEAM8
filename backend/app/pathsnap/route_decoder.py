"""Member 3 HMM / Viterbi route decoder implementation.

Fulfills the PathSnapRouteDecoder protocol:
    - Input: PathSnapDecodeRequest with candidate_scores, observations, and RoadNetwork.
    - Output: PathSnapDecodeResult with decoded_path and path_log_probability.
    - Output columns: trip_id, obs_idx, arc_idx, arc_id, decode_status.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from app.adapters.hmm_viterbi import PathSnapRouteDecoder
from app.pathsnap.interfaces import (
    CANDIDATE_KEY_COLUMNS,
    DECODED_PATH_COLUMNS,
    EMISSION_COLUMN,
    PathSnapDecodeRequest,
    PathSnapDecodeResult,
)


class Member3RouteDecoder(PathSnapRouteDecoder):
    """Hidden Markov Model route decoder using directed network distance and Viterbi algorithm."""

    name: str = "hmm_viterbi"
    version: str = "member3-viterbi-v1"

    def __init__(
        self,
        beta: float = 10.0,
        emit_weight: float = 2.0,
        emit_floor: float = 1e-4,
        max_hop_dist: float = 200.0,
        disconnect_penalty: float = -50.0,
        max_dt_gap: float = 30.0,
        back_tol_m: float = 10.0,
    ):
        """Configure HMM parameters.

        Args:
            beta: Scale parameter for transition error exponential distribution (|d_net - d_gps| / beta).
            emit_weight: Weight multiplier on log emission probability.
            emit_floor: Minimum emission probability to prevent log(0).
            max_hop_dist: Maximum network distance in meters for Dijkstra shortest path search.
            disconnect_penalty: Log probability penalty applied when road segments are disconnected.
            max_dt_gap: Maximum GPS time gap (seconds) before resetting the Markov chain.
            back_tol_m: Tolerance for backward movement along the same arc due to GPS jitter.
        """
        self.beta = float(beta)
        self.emit_weight = float(emit_weight)
        self.emit_floor = float(emit_floor)
        self.max_hop_dist = float(max_hop_dist)
        self.disconnect_penalty = float(disconnect_penalty)
        self.max_dt_gap = float(max_dt_gap)
        self.back_tol_m = float(back_tol_m)

    def decode_path(self, request: PathSnapDecodeRequest) -> PathSnapDecodeResult:
        if not isinstance(request, PathSnapDecodeRequest):
            raise TypeError("Expected PathSnapDecodeRequest")

        obs = request.observations
        cands = request.candidate_scores
        net = request.network
        trip_id = request.trip_id
        score_col = request.score_column or EMISSION_COLUMN

        T = len(obs)
        if T == 0:
            empty_df = pd.DataFrame(columns=DECODED_PATH_COLUMNS)
            return PathSnapDecodeResult(decoded_path=empty_df, path_log_probability=None)

        if not isinstance(cands, pd.DataFrame):
            raise TypeError("candidate_scores must be a DataFrame")

        if len(cands) > 0:
            if score_col not in cands.columns:
                raise ValueError(f"Score column '{score_col}' not found in candidate_scores")
            if not cands["trip_id"].eq(trip_id).all():
                raise ValueError("candidate_scores trip_id differs from request.trip_id")
            for col in ("obs_idx", "arc_idx", "arc_id"):
                if col not in cands.columns:
                    raise ValueError(f"Required candidate column '{col}' missing")

        # Prepare road network graph for Dijkstra queries
        n_nodes = len(net.node_xy)
        adj = sp.csr_matrix(
            (net.length, (net.arc_from, net.arc_to)),
            shape=(n_nodes, n_nodes),
        )

        # Projected GPS coordinates
        if "gps_lon" in obs.columns and "gps_lat" in obs.columns:
            gx, gy = net.projector.forward(obs["gps_lon"].to_numpy(), obs["gps_lat"].to_numpy())
        elif "x" in obs.columns and "y" in obs.columns:
            gx, gy = obs["x"].to_numpy(dtype=float), obs["y"].to_numpy(dtype=float)
        else:
            gx = np.zeros(T, dtype=float)
            gy = np.zeros(T, dtype=float)
        g_xy = np.column_stack([gx, gy])

        # Timestamps
        if "gps_timestamp_s" in obs.columns:
            timestamps = obs["gps_timestamp_s"].to_numpy(dtype=float)
        elif "t" in obs.columns:
            timestamps = obs["t"].to_numpy(dtype=float)
        else:
            timestamps = np.arange(T, dtype=float)

        # Group candidates by observation index
        cands_by_obs: List[pd.DataFrame] = []
        if len(cands) > 0:
            by_idx = {k: v for k, v in cands.groupby("obs_idx", sort=False)}
            for t in range(T):
                cands_by_obs.append(by_idx.get(t, cands.iloc[:0]))
        else:
            for t in range(T):
                cands_by_obs.append(cands.iloc[:0])

        # Viterbi structures
        dijkstra_cache: Dict[int, np.ndarray] = {}
        v_prev: Optional[np.ndarray] = None
        prev_cands: Optional[pd.DataFrame] = None
        prev_t: Optional[int] = None
        backpointers: List[Optional[np.ndarray]] = []
        step_best_scores: List[float] = []

        for t in range(T):
            curr = cands_by_obs[t]
            K = len(curr)

            if K == 0:
                backpointers.append(None)
                v_prev = None
                prev_cands = None
                prev_t = None
                continue

            p_emit = np.clip(curr[score_col].to_numpy(dtype=float), self.emit_floor, 1.0)
            log_emit = self.emit_weight * np.log(p_emit)

            if v_prev is None or prev_cands is None or prev_t is None:
                # Start new chain
                v_curr = log_emit
                backpointers.append(None)
            else:
                K_prev = len(prev_cands)
                dt = max(0.01, timestamps[t] - timestamps[prev_t])
                dgps = float(np.hypot(g_xy[t, 0] - g_xy[prev_t, 0], g_xy[t, 1] - g_xy[prev_t, 1]))

                if dt > self.max_dt_gap:
                    # Time gap exceeded: reset chain
                    v_curr = log_emit
                    backpointers.append(None)
                else:
                    u_arc = prev_cands["arc_idx"].to_numpy(dtype=int)
                    v_arc = curr["arc_idx"].to_numpy(dtype=int)
                    u_frac = prev_cands["proj_frac"].to_numpy(dtype=float) if "proj_frac" in prev_cands.columns else np.full(K_prev, 0.5)
                    v_frac = curr["proj_frac"].to_numpy(dtype=float) if "proj_frac" in curr.columns else np.full(K, 0.5)
                    u_len = net.length[u_arc]
                    v_len = net.length[v_arc]

                    u_to = net.arc_to[u_arc]
                    v_from = net.arc_from[v_arc]

                    trans_log = np.full((K_prev, K), self.disconnect_penalty, dtype=float)

                    # Query Dijkstra for unique origin nodes needed
                    unique_u_to = np.unique(u_to)
                    needed = [node for node in unique_u_to if node not in dijkstra_cache]
                    if needed:
                        search_limit = max(self.max_hop_dist, 40.0 * dt)
                        d_mat = csg.dijkstra(adj, directed=True, indices=needed, limit=search_limit)
                        for idx, node in enumerate(needed):
                            dijkstra_cache[node] = d_mat[idx]

                    for i in range(K_prev):
                        ui = u_arc[i]
                        u_to_node = u_to[i]
                        dists_from_u = dijkstra_cache[u_to_node]

                        for j in range(K):
                            vj = v_arc[j]
                            if ui == vj:
                                # Traversed along the same arc
                                dx = (v_frac[j] - u_frac[i]) * u_len[i]
                                if dx >= -self.back_tol_m:
                                    dnet = max(0.0, dx)
                                    diff = abs(dnet - dgps)
                                    trans_log[i, j] = - diff / self.beta
                            else:
                                v_from_node = v_from[j]
                                if u_to_node == v_from_node:
                                    # Direct successor arc
                                    dnet = (1.0 - u_frac[i]) * u_len[i] + v_frac[j] * v_len[j]
                                    diff = abs(dnet - dgps)
                                    trans_log[i, j] = - diff / self.beta
                                else:
                                    # Multi-hop path via network shortest path
                                    graph_d = dists_from_u[v_from_node]
                                    if np.isfinite(graph_d):
                                        dnet = (1.0 - u_frac[i]) * u_len[i] + graph_d + v_frac[j] * v_len[j]
                                        diff = abs(dnet - dgps)
                                        trans_log[i, j] = - diff / self.beta

                    scores = v_prev[:, None] + trans_log
                    best_prev = np.argmax(scores, axis=0)
                    best_val = scores[best_prev, np.arange(K)]

                    # If all transitions are severely disconnected, restart chain
                    if not np.isfinite(best_val).any() or (best_val < -1000).all():
                        v_curr = log_emit
                        backpointers.append(None)
                    else:
                        v_curr = best_val + log_emit
                        backpointers.append(best_prev)

            v_prev = v_curr
            prev_cands = curr
            prev_t = t

        # Backtracking
        pred_arc_idx = [None] * T
        pred_arc_id = [None] * T
        decode_status = ["no_candidates"] * T

        # Find the last observation that has candidates
        last_t = T - 1
        while last_t >= 0 and len(cands_by_obs[last_t]) == 0:
            last_t -= 1

        total_log_prob = None
        if last_t >= 0 and v_prev is not None and len(v_prev) > 0:
            best_k = int(np.argmax(v_prev))
            total_log_prob = float(v_prev[best_k])
            curr_k = best_k

            for t in range(last_t, -1, -1):
                c_t = cands_by_obs[t]
                if len(c_t) == 0:
                    decode_status[t] = "no_candidates"
                    continue

                chosen_row = c_t.iloc[curr_k]
                pred_arc_idx[t] = int(chosen_row["arc_idx"])
                pred_arc_id[t] = int(chosen_row["arc_id"])
                decode_status[t] = "matched"

                bp = backpointers[t]
                if bp is not None:
                    curr_k = int(bp[curr_k])
                else:
                    # Find previous step with candidates and use its best emission score
                    prev_step = t - 1
                    while prev_step >= 0 and len(cands_by_obs[prev_step]) == 0:
                        prev_step -= 1
                    if prev_step >= 0:
                        curr_k = int(np.argmax(cands_by_obs[prev_step][score_col].to_numpy(dtype=float)))

        decoded_df = pd.DataFrame({
            "trip_id": trip_id,
            "obs_idx": np.arange(T, dtype=np.int64),
            "arc_idx": pd.Series(pred_arc_idx, dtype="object"),
            "arc_id": pd.Series(pred_arc_id, dtype="object"),
            "decode_status": decode_status,
        })

        return PathSnapDecodeResult(
            decoded_path=decoded_df,
            path_log_probability=total_log_prob,
        )


def load_member3_decoder(**kwargs) -> PathSnapRouteDecoder:
    """Instantiate and return the Member 3 HMM route decoder."""
    return Member3RouteDecoder(**kwargs)
