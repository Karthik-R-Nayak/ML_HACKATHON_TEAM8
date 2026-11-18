"""Typed Member 3 bridge: HMM/Viterbi route decoder on directed road network topology."""
import json
from pathlib import Path
from typing import Any

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


class Member3ViterbiDecoder(PathSnapRouteDecoder):
    name = "viterbi"
    version = "member3-viterbi-v1"

    def __init__(
        self,
        beta_m: float = 10.0,
        disconnect_penalty: float = 50.0,
        max_gap_s: float = 15.0,
        max_speed_mps: float = 40.0,
        min_emission_prob: float = 1e-6,
        back_tol_m: float = 5.0,
        version: str = "member3-viterbi-v1",
    ):
        if beta_m <= 0:
            raise ValueError("beta_m must be strictly positive")
        if disconnect_penalty < 0:
            raise ValueError("disconnect_penalty must be non-negative")
        if max_gap_s <= 0:
            raise ValueError("max_gap_s must be strictly positive")
        self.beta_m = float(beta_m)
        self.disconnect_penalty = float(disconnect_penalty)
        self.max_gap_s = float(max_gap_s)
        self.max_speed_mps = float(max_speed_mps)
        self.min_emission_prob = float(min_emission_prob)
        self.back_tol_m = float(back_tol_m)
        self.version = version

    def decode_path(self, request: PathSnapDecodeRequest) -> PathSnapDecodeResult:
        if not isinstance(request, PathSnapDecodeRequest):
            raise TypeError("Expected PathSnapDecodeRequest")

        obs = request.observations
        cand = request.candidate_scores
        net = request.network
        trip_id = request.trip_id
        score_col = request.score_column or EMISSION_COLUMN

        if not isinstance(obs, pd.DataFrame):
            raise TypeError("observations must be a pandas DataFrame")
        if not isinstance(cand, pd.DataFrame):
            raise TypeError("candidate_scores must be a pandas DataFrame")

        # Empty observations
        if len(obs) == 0:
            empty_df = pd.DataFrame(
                columns=list(DECODED_PATH_COLUMNS)
            ).astype({
                "trip_id": str,
                "obs_idx": np.int64,
                "arc_idx": "Int64",
                "arc_id": "Int64",
                "decode_status": str,
            })
            return PathSnapDecodeResult(decoded_path=empty_df, path_log_probability=None)

        if "obs_idx" not in obs.columns:
            raise ValueError("observations must contain obs_idx")

        # Validate candidates schema if candidates present
        if len(cand) > 0:
            required_cand_cols = set(CANDIDATE_KEY_COLUMNS) | {score_col}
            missing = required_cand_cols - set(cand.columns)
            if missing:
                raise ValueError(f"candidate_scores missing columns: {sorted(missing)}")
            if not cand["trip_id"].eq(trip_id).all():
                raise ValueError("candidate_scores trip_id differs from request.trip_id")
            scores = cand[score_col].to_numpy(dtype=float)
            if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ValueError(f"{score_col} must contain finite probabilities in [0, 1]")

        T = len(obs)
        sorted_obs = obs.sort_values("obs_idx", kind="stable").reset_index(drop=True)
        obs_indices = sorted_obs["obs_idx"].to_numpy(dtype=np.int64)

        # Handle empty candidates case
        if len(cand) == 0:
            result_df = pd.DataFrame({
                "trip_id": [trip_id] * T,
                "obs_idx": obs_indices,
                "arc_idx": pd.Series([None] * T, dtype="Int64"),
                "arc_id": pd.Series([None] * T, dtype="Int64"),
                "decode_status": ["no_candidates"] * T,
            })
            return PathSnapDecodeResult(decoded_path=result_df, path_log_probability=None)

        # Extract timestamps and project GPS coordinates if available
        if "gps_timestamp_s" in sorted_obs.columns:
            timestamps = sorted_obs["gps_timestamp_s"].to_numpy(dtype=float)
        elif "t" in sorted_obs.columns:
            timestamps = sorted_obs["t"].to_numpy(dtype=float)
        else:
            timestamps = np.arange(T, dtype=float)

        if "gps_lon" in sorted_obs.columns and "gps_lat" in sorted_obs.columns:
            gx, gy = net.projector.forward(
                sorted_obs["gps_lon"].to_numpy(dtype=float),
                sorted_obs["gps_lat"].to_numpy(dtype=float),
            )
            gps_xy_m = np.column_stack([gx, gy])
        else:
            gps_xy_m = np.zeros((T, 2), dtype=float)

        # Build directed graph from network topology
        n_nodes = len(net.node_xy)
        adj = sp.csr_matrix(
            (net.length, (net.arc_from, net.arc_to)),
            shape=(n_nodes, n_nodes),
        )

        # Group candidates by obs_idx
        cands_by_obs = {
            obs_i: grp.reset_index(drop=True)
            for obs_i, grp in cand.groupby("obs_idx", observed=True)
        }

        # Viterbi Dynamic Programming variables
        V: list[np.ndarray | None] = []
        backtrack: list[np.ndarray | None] = []
        path_cache: dict[int, np.ndarray] = {}
        last_valid_step: int | None = None

        for step_idx in range(T):
            obs_id = obs_indices[step_idx]
            if obs_id not in cands_by_obs or len(cands_by_obs[obs_id]) == 0:
                V.append(None)
                backtrack.append(None)
                continue

            ct = cands_by_obs[obs_id]
            kt = len(ct)
            p_emit = ct[score_col].to_numpy(dtype=float)
            log_emis = np.log(np.maximum(p_emit, self.min_emission_prob))

            # Initial step or reset step
            if last_valid_step is None or V[last_valid_step] is None:
                V.append(log_emis)
                backtrack.append(np.full(kt, -1, dtype=int))
                last_valid_step = step_idx
                continue

            prev_V = V[last_valid_step]
            prev_obs_id = obs_indices[last_valid_step]
            c_prev = cands_by_obs[prev_obs_id]
            kp = len(c_prev)

            dt = float(timestamps[step_idx] - timestamps[last_valid_step])
            dgps = float(np.hypot(
                gps_xy_m[step_idx, 0] - gps_xy_m[last_valid_step, 0],
                gps_xy_m[step_idx, 1] - gps_xy_m[last_valid_step, 1],
            ))

            # If large time gap, treat as trajectory boundary / reset
            if dt > self.max_gap_s:
                V.append(log_emis)
                backtrack.append(np.full(kt, -1, dtype=int))
                last_valid_step = step_idx
                continue

            u_arcs = c_prev["arc_idx"].to_numpy(dtype=int)
            v_arcs = ct["arc_idx"].to_numpy(dtype=int)
            u_to = net.arc_to[u_arcs]
            v_from = net.arc_from[v_arcs]
            u_frac = c_prev["proj_frac"].to_numpy(dtype=float) if "proj_frac" in c_prev.columns else np.full(kp, 0.5)
            v_frac = ct["proj_frac"].to_numpy(dtype=float) if "proj_frac" in ct.columns else np.full(kt, 0.5)
            u_len = net.length[u_arcs]
            v_len = net.length[v_arcs]

            # Cache shortest path distances from unique u_to nodes
            unique_u_to = [int(u) for u in np.unique(u_to)]
            needed = [u for u in unique_u_to if u not in path_cache]
            if needed:
                search_limit = max(150.0, dgps * 2.5 + 50.0, self.max_speed_mps * dt + 50.0)
                dists = csg.dijkstra(adj, directed=True, indices=needed, limit=search_limit)
                for idx, u_node in enumerate(needed):
                    path_cache[u_node] = dists[idx]

            # Compute transition log-probabilities
            trans = np.full((kp, kt), -self.disconnect_penalty, dtype=float)
            for i in range(kp):
                u = u_arcs[i]
                to_node = u_to[i]
                d_out = path_cache[to_node]
                u_rem = (1.0 - u_frac[i]) * u_len[i]
                for j in range(kt):
                    v = v_arcs[j]
                    if u == v:
                        ds = (v_frac[j] - u_frac[i]) * u_len[i]
                        if ds >= -self.back_tol_m:
                            dnet = max(0.0, ds)
                            trans[i, j] = - abs(dnet - dgps) / self.beta_m
                    elif to_node == v_from[j]:
                        dnet = u_rem + v_frac[j] * v_len[j]
                        trans[i, j] = - abs(dnet - dgps) / self.beta_m
                    else:
                        d_graph = d_out[v_from[j]]
                        if np.isfinite(d_graph):
                            dnet = u_rem + d_graph + v_frac[j] * v_len[j]
                            trans[i, j] = - abs(dnet - dgps) / self.beta_m

            total_score = prev_V[:, None] + trans  # shape (kp, kt)
            best_prev = np.argmax(total_score, axis=0)
            best_score = total_score[best_prev, np.arange(kt)]

            # If all paths are severely disconnected, soft reset
            if not np.isfinite(best_score).any() or (best_score <= -1e8).all():
                V.append(log_emis)
                backtrack.append(np.full(kt, -1, dtype=int))
            else:
                V.append(best_score + log_emis)
                backtrack.append(best_prev)

            last_valid_step = step_idx

        # Backtracking phase
        decoded_arc_idx = [None] * T
        decoded_arc_id = [None] * T
        decode_status = ["no_candidates"] * T

        # Find best end state
        if last_valid_step is not None and V[last_valid_step] is not None:
            final_log_prob = float(np.max(V[last_valid_step]))
            curr_k = int(np.argmax(V[last_valid_step]))
            step_cursor = last_valid_step

            while step_cursor is not None and step_cursor >= 0:
                if V[step_cursor] is not None:
                    cand_table = cands_by_obs[obs_indices[step_cursor]]
                    selected_row = cand_table.iloc[curr_k]
                    decoded_arc_idx[step_cursor] = int(selected_row["arc_idx"])
                    decoded_arc_id[step_cursor] = int(selected_row["arc_id"])
                    decode_status[step_cursor] = "matched"

                    prev_k = backtrack[step_cursor][curr_k]
                    # Find previous valid step
                    prev_step = step_cursor - 1
                    while prev_step >= 0 and V[prev_step] is None:
                        prev_step -= 1

                    if prev_step >= 0 and V[prev_step] is not None:
                        if prev_k == -1:
                            curr_k = int(np.argmax(V[prev_step]))
                        else:
                            curr_k = int(prev_k)
                        step_cursor = prev_step
                    else:
                        break
                else:
                    step_cursor -= 1
        else:
            final_log_prob = None

        # Build output DataFrame
        result_df = pd.DataFrame({
            "trip_id": [trip_id] * T,
            "obs_idx": obs_indices,
            "arc_idx": pd.Series(decoded_arc_idx, dtype="Int64"),
            "arc_id": pd.Series(decoded_arc_id, dtype="Int64"),
            "decode_status": decode_status,
        })

        return PathSnapDecodeResult(
            decoded_path=result_df,
            path_log_probability=final_log_prob if (final_log_prob is not None and np.isfinite(final_log_prob)) else None,
        )


def load_member3_decoder(artifact_dir: Path) -> PathSnapRouteDecoder:
    directory = Path(artifact_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {directory}")

    manifest_path = directory / "manifest.json"
    config_path = directory / "decoder_config.json"

    params: dict[str, Any] = {}
    if config_path.is_file():
        try:
            params = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid decoder_config.json: {exc}") from exc
    elif manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            params = manifest.get("parameters", {})
        except Exception as exc:
            raise ValueError(f"Invalid manifest.json: {exc}") from exc

    beta_m = params.get("beta_m", 10.0)
    disconnect_penalty = params.get("disconnect_penalty", 50.0)
    max_gap_s = params.get("max_gap_s", 15.0)
    max_speed_mps = params.get("max_speed_mps", 40.0)
    min_emission_prob = params.get("min_emission_prob", 1e-6)
    back_tol_m = params.get("back_tol_m", 5.0)

    return Member3ViterbiDecoder(
        beta_m=beta_m,
        disconnect_penalty=disconnect_penalty,
        max_gap_s=max_gap_s,
        max_speed_mps=max_speed_mps,
        min_emission_prob=min_emission_prob,
        back_tol_m=back_tol_m,
    )
