"""Laptop 1 FastAPI host using supplied GPS contracts and original ML inference."""
from contextlib import asynccontextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pathsnap"))
sys.path.insert(0, str(ROOT / "backend"))
from app.contracts import PathSnapPreprocessRequest
from app.pathsnap.pipeline import PATHSNAP_PIPELINE
from app.pathsnap.interfaces import PathSnapModelRequest
from app.pathsnap.member2_scorer import Member2Scorer
from pathsnap.config import PipelineConfig, IOConfig
from pathsnap.candidates import build_index
from pathsnap.dataset import discover_trips, process_trip
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.io import load_trip_files, parse_arcs, parse_nodes
from pathsnap.network import build_network
from pathsnap.ml.predict import score_candidates

DEFAULT_WORK = ROOT.parent / "experiments/member2"
PACKAGED_MODEL = ROOT / "backend/artifacts/member2/selected_model.joblib"


def load_selected(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Selected model not found: {path}; set PATHSNAP_MODEL_PATH")
    bundle = joblib.load(path)
    if bundle.get("model") != "random_forest" or bundle.get("variant") != "untuned":
        raise ValueError("Only the selected untuned Random Forest is supported")
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Artifact feature order is incompatible")
    classifier = bundle["pipeline"].named_steps["classifier"]
    from sklearn.ensemble import RandomForestClassifier
    if not isinstance(classifier, RandomForestClassifier) or classifier.classes_.tolist() != [0, 1]:
        raise ValueError("Expected fitted binary RandomForestClassifier with classes [0,1]")
    if classifier.n_features_in_ != 17 or bundle["pipeline_config"] != asdict(PipelineConfig()):
        raise ValueError("Artifact preprocessing/feature configuration is incompatible")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = os.getenv("PATHSNAP_MODEL_SHA256", "b5533a8709b43a90e06376991b659edf848d188079cebcccee502ba203476eb4")
    if digest != expected:
        raise ValueError("Artifact checksum does not match the selected model")
    return bundle, digest


def run_scoring(bundle, trip_id, track, nodes, arcs):
    started = time.perf_counter()
    # No route parameter is accepted from the caller or disk at this boundary.
    prepared = PATHSNAP_PIPELINE.prepare(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route_arc_ids=None)
    scorer = Member2Scorer(bundle)
    scored = scorer.score_candidates(PathSnapModelRequest(prepared.trip_id,
        prepared.candidate_features, prepared.candidate_feature_columns)).candidate_scores
    probability = scored.emission_probability.to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid model probability output")
    # Copy the complete candidate table: arc_idx and every identity field survive.
    scored["emission_probability"] = probability
    if len(scored):
        chosen = scored.sort_values(["obs_idx", "emission_probability", "dist_rank", "arc_id"],
            ascending=[True, False, True, True], kind="stable").drop_duplicates("obs_idx")
    else:
        chosen = scored
    predictions = pd.DataFrame({"trip_id": trip_id, "obs_idx": np.arange(len(track)), "t": track.t.to_numpy()})
    predictions = predictions.merge(chosen[["obs_idx", "arc_id", "arc_idx", "emission_probability"]], on="obs_idx", how="left", validate="one_to_one")
    predictions["status"] = np.where(predictions.arc_id.isna(), "no_candidates", "scored")
    # pandas JSON converts feature NaNs to null, permitting strict JSON responses.
    return dict(trip_id=trip_id, model="random_forest", variant="untuned", is_fixture=False,
        hmm_connected=False, labeling_performed=False, feature_columns=FEATURE_COLUMNS,
        observation_count=len(track), candidate_count=len(scored),
        probability_dtype=str(scored.emission_probability.dtype),
        elapsed_seconds=time.perf_counter()-started,
        candidates=json.loads(scored.to_json(orient="records", double_precision=15)),
        predictions=json.loads(predictions.to_json(orient="records", double_precision=15)),
        notice="Real classifier scores only. Candidate-match posteriors are not calibrated HMM emission likelihoods. No route reconstruction is connected.")


def create_app(model_path=None, trips_dir=None):
    default_model = PACKAGED_MODEL if PACKAGED_MODEL.is_file() else DEFAULT_WORK / "models/selected_model.joblib"
    model_path = model_path or os.getenv("PATHSNAP_MODEL_PATH", str(default_model))
    trips_dir = Path(trips_dir or os.getenv("PATHSNAP_TRIPS_DIR", str(DEFAULT_WORK / "data/student_test/raw"))).resolve()

    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle, app.state.checksum = load_selected(model_path)
        yield

    app = FastAPI(title="PathSnap Laptop 1 - real ML scoring", lifespan=lifespan)
    origins = [s.strip() for s in os.getenv("API_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if s.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"], allow_credentials=False)

    def registry():
        if not trips_dir.is_dir():
            return {}
        result = {}
        for entry in discover_trips(trips_dir):
            key = entry[1].stem
            if key in result:
                raise ValueError(f"Ambiguous trip ID: {key}")
            result[key] = entry
        return result

    @app.get("/healthz")
    def health():
        return dict(status="ok", model_loaded=True, model="random_forest", model_sha256=app.state.checksum, is_fixture=False)

    @app.get("/v1/pathsnap/capabilities")
    def capabilities():
        return dict(preprocessing="ready", candidate_generation="ready", feature_extraction="ready",
                    ml_model="ready", hmm_viterbi="not_connected", full_route_pipeline_ready=False, fixture_mode=False)

    @app.get("/v1/pathsnap/trips")
    def trips():
        return dict(trips=[dict(trip_id=k) for k in sorted(registry())])

    @app.post("/v1/pathsnap/trips/{trip_id}/score")
    def score_stored(trip_id: str):
        entry = registry().get(trip_id)
        if entry is None:
            raise HTTPException(404, detail="Unknown trip ID")
        _, tp, _, npth, apth = entry
        try:
            cfg = PipelineConfig()
            track, _ = load_trip_files(tp, None, cfg.io)
            return run_scoring(app.state.bundle, trip_id, track, parse_nodes(npth), parse_arcs(apth))
        except (ValueError, KeyError, IndexError) as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    @app.post("/v1/pathsnap/score")
    def score_payload(payload: PathSnapPreprocessRequest):
        if payload.node_id_base != 0 or payload.arc_id_base != 0:
            # The selected artifact was trained with zero-based IDs.
            raise HTTPException(422, detail="Selected model demo requires zero-based network IDs")
        try:
            cfg = PipelineConfig()
            track = pd.DataFrame([x.model_dump() for x in payload.observations]).sort_values("t", kind="stable").reset_index(drop=True)
            nodes = pd.DataFrame([x.model_dump() for x in payload.nodes])
            arcs = pd.DataFrame([x.model_dump() for x in payload.arcs])
            result = run_scoring(app.state.bundle, payload.trip_id, track, nodes, arcs)
            result["reference_route_ignored"] = payload.route_arc_ids is not None
            return result
        except (ValueError, KeyError, IndexError) as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    return app


app = create_app()
