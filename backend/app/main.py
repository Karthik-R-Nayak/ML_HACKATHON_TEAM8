import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from app.contracts import AnalysisResponse, PathSnapPreprocessRequest, PathSnapPreprocessResponse, SequenceInput
from app.pipeline import AdapterNotConfigured, Pipeline
from app.fixtures import fixture_pipeline
from app.pathsnap.pipeline import PATHSNAP_PIPELINE


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    env = os.getenv("APP_ENV", "development").lower()
    fixtures_enabled = os.getenv("ENABLE_TEST_FIXTURES", "false").lower() == "true"
    if fixtures_enabled and env == "production":
        raise RuntimeError("Test fixtures cannot be enabled in production")
    configured = pipeline or (fixture_pipeline() if fixtures_enabled else Pipeline())
    if configured.fixture and env == "production":
        raise RuntimeError("Fixture pipeline cannot be registered in production")
    app = FastAPI(title="Sequence Inference API", version="1.0.0",
                  description="Adapter-based sequence analysis API. Adapter readiness is reported by /v1/capabilities.")
    origins = [item.strip() for item in os.getenv("API_CORS_ORIGINS", "http://localhost:5173").split(",") if item.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], allow_credentials=False)
    app.state.pipeline = configured
    app.state.pathsnap_pipeline = PATHSNAP_PIPELINE

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/v1/capabilities")
    def capabilities():
        return configured.capabilities()

    @app.get("/v1/pathsnap/capabilities")
    def pathsnap_capabilities():
        return {
            "preprocessing": "ready",
            "candidate_generation": "ready",
            "feature_extraction": "ready",
            "heading_mode": "central",
            "ml_model": "not_connected",
            "hmm_viterbi": "not_connected",
        }

    @app.post("/v1/pathsnap/preprocess", response_model=PathSnapPreprocessResponse)
    def pathsnap_preprocess(payload: PathSnapPreprocessRequest):
        try:
            return app.state.pathsnap_pipeline.run_payload(payload)
        except (ValueError, KeyError, IndexError) as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_pathsnap_input", "message": str(exc)}) from exc

    @app.post("/v1/analyze", response_model=AnalysisResponse)
    def analyze(payload: SequenceInput):
        try:
            return configured.analyze(payload)
        except AdapterNotConfigured as exc:
            raise HTTPException(status_code=503, detail={"code": "adapter_not_configured", "message": str(exc)}) from exc
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=502, detail={"code": "adapter_contract_violation", "message": str(exc)}) from exc

    @app.get("/")
    def frontend():
        index = os.path.join(os.path.dirname(__file__), "static", "index.html")
        if os.path.exists(index):
            return FileResponse(index)
        return {"service": "Sequence Inference API", "docs": "/docs", "capabilities": "/v1/capabilities"}

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/assets", StaticFiles(directory=os.path.join(static_dir, "assets")), name="assets")
    return app


app = create_app()
