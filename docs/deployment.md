# Deployment

## Container

Build from the repository root and run the resulting image with a production adapter package and factory registered by the deployer:

```sh
docker build -t sequence-inference:latest .
docker run --rm -p 8000:8000 -e APP_ENV=production sequence-inference:latest
```

The image defaults to no adapters and serves `503 adapter_not_configured` for analysis. This is deliberate: package/import paths and credentials for upstream components are not assumed. Add the actual project modules and a small composition factory that constructs `Pipeline(preprocessor=..., candidate_generator=..., feature_extractor=..., model=..., decoder=...)`, then configure the ASGI command to load it. Do not enable `ENABLE_TEST_FIXTURES` in production; app startup refuses that combination.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Environment label; `production` enforces fixture prohibition |
| `ENABLE_TEST_FIXTURES` | `false` | Explicit local transport test switch; synthetic output only |
| `VITE_API_BASE_URL` | same origin | Frontend build-time API origin when UI and API are hosted separately |
| `API_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed browser origins; set the exact frontend origin(s) in deployment |

Use TLS termination and request-size limits at the ingress. Restrict `/docs` if your environment requires it. Set worker count based on adapter concurrency and model memory; avoid multiple model copies without sizing. The service does not persist inputs/results and should not log sequence payloads.

## Health and readiness

`GET /healthz` is a liveness check. `GET /v1/capabilities` reports stage readiness and adapter name/version without exposing input data. Use `pipeline_ready` for readiness. A live process with missing adapters is healthy but not ready for inference.

## Frontend-only development

Run Vite separately and point `VITE_API_BASE_URL` to the API origin. For single-origin deployment, the Docker build copies Vite's static output into the backend package and FastAPI serves `/` plus `/assets`.
