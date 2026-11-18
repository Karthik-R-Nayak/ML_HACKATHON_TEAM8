# PathSnap • Distributed Map-Matching & Route Reconstruction

**Team 8 • UE24CS352A: Machine Learning • Department of CSE, PES University**

PathSnap is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy, urban GPS observations to a directed road network. It couples offline candidate-level machine learning scoring with a Hidden Markov Model (HMM) Viterbi sequential decoder to resolve road network connectivity and GPS multipath inaccuracies.

---

## 🏗️ Repository Architecture & Deliverables Mapping

This repository is structured in accordance with the mandatory hackathon submission components:

| Required Component | Repository Location | Description |
|---|---|---|
| **Data processing** | `pathsnap/pathsnap/io.py`<br>`backend/app/pathsnap/preprocessing/` | Clean parsing of `.track`, `.nodes`, `.arcs`, and geodesic/local projection. |
| **Feature engineering** | `pathsnap/pathsnap/features.py`<br>`pathsnap/pathsnap/candidates.py` | 17 ordered spatial, kinematic, and topological features (central heading mode). |
| **Model training** | `pathsnap/pathsnap/ml/experiment.py`<br>`pathsnap/pathsnap/ml/models.py` | GroupKFold cross-validation across 5 model families (LR, KNN, DT, RF, HGB). |
| **Model evaluation** | `pathsnap/pathsnap/ml/evaluation.py`<br>`scripts/generate_figures_and_maps.py` | Observation-level Top-1 accuracy, Macro F1, Log Loss, and confusion matrices. |
| **Hyperparameter optimisation** | `pathsnap/pathsnap/ml/experiment.py`<br>`member2_results/tuning_trials.csv` | 12 Optuna TPE Bayesian optimization trials on boosting configurations. |
| **HMM and Viterbi** | `backend/app/pathsnap/member3_decoder.py`<br>`backend/artifacts/member3/` | Directed road-network Viterbi decoder with Dijkstra graph transition modeling. |
| **API Backend** | `laptop1_demo/server.py`<br>`backend/app/main.py` | Production FastAPI server hosting ML scoring, HMM matching, and metrics. |
| **Evaluator UI** | `frontend/evaluator_ui/index.html`<br>`laptop1_demo/server.py` (`/ui`) | Evaluator dashboard with GPU-accelerated Leaflet map and live console. |
| **Deployment** | `docs/laptop1-demo.md`<br>`laptop1_demo/requirements.txt` | Two-laptop deployment instructions, environment configs, and startup scripts. |
| **Visualisations** | `docs/maps/*.html`<br>`docs/figures/*.png` | Interactive multi-layer Folium HTML maps and evaluation comparison plots. |
| **Supporting results** | `member2_results/*.csv`<br>`docs/PathSnap_Technical_Report.pdf` | Consolidated 5-page Technical Report PDF and benchmark metrics logs. |

---

## 📊 Summary of Experimental Results

### 1. Model Selection (Member 2)
Five model families were evaluated using 3-fold `GroupKFold` cross-validation (16 holdout trips, 64 training trips):

| Model Family | Key Configuration | CV Macro F1 | CV Top-1 | Student Top-1 | Route Continuity | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | L2 penalty, liblinear, C=1.0 | 89.82% | 82.59% | 86.22% | 88.4% | Evaluated |
| **KNN (Sampled)** | k=15, distance-weighted | 85.12% | 76.42% | 81.93% | 82.1% | Evaluated |
| **Decision Tree** | max_depth=14, min_leaf=20 | 89.45% | 82.10% | 86.85% | 89.3% | Evaluated |
| **Random Forest** | **120 trees, max_depth=18, min_leaf=10** | **90.61%** | **83.40%** | **87.81%** | **92.1%** | **Selected** |
| **Hist Gradient Boosting** | 120 trees, lr=0.08, max_bins=255 | 90.44% | 83.16% | 87.94% | 91.8% | Tuned |

*Selected production artifact:* `backend/artifacts/member2/selected_model.joblib` (SHA-256: `b5533a8709b43a90e06376991b659edf848d188079cebcccee502ba203476eb4`).

### 2. Route Reconstruction Benchmark (Member 3 HMM vs Top-1)
Evaluated across all 10 student-testing trips against ground truth:

| Trip ID | Observations | Top-1 Accuracy | HMM Accuracy | Top-1 Continuity | HMM Continuity | Latency |
|---|---:|---:|---:|---:|---:|---:|
| `00000000` | 2,503 | 93.97% | **95.96%** | 96.24% | **99.60%** | 1.61 s |
| `00000007` | 1,354 | 93.87% | 88.63% | 99.41% | **99.85%** | 0.63 s |
| `00000009` | 1,049 | 86.37% | **91.80%** | 87.79% | **95.04%** | 1.97 s |
| `00000040` | 2,995 | 91.95% | **92.59%** | 96.16% | **99.40%** | 1.77 s |
| `00000052` | 262 | 95.42% | **100.00%** | 95.02% | **96.55%** | 0.30 s |
| `00000056` | 3,958 | 88.65% | **93.10%** | 88.43% | **96.54%** | 4.61 s |
| `00000065` | 768 | 86.85% | **88.67%** | 91.40% | **98.17%** | 1.15 s |
| `00000076` | 1,486 | 87.08% | **92.60%** | 87.81% | **96.30%** | 2.47 s |
| `00000077` | 3,911 | 68.78% | **70.01%** | 91.64% | **97.85%** | 4.85 s |
| `00000091` | 874 | 85.13% | **86.61%** | 87.51% | **96.56%** | 1.20 s |
| **Mean** | **1,916** | **87.81%** | **90.00% (+2.19%)** | **92.14%** | **97.59% (+5.45%)** | **~2.0 s** |

---

## 🚀 Two-Laptop Deployment Guide

The demonstration uses a distributed architecture over a shared Wi-Fi network:

### Laptop 1: Model & API Server
Hosts the trained model, HMM/Viterbi engine, and FastAPI endpoints.
```cmd
cd pathsnap-repo
..\.venv\Scripts\python.exe -m uvicorn laptop1_demo.server:app --host 0.0.0.0 --port 8000
```
- **Endpoints:**
  - `GET /healthz`: Model & HMM readiness verification.
  - `GET /v1/pathsnap/capabilities`: Status of all 5 pipeline stages.
  - `GET /v1/pathsnap/trips`: Lists reserved evaluation trips.
  - `POST /v1/pathsnap/trips/{trip_id}/match`: Full reconstruction + server-side ground truth metrics.
  - `POST /v1/pathsnap/trips/{trip_id}/score`: Member 2 classifier scoring only.
  - `GET /ui`: Serves the interactive Evaluator Dashboard.

### Laptop 2: Evaluator Interface
Connect Laptop 2 to the same Wi-Fi network as Laptop 1.

**Option A (Direct Browser):**
Open your browser and navigate to:
```text
http://<LAPTOP1_IP>:8000/ui
```
*(Default IP: `http://10.0.13.39:8000/ui`)*

**Option B (Local Web Server):**
Run the client from `frontend/evaluator_ui/` or `PathSnap_Laptop2_Client.zip`:
```powershell
cd frontend/evaluator_ui
python -m http.server 5173 --bind 127.0.0.1
```
Open `http://localhost:5173`, enter `http://<LAPTOP1_IP>:8000`, and click **Connect**.

---

## 🧪 Testing and Verification

Run the full end-to-end regression test suite (68 tests):
```powershell
cd pathsnap-repo
$env:PYTHONPATH="backend;pathsnap"
& "..\.venv\Scripts\python.exe" -m pytest backend/tests laptop1_demo/test_server.py pathsnap/tests -v
```

---

## 📄 Deliverables & Reports

- **Technical Report (PDF):** [`docs/PathSnap_Technical_Report.pdf`](docs/PathSnap_Technical_Report_Team8.pdf)
- **Folium Interactive Maps:**
  - [`docs/maps/trip_00000052_folium_map.html`](docs/maps/trip_00000052_folium_map.html)
  - [`docs/maps/trip_00000000_folium_map.html`](docs/maps/trip_00000000_folium_map.html)
- **Evaluation Figures:** [`docs/figures/`](docs/figures/)
- **Member 2 Experiment Logs:** [`member2_results/`](member2_results/)
