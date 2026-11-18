# Laptop 1 model/API and Laptop 2 evaluator client

## Verified scope

Laptop 1 holds the exact selected untuned Random Forest and runs FastAPI. Laptop 2
runs a browser UI and sends requests over the shared LAN. Real ML scoring is wired
through Member 4's supplied preprocessing pipeline, the actual PathSnapModelRequest,
Member2Scorer and PathSnapModelResult. No fixture model or reference route is used.

The supplied project has a Member 3 decoder **interface**, not its implementation.
This demo therefore reports `hmm_viterbi: not_connected` and does not claim a
reconstructed route or route accuracy. The separate diagnostic client is provided
because Member 4's supplied React UI currently targets the generic sequence API.
That UI and the supplied backend files have not been overwritten.

## Laptop 1 (this machine)

The server was started on port 8000. Check the existing server before launching a
second instance. Its currently detected Wi-Fi address is **10.0.13.39**; DHCP may
change it. Find the active address with `ipconfig`.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
Invoke-RestMethod http://10.0.13.39:8000/v1/pathsnap/trips
```

To start it in your own PowerShell terminal after stopping an existing instance:

```powershell
cd "C:\Users\kaush\Downloads\ML Hackathon\pathsnap-repo"
& "..\.venv\Scripts\python.exe" -m uvicorn laptop1_demo.server:app --host 0.0.0.0 --port 8000
```

Keep this terminal running; Ctrl+C stops that instance. `0.0.0.0` is the bind
address, not a URL to enter on Laptop 2. Interactive API documentation is at
`http://10.0.13.39:8000/docs` from the shared network.

Configuration variables (set before starting):

- `PATHSNAP_MODEL_PATH`: path to the original selected_model.joblib. Defaults to
  `backend/artifacts/member2/selected_model.joblib` when available, otherwise the original
  `../experiments/member2/models/selected_model.joblib` outside the checkout.
- `PATHSNAP_MODEL_SHA256`: checksum of that exact selected artifact. The default is
  the verified `b5533a8709b43a90e06376991b659edf848d188079cebcccee502ba203476eb4`.
- `PATHSNAP_TRIPS_DIR`: local data root containing .track/.nodes/.arcs files. The
  default is the existing student-test raw directory, with ten real student trips.
  The API never opens .route files. No evaluator archive was inspected or used.
- `API_CORS_ORIGINS`: comma-separated UI origins. Defaults are
  `http://localhost:5173,http://127.0.0.1:5173`, which work when Laptop 2 hosts its UI locally.

If creating a new environment, install `laptop1_demo/requirements.txt`. Do not copy
the ZIP's `.venv`: it was built for a different Python installation. The existing
project environment uses Python 3.13.7 and scikit-learn 1.7.2.

## Laptop 2

Transfer `PathSnap_Laptop2_Client.zip` separately; no model or GPS dataset belongs
on Laptop 2 for this client. Extract it, open a terminal in its folder, and run:

```powershell
python -m http.server 5173 --bind 127.0.0.1
```

Open `http://localhost:5173`, enter **http://10.0.13.39:8000** as Laptop 1's API URL,
then choose **Connect & load trips**, select a trip, and score it. The table displays
the first 200 independent observation predictions; full scores remain in the API response.
Do not use `localhost:8000` on Laptop 2: that would refer to Laptop 2 itself.

## Network verification on the actual second laptop

Both laptops must join a network that permits peer-to-peer traffic. On Laptop 2:

```powershell
Test-NetConnection 10.0.13.39 -Port 8000
Invoke-RestMethod http://10.0.13.39:8000/healthz
```

Require `TcpTestSucceeded: True` and `model_loaded: true`. A successful request to
this machine's own LAN address does not establish connectivity from a second machine.
Physical Laptop 2 verification remains pending until these commands succeed there.

If blocked, check the IP, Wi-Fi/client isolation and Windows Firewall. For a trusted
private LAN, a narrow rule can be added manually in administrator PowerShell:

```powershell
New-NetFirewallRule -DisplayName "PathSnap demo API 8000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -RemoteAddress LocalSubnet -Profile Private
```

No firewall settings have been changed by this task. The rule above only applies
to a Windows network marked Private; verify the network profile yourself. A campus
SSID may isolate clients even when both laptops are connected. Both laptops should
have chargers and remain awake throughout the demo.

## Endpoints

- `GET /healthz`: real model readiness and checksum.
- `GET /v1/pathsnap/capabilities`: ML ready, HMM explicitly not connected.
- `GET /v1/pathsnap/trips`: actual server-side trip IDs.
- `POST /v1/pathsnap/trips/{trip_id}/score`: real stored GPS/network â†’ backend preparation â†’ typed scorer.
- `POST /v1/pathsnap/score`: the supplied PathSnapPreprocessRequest GPS/network JSON contract.
  Optional reference-route IDs are ignored, never forwarded into inference.

These scoring routes are the new standalone demo composition, not an undocumented
change to Member 4's existing `/v1/pathsnap/preprocess` endpoint. Start
`laptop1_demo.server:app`, not the unmodified `app.main:app`, to serve them.

## Testing

```powershell
cd "C:\Users\kaush\Downloads\ML Hackathon\pathsnap-repo"
$env:PYTHONPATH="backend;pathsnap"
& "..\.venv\Scripts\python.exe" -m pytest backend/tests pathsnap/tests laptop1_demo/test_server.py -q --basetemp "../.runtime-temp/pytest-demo-manual"
```

Use a normal local PowerShell terminal for HTTP tests. The restricted execution
sandbox hung when entering the ASGI test client; the tests run successfully outside
that sandbox. Windows PowerShell HTTP probes also succeeded on localhost and the
Wi-Fi address. Python httpx TCP probes were aborted by the host; that failure is
recorded rather than being counted as a successful LAN test.

No commits, merges, uploads or pushes have been made during this integration work.
