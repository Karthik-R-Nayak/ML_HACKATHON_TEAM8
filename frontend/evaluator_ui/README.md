# PathSnap • Laptop 2 Evaluator Client

## Quick Start on Laptop 2:

1. Open PowerShell / Terminal in this folder.
2. Start the local client server:
   python -m http.server 5173 --bind 127.0.0.1

3. Open your browser:
   http://localhost:5173

4. In the top connection bar, enter Laptop 1's API address (e.g. http://10.0.13.39:8000) and click 'Connect'.
5. Select any of the 10 evaluation trips and click 'Run Route Matching (HMM)'!

Note: You can also open http://<laptop1-ip>:8000/ directly in any browser on Laptop 2 if connected to the same Wi-Fi/LAN.
