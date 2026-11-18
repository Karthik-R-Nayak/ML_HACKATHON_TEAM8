import zipfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
server_py = (ROOT / 'laptop1_demo/server.py').read_text(encoding='utf-8')
start = server_py.find('CLIENT_HTML = """') + len('CLIENT_HTML = """')
end = server_py.find('"""\n\n\ndef create_app', start)
html = server_py[start:end]

readme = """# PathSnap • Laptop 2 Evaluator Client

## Quick Start on Laptop 2:

1. Open PowerShell / Terminal in this folder.
2. Start the local client server:
   python -m http.server 5173 --bind 127.0.0.1

3. Open your browser:
   http://localhost:5173

4. In the top connection bar, enter Laptop 1's API address (e.g. http://10.0.13.39:8000) and click 'Connect'.
5. Select any of the 10 evaluation trips and click 'Run Route Matching (HMM)'!

Note: You can also open http://<laptop1-ip>:8000/ directly in any browser on Laptop 2 if connected to the same Wi-Fi/LAN.
"""

client_dir = ROOT.parent / "client_dist"
client_dir.mkdir(exist_ok=True)
(client_dir / "index.html").write_text(html, encoding="utf-8")
(client_dir / "README.txt").write_text(readme, encoding="utf-8")

zip_path = ROOT.parent / "PathSnap_Laptop2_Client.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(client_dir / "index.html", "index.html")
    z.write(client_dir / "README.txt", "README.txt")

print(f"Updated {zip_path} successfully! Size: {os.path.getsize(zip_path)} bytes")
