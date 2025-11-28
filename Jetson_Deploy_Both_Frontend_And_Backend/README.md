# Jetson Deploy: Backend + Frontend

This folder bundles instructions and a tiny static server to run both the FastAPI backend and the built React frontend on a Jetson device.

## Prerequisites
- Jetson device with Python 3.8+.
- `best_model.pth` present in repo root (already included here).
- `dataset/train` present for class names (read-only ok).
- Backend deps: `pip install -r requirements.txt` plus Jetson-appropriate torch/torchvision wheels (see below).
- Frontend build deps: Node.js (18.x recommended) if building on-device; otherwise build elsewhere and copy `frontend/dist` to the Jetson.
- Optional Gemini suggestions: set `GEMINI_API_KEY` or pass via API requests.

## Backend setup (FastAPI + uvicorn)
```bash
cd /path/to/Comp4989Proj
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install dependencies
python -m pip install -r requirements.txt
# On Jetson, install the matching torch/torchvision wheels (example; adjust for your JetPack/Python):
# python -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
# Run the API
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

## Frontend build and serve
Build on Jetson (or build elsewhere and copy `frontend/dist`):
```bash
cd /path/to/Comp4989Proj/frontend
export VITE_API_URL="http://<jetson-ip>:8000"
npm ci  # or npm install
npm run build
```
Serve the static build on Jetson (Python-based, no Node required to serve):
```bash
cd /path/to/Comp4989Proj/frontend
python ../Jetson_Deploy_Both_Frontend_And_Backend/serve_static.py --dir dist --port 4173
```
If you prefer Node: `npx serve -s dist -l 4173`.

## Quick combined test
1) Start backend: `uvicorn api:app --host 0.0.0.0 --port 8000`
2) Serve frontend: `python Jetson_Deploy_Both_Frontend_And_Backend/serve_static.py --dir frontend/dist --port 4173`
3) On a laptop/phone, open `http://<jetson-ip>:4173` and run a prediction.

## Systemd templates (edit paths, then place in /etc/systemd/system/)
### Backend service (fastapi-jetson.service)
```
[Unit]
Description=FastAPI vitamin deficiency backend
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/jetson/Comp4989Proj
ExecStart=/home/jetson/Comp4989Proj/.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
Environment=GEMINI_API_KEY=

[Install]
WantedBy=multi-user.target
```

### Frontend service (frontend-static.service)
```
[Unit]
Description=Static frontend server
After=network.target fastapi-jetson.service

[Service]
Type=simple
WorkingDirectory=/home/jetson/Comp4989Proj/frontend
ExecStart=/usr/bin/python3 /home/jetson/Comp4989Proj/Jetson_Deploy_Both_Frontend_And_Backend/serve_static.py --dir dist --port 4173
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable + start: `sudo systemctl enable --now fastapi-jetson.service frontend-static.service`

## Notes / Caveats
- Camera features in the web UI may require HTTPS or localhost for permissions; for LAN IPs you might need to allow insecure context or set up TLS/ingress.
- Grad-CAM and suggestions depend on the backend having OpenCV (`opencv-python`) and the Gemini key. Frontend just forwards toggles.
- This remains a research demo and is not a diagnostic tool.
