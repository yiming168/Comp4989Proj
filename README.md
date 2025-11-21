# Comp4989Proj

Visual screening of vitamin deficiency¨Crelated symptoms. PyTorch MobileNetV3 model (ImageNet normalization, weighted cross entropy, macro-F1 selection), FastAPI service, and a React (Vite) web UI. Data comes from inal_dataset/ (read-only) and is split into dataset/train|val|test/.

## Quick start
1) Split data (copies from inal_dataset/):

`ash
python split_dataset.py
`

2) Train model (saves est_model.pth):

`ash
python train.py
`

3) Run API (serves predictions, Grad-CAM, optional Gemini food suggestions):

`ash
uvicorn api:app --reload --port 8000
# set GEMINI_API_KEY for suggestions (or pass api_key form field)
`

4) Run web UI (Vite + React):

`ash
cd frontend
npm install
npm run dev
# set VITE_API_URL if API is not on http://localhost:8000
`

## CLI utilities
- Inference with Grad-CAM & suggestions:

`ash
python inference.py --image path/to/img.jpg --grad-cam --suggestions
`

- Grad-CAM only:

`ash
python grad_cam.py --image test/image.png --weights best_model.pth --data_root dataset
`

- Food suggestions prompt demo:

`ash
python test_food_suggestions.py
`

## API (FastAPI)
- GET /health ¡ú status, device, classes, suggestions availability
- POST /predict ¡ú form-data: ile (image), optional grad_cam (bool), suggestions (bool), 	hreshold (float, default 0.20), pi_key (Gemini override). Returns predicted class, confidence, probs, optional Grad-CAM data URL, and Gemini suggestions/metadata when enabled.

## Frontend (Vite + React)
- rontend/ contains the web client: upload or camera capture, toggles for Grad-CAM and suggestions, renders probabilities, Grad-CAM overlay, and Gemini food recommendations.
- Configure API base via VITE_API_URL; defaults to http://localhost:8000.

## Jetson / ONNX
Export and run on Jetson with ONNX Runtime:

`ash
python jetson_deploy/export_onnx.py --weights best_model.pth --out jetson_deploy/model.onnx
python jetson_deploy/onnx_inference.py --image /path/to/img.jpg --model jetson_deploy/model.onnx
`

If onnxruntime-gpu wheels lag for newer Python, export on Python 3.10/3.11 and copy the ONNX file to Jetson. Tensorrt/CUDA providers are preferred when available; the script falls back to CPU otherwise.

## Ethics notice
This is a research/demo tool and not a medical device. Outputs should not be used for diagnosis; consult a healthcare professional for medical advice.
