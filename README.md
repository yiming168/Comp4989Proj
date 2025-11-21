# Comp4989Proj
Group1 nutrition deficiency detection model

export: python jetson_deploy/export_onnx.py --weights best_model.pth --out jetson_deploy/model.onnx.
After export completes:

Copy jetson_deploy/model.onnx and jetson_deploy/classes.json to the Jetson.
On Jetson install runtime deps (match JetPack): python -m pip install onnxruntime-gpu==1.16.1 Pillow numpy (plus PyTorch/torchvision if you want native inference too).
Run ONNX inference on Jetson: python jetson_deploy/onnx_inference.py --image /path/to/img.jpg --model model.onnx.
If onnxscript wheels lag for Python 3.13, use Python 3.10/3.11 for export, then copy the ONNX file to Jetson.
