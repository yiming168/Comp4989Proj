# Comp4989Proj
Group1 nutrition deficiency detection model

 python inference.py --image test\Scurvy_06.jpeg --suggestions --grad-cam 

export: python jetson_deploy/export_onnx.py --weights best_model.pth --out jetson_deploy/model.onnx.
After export completes:

Copy jetson_deploy/model.onnx and jetson_deploy/classes.json to the Jetson.
On Jetson install runtime deps (match JetPack): python -m pip install onnxruntime-gpu==1.16.1 Pillow numpy (plus PyTorch/torchvision if you want native inference too).
Run ONNX inference on Jetson: python jetson_deploy/onnx_inference.py --image /path/to/img.jpg --model model.onnx.
If onnxscript wheels lag for Python 3.13, use Python 3.10/3.11 for export, then copy the ONNX file to Jetson.


Here’s the quickest way to run your ONNX model on a Jetson using the provided jetson_deploy/onnx_inference.py:

Copy artifacts to the Jetson
jetson_deploy/model.onnx
jetson_deploy/classes.json
jetson_deploy/onnx_inference.py
(You don’t need the whole dataset for inference.)
Install dependencies on the Jetson (match your JetPack/Python; 3.8–3.10 usually safest)
Update pip if needed: python -m pip install --upgrade pip
Install runtime packages (GPU build):
python -m pip install onnxruntime-gpu==1.16.1 numpy pillow torchvision
If you don’t need PyTorch there, you can skip torch/torchvision.
If onnxruntime-gpu for your Python/JetPack isn’t available, try onnxruntime==1.16.1 (CPU) or use a matching wheel from NVIDIA’s index.
Run inference
python onnx_inference.py --image /path/to/img.jpg --model model.onnx
The script auto-reads classes.json; it will print which providers are used (it prefers TensorRT → CUDA → CPU). If TensorRT is installed and supported, you’ll see TensorrtExecutionProvider; otherwise CUDAExecutionProvider or CPUExecutionProvider.
Verify providers
Optional quick check:
python - <<'PY'\nimport onnxruntime as ort; print(ort.get_available_providers())\nPY
If TensorRT/CUDA don’t appear, verify you installed the GPU wheel and that the Jetson CUDA/TensorRT runtimes are available.
If you need Grad-CAM on Jetson
That uses PyTorch weights, so copy best_model.pth and run your grad_cam.py there with PyTorch installed. ONNX Runtime alone won’t generate Grad-CAM.
That’s it—once dependencies are installed and the files are in place, the single command in step 3 runs your model on the Jetson.