"""
Jetson-friendly interface to:
1) Run ONNX inference,
2) Capture an image from the camera,
3) Optionally generate Grad-CAM with PyTorch weights,
4) Optionally fetch food suggestions for predicted deficiencies.

Typical Jetson camera run (ONNX + Grad-CAM + suggestions):
  python jetson_deploy/onnx_inference.py --camera --model jetson_deploy/model.onnx \\
      --weights best_model.pth --classes jetson_deploy/classes.json --grad-cam \\
      --suggestions --api-key $GEMINI_API_KEY

Single image:
  python jetson_deploy/onnx_inference.py --image path/to/img.jpg --grad-cam --suggestions

Keys in camera UI:
  c : capture and run inference
  q : quit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import datasets, transforms

# Ensure repo root is on sys.path when running as a script from jetson_deploy/.
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Grad-CAM (PyTorch) imports are optional to allow ONNX-only use.
try:
    import torch
    from torch import nn

    from grad_cam import (
        GradCAM,
        build_model as build_torch_model,
        denormalize_to_uint8,
        overlay_heatmap,
    )

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    GradCAM = None  # type: ignore
    build_torch_model = None  # type: ignore
    denormalize_to_uint8 = None  # type: ignore
    overlay_heatmap = None  # type: ignore

try:
    from food_suggestions import (
        format_suggestions_output,
        get_deficiency_info,
        get_food_suggestions,
        initialize_ai_client,
    )

    FOOD_SUGGESTIONS_AVAILABLE = True
except Exception:
    FOOD_SUGGESTIONS_AVAILABLE = False

IMAGENET_MEAN: Sequence[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: Sequence[float] = [0.229, 0.224, 0.225]
DEFAULT_MODEL = Path("jetson_deploy/model.onnx")
DEFAULT_WEIGHTS = Path("best_model.pth")
DEFAULT_CLASSES = Path("jetson_deploy/classes.json")
DEFAULT_DATA_ROOT = Path("dataset/train")
SUGGESTION_THRESHOLD = 0.20
WINDOW_NAME = "Vitamin Deficiency Screening (Not diagnostic)"


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def select_providers() -> list[str]:
    preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    return [p for p in preferred if p in available]


def load_classes(classes_path: Optional[Path], fallback_train_root: Path) -> List[str]:
    if classes_path and classes_path.exists():
        return json.loads(classes_path.read_text())
    ds = datasets.ImageFolder(root=str(fallback_train_root))
    return ds.classes


def create_session(model_path: Path) -> Tuple[ort.InferenceSession, list[str], str, str]:
    providers = select_providers()
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, providers, input_name, output_name


def run_onnx_inference(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    pil_img: Image.Image,
    tfm: transforms.Compose,
    classes: List[str],
) -> Tuple[dict, np.ndarray]:
    x = tfm(pil_img).unsqueeze(0).numpy()
    logits = session.run([output_name], {input_name: x})[0]
    probs = softmax(logits)[0]
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    result = {
        "pred_class": classes[top_idx],
        "confidence": confidence,
        "probs": {cls: float(probs[i]) for i, cls in enumerate(classes)},
    }
    return result, probs


def load_torch_model(num_classes: int, weights_path: Path, device: torch.device) -> nn.Module:
    if not TORCH_AVAILABLE or GradCAM is None or build_torch_model is None:
        raise RuntimeError("PyTorch/Grad-CAM is not available but --grad-cam was requested.")
    model = build_torch_model(num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def build_grad_cam_overlay(
    pil_img: Image.Image, target_idx: int, model: nn.Module, device: torch.device
) -> np.ndarray:
    if not TORCH_AVAILABLE or GradCAM is None or denormalize_to_uint8 is None or overlay_heatmap is None:
        raise RuntimeError("Grad-CAM dependencies are missing.")
    transform = get_transform()
    tensor = transform(pil_img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    heatmap, _ = grad_cam.generate(tensor, target_category=target_idx)
    grad_cam.remove_hooks()

    base_img = denormalize_to_uint8(tensor[0])
    return overlay_heatmap(base_img, heatmap, alpha=0.5)


def maybe_get_food_suggestions(
    pred_class: str,
    confidence: float,
    threshold: float,
    api_key: Optional[str],
    ai_model,
) -> dict:
    if not FOOD_SUGGESTIONS_AVAILABLE:
        return {"food_suggestions_error": "food_suggestions dependency is missing"}

    deficiency_info = get_deficiency_info(pred_class)
    if deficiency_info.get("deficiency") is None:
        return {"food_suggestions_skipped": "no deficiency detected"}
    if confidence < threshold:
        return {"food_suggestions_skipped": f"confidence {confidence:.2f} < {threshold:.2f}"}

    try:
        suggestions_result = get_food_suggestions(pred_class, confidence, api_key=api_key, model=ai_model)
        return {
            "food_suggestions": suggestions_result["suggestions"],
            "deficiency_info": suggestions_result["deficiency_info"],
        }
    except Exception as exc:  # pragma: no cover - runtime API call
        return {"food_suggestions_error": str(exc)}


def pil_from_bgr(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def analyze_image(
    pil_img: Image.Image,
    image_id: Path,
    classes: List[str],
    session: ort.InferenceSession,
    providers: list[str],
    input_name: str,
    output_name: str,
    tfm: transforms.Compose,
    want_grad_cam: bool,
    grad_cam_dir: Optional[Path],
    torch_model: Optional[nn.Module],
    device: Optional[torch.device],
    want_suggestions: bool,
    suggestion_threshold: float,
    api_key: Optional[str],
    ai_model,
) -> Tuple[dict, Optional[np.ndarray]]:
    result, probs = run_onnx_inference(session, input_name, output_name, pil_img, tfm, classes)
    result["image"] = str(image_id)
    result["providers"] = providers

    grad_cam_img = None
    if want_grad_cam:
        if not TORCH_AVAILABLE or torch_model is None or device is None:
            result["grad_cam"] = None
        else:
            grad_cam_img = build_grad_cam_overlay(pil_img, classes.index(result["pred_class"]), torch_model, device)
            if grad_cam_dir:
                grad_cam_dir.mkdir(parents=True, exist_ok=True)
                out_path = grad_cam_dir / f"{image_id.stem}_gradcam.jpg"
            else:
                out_path = image_id.with_name(image_id.stem + "_gradcam.jpg")
            cv2.imwrite(str(out_path), grad_cam_img)
            result["grad_cam"] = str(out_path)

    if want_suggestions:
        suggestion_info = maybe_get_food_suggestions(
            result["pred_class"], result["confidence"], suggestion_threshold, api_key, ai_model
        )
        result.update(suggestion_info)

    return result, grad_cam_img


def default_csi_pipeline(width: int = 1280, height: int = 720, fps: int = 30) -> str:
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )


def overlay_label(frame: np.ndarray, text: str, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Press c to capture | q to quit | Not diagnostic",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def run_camera_interface(
    args: argparse.Namespace,
    classes: List[str],
    session: ort.InferenceSession,
    providers: list[str],
    input_name: str,
    output_name: str,
    torch_model: Optional[nn.Module],
    device: Optional[torch.device],
    tfm: transforms.Compose,
    ai_model,
) -> None:
    if args.gstreamer:
        cap = cv2.VideoCapture(args.gstreamer, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        raise SystemExit("Camera not opened. Check pipeline/index and permissions.")

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Camera ready. Press 'c' to capture, 'q' to quit. Window title reminder: Not diagnostic.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from camera.")
            break

        display = overlay_label(frame.copy(), "Ready to capture")
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            capture_path = save_dir / f"capture_{timestamp}.jpg"
            cv2.imwrite(str(capture_path), frame)

            pil_img = pil_from_bgr(frame)
            result, grad_cam_img = analyze_image(
                pil_img=pil_img,
                image_id=capture_path,
                classes=classes,
                session=session,
                providers=providers,
                input_name=input_name,
                output_name=output_name,
                tfm=tfm,
                want_grad_cam=args.grad_cam,
                grad_cam_dir=save_dir,
                torch_model=torch_model,
                device=device,
                want_suggestions=args.suggestions,
                suggestion_threshold=args.suggestion_threshold,
                api_key=args.api_key,
                ai_model=ai_model,
            )

            summary = {
                "image": result["image"],
                "pred_class": result["pred_class"],
                "confidence": result["confidence"],
                "providers": result["providers"],
                "grad_cam": result.get("grad_cam"),
                "food_suggestions": bool(result.get("food_suggestions")),
                "food_suggestions_error": result.get("food_suggestions_error"),
                "food_suggestions_skipped": result.get("food_suggestions_skipped"),
            }
            print(json.dumps(summary, indent=2))

            if result.get("food_suggestions"):
                print(
                    format_suggestions_output(
                        {
                            "suggestions": result["food_suggestions"],
                            "deficiency_info": result["deficiency_info"],
                        }
                    )
                )
            elif result.get("food_suggestions_error"):
                print(f"Food suggestions error: {result['food_suggestions_error']}")
            elif result.get("food_suggestions_skipped"):
                print(f"Food suggestions skipped: {result['food_suggestions_skipped']}")

            overlay_img = grad_cam_img if grad_cam_img is not None else frame.copy()
            label_text = f"{result['pred_class']} ({result['confidence']:.2f})"
            cv2.putText(overlay_img, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(
                overlay_img,
                "Not diagnostic; consult a clinician",
                (20, overlay_img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Last Capture (with Grad-CAM if enabled)", overlay_img)
            cv2.waitKey(200)  # brief pause so the overlay is visible

    cap.release()
    cv2.destroyAllWindows()


def run_single_image(
    args: argparse.Namespace,
    classes: List[str],
    session: ort.InferenceSession,
    providers: list[str],
    input_name: str,
    output_name: str,
    torch_model: Optional[nn.Module],
    device: Optional[torch.device],
    tfm: transforms.Compose,
    ai_model,
) -> None:
    pil_img = Image.open(args.image).convert("RGB")
    grad_dir = args.save_dir if args.grad_cam else None
    result, _ = analyze_image(
        pil_img=pil_img,
        image_id=args.image,
        classes=classes,
        session=session,
        providers=providers,
        input_name=input_name,
        output_name=output_name,
        tfm=tfm,
        want_grad_cam=args.grad_cam,
        grad_cam_dir=grad_dir,
        torch_model=torch_model,
        device=device,
        want_suggestions=args.suggestions,
        suggestion_threshold=args.suggestion_threshold,
        api_key=args.api_key,
        ai_model=ai_model,
    )

    print(json.dumps(result, indent=2))
    if result.get("food_suggestions"):
        print(
            format_suggestions_output(
                {
                    "suggestions": result["food_suggestions"],
                    "deficiency_info": result["deficiency_info"],
                }
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jetson ONNX inference with optional camera capture, Grad-CAM, and food suggestions."
    )
    parser.add_argument("--image", type=Path, help="Image to classify (skip for camera).")
    parser.add_argument("--camera", action="store_true", help="Open camera interface to capture and classify.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for USB webcams (default: 0).")
    parser.add_argument(
        "--gstreamer",
        type=str,
        default=None,
        help="Custom GStreamer pipeline (overrides --camera-index). Use for CSI camera: "
        f"{default_csi_pipeline()}",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to ONNX model.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="PyTorch weights for Grad-CAM.")
    parser.add_argument("--grad-cam", action="store_true", help="Generate Grad-CAM using PyTorch weights.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("captures"),
        help="Folder to save captures and Grad-CAM outputs (camera mode also saves raw frames).",
    )
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES, help="Optional class list JSON.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Fallback train root to read class names if classes.json is missing.",
    )
    parser.add_argument(
        "--suggestions",
        action="store_true",
        help="Fetch food suggestions for deficiency predictions (requires GEMINI_API_KEY or --api-key).",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key (else use GEMINI_API_KEY env var).")
    parser.add_argument(
        "--suggestion-threshold",
        type=float,
        default=SUGGESTION_THRESHOLD,
        help="Minimum confidence to trigger food suggestions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")
    if not args.camera and not args.image:
        raise SystemExit("Provide --camera or --image.")
    if args.image and not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.grad_cam and not args.weights.exists():
        raise FileNotFoundError(f"PyTorch weights for Grad-CAM not found: {args.weights}")

    classes = load_classes(args.classes, args.data_root)
    session, providers, input_name, output_name = create_session(args.model)
    transform = get_transform()

    device = None
    torch_model = None
    if args.grad_cam:
        if not TORCH_AVAILABLE:
            raise SystemExit("PyTorch/Grad-CAM not available. Install torch/torchvision to enable --grad-cam.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model = load_torch_model(len(classes), args.weights, device)

    ai_model = None
    if args.suggestions and FOOD_SUGGESTIONS_AVAILABLE:
        ai_model = initialize_ai_client(args.api_key)
    elif args.suggestions and not FOOD_SUGGESTIONS_AVAILABLE:
        raise SystemExit(
            "Food suggestions requested but dependencies are missing. Install google-generativeai to enable it."
        )

    if args.camera:
        run_camera_interface(
            args=args,
            classes=classes,
            session=session,
            providers=providers,
            input_name=input_name,
            output_name=output_name,
            torch_model=torch_model,
            device=device,
            tfm=transform,
            ai_model=ai_model,
        )
    else:
        run_single_image(
            args=args,
            classes=classes,
            session=session,
            providers=providers,
            input_name=input_name,
            output_name=output_name,
            torch_model=torch_model,
            device=device,
            tfm=transform,
            ai_model=ai_model,
        )


if __name__ == "__main__":
    main()
