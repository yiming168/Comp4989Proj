"""
FastAPI service exposing the nutrition deficiency classifier with optional Grad-CAM and Gemini food suggestions.
Run: uvicorn api:app --reload --port 8000
"""

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torch import nn
from torchvision import datasets, models, transforms

from grad_cam import GradCAM, denormalize_to_uint8, overlay_heatmap

try:
    from food_suggestions import (
        get_deficiency_info,
        get_food_suggestions,
        initialize_ai_client,
    )

    FOOD_SUGGESTIONS_AVAILABLE = True
except Exception:
    FOOD_SUGGESTIONS_AVAILABLE = False

DATA_ROOT = Path("dataset/train")  # only for class names
WEIGHTS = Path("best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUGGESTION_THRESHOLD = 0.20

app = FastAPI(
    title="Nutrition Deficiency Classifier",
    description="Image classifier with optional Grad-CAM and Gemini food suggestions.",
    version="0.1.0",
)

# Open up CORS for local dev/frontend; tighten if deploying
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

eval_tfm = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_classes(data_root: Path) -> List[str]:
    if not data_root.exists():
        raise FileNotFoundError(f"Train data root not found: {data_root}")
    ds = datasets.ImageFolder(root=str(data_root))
    return ds.classes


def load_model(class_names: List[str]) -> nn.Module:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")
    model = build_model(len(class_names))
    state = torch.load(WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def predict_pil(model: nn.Module, class_names: List[str], pil_img: Image.Image) -> Dict:
    x = eval_tfm(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    top_idx = int(torch.argmax(probs).item())
    confidence = float(probs[top_idx].item())
    return {
        "pred_class": class_names[top_idx],
        "confidence": confidence,
        "probs": {cls: float(probs[i]) for i, cls in enumerate(class_names)},
        "embedding": x,
    }


def make_grad_cam(model: nn.Module, class_names: List[str], embedding: torch.Tensor, target_idx: int) -> str:
    # embedding is the transformed tensor [1, 3, H, W]
    embedding = embedding.clone().detach().to(DEVICE)
    embedding.requires_grad_(True)

    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    heatmap, _ = grad_cam.generate(embedding, target_category=target_idx)
    grad_cam.remove_hooks()

    base_img = denormalize_to_uint8(embedding[0])
    overlay = overlay_heatmap(base_img, heatmap, alpha=0.5)
    ok, buf = cv2.imencode(".jpg", overlay)  # type: ignore[name-defined]
    if not ok:
        raise RuntimeError("Failed to encode Grad-CAM image")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


@app.on_event("startup")
async def startup() -> None:
    # Lazy import for cv2 to keep import cost minimal if not needed
    global cv2  # noqa: PLW0603
    import cv2 as _cv2

    globals()["cv2"] = _cv2
    app.state.class_names = load_classes(DATA_ROOT)
    app.state.model = load_model(app.state.class_names)
    app.state.gemini_model = None
    print("Loaded model with classes:", app.state.class_names)


@app.get("/health")
async def health() -> Dict:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "classes": app.state.class_names,
        "suggestions_enabled": FOOD_SUGGESTIONS_AVAILABLE,
    }


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    grad_cam: bool = Form(False),
    suggestions: bool = Form(True),
    threshold: float = Form(SUGGESTION_THRESHOLD),
    api_key: Optional[str] = Form(None),
) -> Dict:
    content = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = predict_pil(app.state.model, app.state.class_names, pil_img)
    result_payload: Dict[str, object] = {
        "image_name": file.filename,
        "pred_class": result["pred_class"],
        "confidence": result["confidence"],
        "probs": result["probs"],
        "device": str(DEVICE),
    }

    if grad_cam:
        target_idx = app.state.class_names.index(result["pred_class"])
        try:
            result_payload["grad_cam_image"] = make_grad_cam(
                app.state.model, app.state.class_names, result["embedding"], target_idx
            )
        except Exception as e:  # pragma: no cover - runtime guard
            result_payload["grad_cam_error"] = str(e)

    if suggestions and FOOD_SUGGESTIONS_AVAILABLE:
        deficiency_info = get_deficiency_info(result["pred_class"])
        meets_threshold = result["confidence"] >= threshold
        has_deficiency = deficiency_info.get("deficiency") is not None
        if meets_threshold and has_deficiency:
            if app.state.gemini_model is None:
                try:
                    app.state.gemini_model = initialize_ai_client(api_key)
                except Exception as e:
                    result_payload["food_suggestions_error"] = f"Gemini init failed: {e}"
            if app.state.gemini_model is not None:
                try:
                    suggestions_result = get_food_suggestions(
                        result["pred_class"],
                        result["confidence"],
                        api_key=api_key,
                        model=app.state.gemini_model,
                    )
                    result_payload["food_suggestions"] = suggestions_result["suggestions"]
                    result_payload["deficiency_info"] = suggestions_result["deficiency_info"]
                except Exception as e:
                    result_payload["food_suggestions_error"] = str(e)
        else:
            skip_reason = []
            if not meets_threshold:
                skip_reason.append(
                    f"confidence {result['confidence']:.2f} < threshold {threshold:.2f}"
                )
            if not has_deficiency:
                skip_reason.append("no deficiency detected")
            result_payload["food_suggestions_skipped"] = ", ".join(skip_reason)

    return result_payload


@app.get("/")
async def root() -> Dict:
    return {
        "message": "Nutrition classifier API. POST /predict with a file to get a prediction.",
        "health": "/health",
    }
