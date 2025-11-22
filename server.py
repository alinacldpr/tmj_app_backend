import os
import base64
import cv2
import numpy as np
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# ---------- config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "runs/segment/train/weights/best.pt")
DEVICE = 0 if os.getenv("CUDA", "0") == "1" else "cpu"  # set CUDA=1 to force GPU

# ---------- app ----------
app = FastAPI(title="TMJ YOLO Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten to ["https://alinacldpr.github.io"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- model (load once) ----------
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO weights at {MODEL_PATH}: {e}")

# ---------- helpers ----------
import base64, urllib.parse, numpy as np, cv2

def data_url_to_bgr(data_url: str) -> np.ndarray:
    """
    Accepts either a full data URL (data:image/jpeg;base64,....) or a raw base64 string.
    Cleans whitespace, fixes missing padding, and decodes to BGR image.
    """
    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    # Some tools URL-encode or insert spaces/newlines
    payload = urllib.parse.unquote(payload)          # undo %2B, %2F, etc.
    payload = payload.replace(" ", "+").replace("\n", "").replace("\r", "")
    # Fix missing padding
    missing = (-len(payload)) % 4
    if missing:
        payload += "=" * missing
    try:
        buf = base64.b64decode(payload, validate=False)
    except Exception as e:
        raise ValueError(f"Invalid base64: {e}")
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image (unsupported/bad format).")
    return img


def bgr_to_png_data_url(img_bgr: np.ndarray) -> str:
    ok, enc = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode PNG.")
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def largest_run_1d(row: np.ndarray):
    """Return (start, end) of the longest continuous run of 1s in a 1D array."""
    best_s, best_e, best_len = None, None, 0
    s, cnt = None, 0
    for i, v in enumerate(row):
        if v:
            if s is None:
                s = i
            cnt += 1
        else:
            if s is not None and cnt > best_len:
                best_s, best_e, best_len = s, i - 1, cnt
            s, cnt = None, 0
    if s is not None and cnt > best_len:
        best_s, best_e, best_len = s, len(row) - 1, cnt
    return best_s, best_e

def get_masks(result):
    """Combine instance masks into two binary masks: upper teeth (mu) and lower teeth (ml)."""
    H, W = result.orig_img.shape[:2]
    mu, ml = None, None
    if result.masks is None:
        return None, None
    for i, b in enumerate(result.boxes):
        cls = int(b.cls[0])
        name = result.names[cls].lower()
        m = result.masks.data[i].cpu().numpy()
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = (m > 0.5).astype(np.uint8)
        if "upper" in name:
            mu = m if mu is None else np.maximum(mu, m)
        elif "lower" in name:
            ml = m if ml is None else np.maximum(ml, m)
    return mu, ml

# tune as needed
MIDLINE_FRACTION = 0.35
EDGE_BAND = 3

def compute_gap_scale_midpoint(mask_u, mask_l, known_mm):
    """Compute AB gap in px, estimate px/mm using centrals, and choose x_mid."""
    H, W = mask_u.shape
    cols_u = np.where(mask_u.sum(axis=0) > 0)[0]
    cols_l = np.where(mask_l.sum(axis=0) > 0)[0]
    if cols_u.size == 0 or cols_l.size == 0:
        return None, None, None, None, None, None

    cmin = max(cols_u.min(), cols_l.min())
    cmax = min(cols_u.max(), cols_l.max())
    width = cmax - cmin + 1
    if width <= 5:
        return None, None, None, None, None, None

    # keep middle region (avoid corners)
    c0 = int(cmin + (1 - MIDLINE_FRACTION) / 2 * width)
    c1 = int(cmax - (1 - MIDLINE_FRACTION) / 2 * width)
    c0, c1 = max(0, c0), min(W - 1, c1)
    if c1 <= c0:
        return None, None, None, None, None, None

    U = mask_u[:, c0:c1]
    L = mask_l[:, c0:c1]

    ys_u = [np.where(U[:, x])[0].max() for x in range(U.shape[1]) if np.where(U[:, x])[0].size]
    ys_l = [np.where(L[:, x])[0].min() for x in range(L.shape[1]) if np.where(L[:, x])[0].size]
    if not ys_u or not ys_l:
        return None, None, None, None, None, None

    y_u = int(np.median(ys_u))
    y_l = int(np.median(ys_l))
    gap_px = y_l - y_u

    # estimate centrals width near upper edge
    y0 = max(0, y_u - EDGE_BAND)
    y1 = min(H - 1, y_u + EDGE_BAND)
    band = U[y0:y1 + 1, :]
    row = (band.sum(axis=0) > 0).astype(np.uint8)
    sx, ex = largest_run_1d(row)

    px_per_mm = None
    span = None
    x_mid = (c0 + c1) // 2
    if sx is not None:
        centrals_px = ex - sx + 1
        px_per_mm = (centrals_px / float(known_mm)) if known_mm else None
        span = (c0 + sx, c0 + ex)
        x_mid = (span[0] + span[1]) // 2

    return gap_px, y_u, y_l, px_per_mm, span, x_mid

# ---------- request/response ----------
class AnalyzePayload(BaseModel):
    image_data_url: str                       # data URL from the browser (image/jpeg|png base64)
    known_centrals_mm: Optional[float] = None
    conf: Optional[float] = 0.35
    iou: Optional[float] = 0.5

@app.get("/health")
def health():
    return {"ok": True, "model": os.path.basename(MODEL_PATH)}

@app.post("/analyze")
def analyze(p: AnalyzePayload):
    img_bgr = data_url_to_bgr(p.image_data_url)

    # inference
    r = model.predict(
        source=img_bgr,
        imgsz=640,
        conf=p.conf or 0.35,
        iou=p.iou or 0.5,
        show=False,
        save=False,
        device=DEVICE,
        verbose=False,
    )[0]

    mu, ml = get_masks(r)
    if mu is None or ml is None:
        return {"ok": False, "error": "No masks detected."}

    # overlay rendered by Ultralytics
    verif_bgr = r.plot()

    # combined binary
    binary = np.maximum(mu, ml) * 255
    binary_bgr = cv2.cvtColor(binary.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # measurement
    gap_px, y_u, y_l, px_per_mm, span, x_mid = compute_gap_scale_midpoint(
        mu, ml, p.known_centrals_mm or 0.0
    )

    # measured visualization
    overlay = img_bgr.copy()
    overlay[mu.astype(bool)] = (
        0.5 * overlay[mu.astype(bool)] + 0.5 * np.array([255, 255, 0])
    ).astype(np.uint8)
    overlay[ml.astype(bool)] = (
        0.5 * overlay[ml.astype(bool)] + 0.5 * np.array([255, 0, 255])
    ).astype(np.uint8)
    measured = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)

    cv2.line(measured, (0, y_u), (measured.shape[1], y_u), (0, 255, 255), 2)
    cv2.line(measured, (0, y_l), (measured.shape[1], y_l), (255, 0, 255), 2)
    ptB = (int(x_mid), int(y_u))
    ptA = (int(x_mid), int(y_l))
    cv2.line(measured, ptB, ptA, (0, 255, 0), 3)
    cv2.circle(measured, ptB, 6, (0, 180, 255), -1)
    cv2.putText(
        measured, "B", (ptB[0] + 6, ptB[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA
    )
    cv2.circle(measured, ptA, 6, (255, 180, 0), -1)
    cv2.putText(
        measured, "A", (ptA[0] + 6, ptA[1] + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2, cv2.LINE_AA
    )

    label = f"AB: {gap_px}px"
    if px_per_mm:
        label += f" ({gap_px/px_per_mm:.2f} mm) | scale {px_per_mm:.2f} px/mm"
    cv2.putText(
        measured, label, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 220, 60), 2, cv2.LINE_AA
    )

    return {
        "ok": True,
        "gap_px": int(gap_px),
        "gap_mm": None if not px_per_mm else float(gap_px / px_per_mm),
        "px_per_mm": None if not px_per_mm else float(px_per_mm),
        "y_u": int(y_u), "y_l": int(y_l), "x_mid": int(x_mid),
        "images": {
            "original": bgr_to_png_data_url(img_bgr),
            "overlay":  bgr_to_png_data_url(verif_bgr),
            "binary":   bgr_to_png_data_url(binary_bgr),
            "measured": bgr_to_png_data_url(measured),
        },
    }
