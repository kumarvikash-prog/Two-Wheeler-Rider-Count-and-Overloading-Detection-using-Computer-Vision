"""Core detection logic for overloaded vehicle detection using Haar Cascade classifiers.

Detection strategy (pure OpenCV, no external model download):
  1. PRIMARY: Frontal face cascade (haarcascade_frontalface_alt2) with well-tuned params
     → Each detected face is expanded into a full upper-body bounding box
  2. SECONDARY: Profile face cascade to catch side-facing riders
  3. Smart NMS removes boxes that substantially overlap each other
  4. No hardcoded person-count cap — count is purely face-detection driven
  5. Status is derived from the final detected count
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectorConfig:
    """Tuneable detection parameters."""

    resize_width:  int = 1280
    resize_height: int = 720

    # ── Frontal face cascade ────────────────────────────────────────────
    # min_size=50: removes spurious 45px detections on clothing/legs
    # min_neighbors=5: robust against background faces in cars etc.
    face_scale:         float = 1.04
    face_min_neighbors: int   = 5
    face_min_size:      Tuple[int, int] = (50, 50)
    face_max_size:      Tuple[int, int] = (280, 280)

    # ── Profile face cascade ────────────────────────────────────────────
    # min_neighbors=6: must be very confident before accepting a profile
    # (shoulder stripes can trigger profile at min_neighbors=5)
    profile_scale:         float = 1.04
    profile_min_neighbors: int   = 6
    profile_min_size:      Tuple[int, int] = (50, 50)
    profile_max_size:      Tuple[int, int] = (280, 280)

    # ── Face → body expansion ───────────────────────────────────────────
    body_width_mult:  float = 2.4
    body_height_mult: float = 3.6
    body_up_offset:   float = 0.15   # shift box upward by this fraction of face_h

    # ── NMS / dedup ─────────────────────────────────────────────────────
    merge_iou:    float = 0.25   # boxes with IoU > this are the same person
    coverage_thr: float = 0.55   # small box fully inside big box → suppress small

    # ── Spatial filters ─────────────────────────────────────────────────
    # Riders always occupy the top ~50% of the frame.
    # This cuts out false face detections on legs/feet in the lower half.
    face_roi_top:    float = 0.04
    face_roi_bottom: float = 0.52


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a; ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b; bx2, by2 = bx1 + bw, by1 + bh
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter == 0: return 0.0
    return inter / (aw * ah + bw * bh - inter)


def _coverage(small: Tuple[int, int, int, int], big: Tuple[int, int, int, int]) -> float:
    """What fraction of 'small' is covered by 'big'?"""
    sx1, sy1, sw, sh = small; sx2, sy2 = sx1 + sw, sy1 + sh
    bx1, by1, bw, bh = big;   bx2, by2 = bx1 + bw, by1 + bh
    ix = max(0, min(sx2, bx2) - max(sx1, bx1))
    iy = max(0, min(sy2, by2) - max(sy1, by1))
    area = sw * sh
    return (ix * iy) / area if area > 0 else 0.0


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def _nms(
    boxes: List[Tuple[int, int, int, int]],
    iou_thr: float,
    cov_thr: float,
) -> List[Tuple[int, int, int, int]]:
    """Greedy NMS: keep largest box, suppress overlapping/contained smaller boxes."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    suppressed = [False] * len(boxes)
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        kept.append(boxes[i])
        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            if (_iou(boxes[i], boxes[j]) > iou_thr or
                    _coverage(boxes[j], boxes[i]) > cov_thr):
                suppressed[j] = True
    return kept


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _load_cascade(fname: str) -> cv2.CascadeClassifier:
    path = os.path.join(cv2.data.haarcascades, fname)
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"Cannot load cascade: {path}")
    return clf


# ---------------------------------------------------------------------------
# Face detection + expansion
# ---------------------------------------------------------------------------

def _detect_faces(
    gray: np.ndarray,
    clf: cv2.CascadeClassifier,
    scale: float,
    min_neighbors: int,
    min_size: Tuple[int, int],
    max_size: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    raw = clf.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=min_neighbors,
        minSize=min_size,
        maxSize=max_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(raw) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in raw]


def _face_to_body(
    fx: int, fy: int, fw: int, fh: int,
    cfg: DetectorConfig,
    img_w: int, img_h: int,
) -> Tuple[int, int, int, int]:
    """Expand a face detection into an upper-body bounding box."""
    bw = int(fw * cfg.body_width_mult)
    bh = int(fh * cfg.body_height_mult)
    bx = fx + fw // 2 - bw // 2
    by = fy - int(fh * cfg.body_up_offset)
    bx = _clamp(bx, 0, img_w - 1)
    by = _clamp(by, 0, img_h - 1)
    bw = _clamp(bw, 1, img_w - bx)
    bh = _clamp(bh, 1, img_h - by)
    return bx, by, bw, bh


def _face_centre_in_roi(fx, fy, fw, fh, roi_top, roi_bot) -> bool:
    """Return True if the face centre falls inside the vertical ROI band."""
    cy = fy + fh / 2
    return roi_top <= cy <= roi_bot


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_people(
    image: np.ndarray,
    config: DetectorConfig,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect all on-vehicle people. Returns one (x, y, w, h) body box per person.
    Count is data-driven — no hardcoded maximum.
    """
    h, w = image.shape[:2]
    roi_top = int(h * config.face_roi_top)
    roi_bot = int(h * config.face_roi_bottom)

    # Pre-process: grayscale + CLAHE (better than plain equalizeHist for complex scenes)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    frontal_clf = _load_cascade("haarcascade_frontalface_alt2.xml")
    profile_clf = _load_cascade("haarcascade_profileface.xml")

    # ── Frontal faces ──────────────────────────────────────────────────
    frontal_faces = _detect_faces(
        gray, frontal_clf,
        config.face_scale, config.face_min_neighbors,
        config.face_min_size, config.face_max_size,
    )
    frontal_faces = [
        f for f in frontal_faces
        if _face_centre_in_roi(f[0], f[1], f[2], f[3], roi_top, roi_bot)
    ]
    # Deduplicate frontal faces first
    frontal_faces = _nms(frontal_faces, config.merge_iou, config.coverage_thr)

    # ── Profile faces ──────────────────────────────────────────────────
    profile_faces = _detect_faces(
        gray, profile_clf,
        config.profile_scale, config.profile_min_neighbors,
        config.profile_min_size, config.profile_max_size,
    )
    profile_faces = [
        f for f in profile_faces
        if _face_centre_in_roi(f[0], f[1], f[2], f[3], roi_top, roi_bot)
    ]

    # ── Merge: Keep frontal as authoritative, add profile only if no overlap ──
    # Use a LOW threshold here: even a small face-level overlap means same person.
    # (profile boxes are often larger than the frontal face they cover, so IoU
    # alone at 0.25 misses subtle overlaps — we use 0.10 to be safe)
    PROFILE_IOU_THR = 0.10
    PROFILE_COV_THR = 0.25
    final_faces = list(frontal_faces)
    for p_face in profile_faces:
        overlaps_any_frontal = any(
            _iou(p_face, ff) > PROFILE_IOU_THR or
            _coverage(p_face, ff) > PROFILE_COV_THR or
            _coverage(ff, p_face) > PROFILE_COV_THR
            for ff in frontal_faces
        )
        if not overlaps_any_frontal:
            final_faces.append(p_face)

    # ── Expand each face to a body box ────────────────────────────────
    body_boxes = [_face_to_body(fx, fy, fw, fh, config, w, h)
                  for fx, fy, fw, fh in final_faces]

    # ── Final NMS on body boxes ────────────────────────────────────────
    body_boxes = _nms(body_boxes, config.merge_iou, config.coverage_thr)

    # ── Sort left → right ─────────────────────────────────────────────
    body_boxes.sort(key=lambda b: b[0])
    return body_boxes


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_riding(person_count: int) -> str:
    if person_count == 0:
        return "No Rider Detected"
    elif person_count == 1:
        return "Single Rider"
    elif person_count == 2:
        return "Normal Riding (2 People)"
    elif person_count == 3:
        return "Triple Riding Detected"
    else:
        return f"Severely Overloaded ({person_count} People)"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def detect_triple_riding(
    image: np.ndarray,
    config: DetectorConfig | None = None,
) -> Dict[str, np.ndarray | int | str | List[Tuple[int, int, int, int]]]:
    cfg = config or DetectorConfig()
    resized = resize_image(image, cfg.resize_width, cfg.resize_height)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    boxes = detect_people(resized, cfg)

    # Fallback: if nothing found, relax thresholds once
    if len(boxes) == 0:
        relaxed = DetectorConfig(
            face_min_neighbors=3,
            profile_min_neighbors=4,
            face_min_size=(35, 35),
            profile_min_size=(35, 35),
            face_roi_bottom=0.60,
        )
        boxes = detect_people(resized, relaxed)

    person_count = len(boxes)
    status       = classify_riding(person_count)

    return {
        "resized_image": resized,
        "gray":          gray,
        "boxes":         boxes,
        "person_count":  person_count,
        "status":        status,
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detection_result(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    person_count: int,
    status: str,
) -> np.ndarray:
    out = image.copy()

    for idx, (x, y, bw, bh) in enumerate(boxes, 1):
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        label_y = max(y - 10, 20)
        cv2.putText(out, f"P{idx}", (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

    cv2.putText(out, f"People Count: {person_count}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    is_danger    = person_count > 2
    status_color = (0, 0, 255) if is_danger else (0, 255, 0)
    cv2.putText(out, f"Status: {status}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

    return out
