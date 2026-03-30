"""Core detection logic for triple riding detection using classical CV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    """Configurable parameters for the detection pipeline."""

    resize_width: int = 640
    resize_height: int = 480
    blur_kernel: Tuple[int, int] = (5, 5)
    canny_threshold_1: int = 50
    canny_threshold_2: int = 150
    morph_kernel: Tuple[int, int] = (5, 5)
    dilate_iterations: int = 1
    roi_ratio: float = 0.5
    min_contour_area: float = 500
    max_contour_area: float = 5000
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 1.5


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize image to fixed dimensions."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def preprocess_image(image: np.ndarray, blur_kernel: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert image to grayscale and apply Gaussian blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    return gray, blurred


def detect_edges(blurred_image: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    """Detect edges using Canny."""
    return cv2.Canny(blurred_image, low_threshold, high_threshold)


def apply_morphological_operations(
    edge_image: np.ndarray,
    kernel_size: Tuple[int, int],
    dilate_iterations: int,
) -> np.ndarray:
    """Apply dilation and closing to connect fragmented regions."""
    kernel = np.ones(kernel_size, dtype=np.uint8)
    dilated = cv2.dilate(edge_image, kernel, iterations=dilate_iterations)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    return closed


def extract_top_roi(image: np.ndarray, roi_ratio: float) -> Tuple[np.ndarray, int]:
    """Return top region of interest and vertical offset."""
    roi_height = int(image.shape[0] * roi_ratio)
    return image[:roi_height, :], 0


def find_valid_person_boxes(
    roi_binary_image: np.ndarray,
    y_offset: int,
    min_area: float,
    max_area: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> List[Tuple[int, int, int, int]]:
    """Find and filter contours that roughly match upper body / head blobs."""
    contours, _ = cv2.findContours(
        roi_binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_boxes: List[Tuple[int, int, int, int]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue

        aspect_ratio = w / float(h)
        if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            valid_boxes.append((x, y + y_offset, w, h))

    return valid_boxes


def classify_riding(person_count: int) -> str:
    """Classify riding status from detected people count."""
    return "Triple Riding Detected" if person_count > 2 else "Normal Riding"


def detect_triple_riding(
    image: np.ndarray,
    config: DetectorConfig | None = None,
) -> Dict[str, np.ndarray | int | str | List[Tuple[int, int, int, int]]]:
    """Run full detection pipeline and return outputs + intermediate images."""
    cfg = config or DetectorConfig()

    resized = resize_image(image, cfg.resize_width, cfg.resize_height)
    gray, blurred = preprocess_image(resized, cfg.blur_kernel)
    edges = detect_edges(blurred, cfg.canny_threshold_1, cfg.canny_threshold_2)
    morphed = apply_morphological_operations(edges, cfg.morph_kernel, cfg.dilate_iterations)

    roi, y_offset = extract_top_roi(morphed, cfg.roi_ratio)
    valid_boxes = find_valid_person_boxes(
        roi,
        y_offset,
        cfg.min_contour_area,
        cfg.max_contour_area,
        cfg.min_aspect_ratio,
        cfg.max_aspect_ratio,
    )

    person_count = len(valid_boxes)
    status = classify_riding(person_count)

    return {
        "resized_image": resized,
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "morphed": morphed,
        "roi": roi,
        "boxes": valid_boxes,
        "person_count": person_count,
        "status": status,
    }


def draw_detection_result(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    person_count: int,
    status: str,
    roi_ratio: float = 0.5,
) -> np.ndarray:
    """Draw boxes, ROI divider, count, and status on image."""
    output_image = image.copy()

    for x, y, w, h in boxes:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    roi_line_y = int(output_image.shape[0] * roi_ratio)
    cv2.line(output_image, (0, roi_line_y), (output_image.shape[1], roi_line_y), (255, 0, 0), 2)

    cv2.putText(
        output_image,
        f"People Count: {person_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    status_color = (0, 0, 255) if status == "Triple Riding Detected" else (0, 255, 0)
    cv2.putText(
        output_image,
        f"Status: {status}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )

    return output_image
