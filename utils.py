"""Utility helpers for file handling and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def list_images(input_dir: Path) -> List[Path]:
    """Return sorted image files from input directory."""
    if not input_dir.exists():
        return []

    image_paths = [
        file_path
        for file_path in input_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def resolve_input_image_path(image_arg: str | None, input_dir: Path) -> Path:
    """Resolve image path from CLI argument or first image in input folder."""
    if image_arg:
        image_path = Path(image_arg)
        if not image_path.is_absolute():
            image_path = input_dir / image_arg
        return image_path

    image_files = list_images(input_dir)
    if not image_files:
        raise FileNotFoundError(
            f"No image found. Add an image to '{input_dir}' or pass --image <path>."
        )
    return image_files[0]


def load_image(image_path: Path) -> np.ndarray:
    """Load image using OpenCV and validate result."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def save_image(image_path: Path, image: np.ndarray) -> None:
    """Save image and raise error on failure."""
    success = cv2.imwrite(str(image_path), image)
    if not success:
        raise IOError(f"Failed to save image: {image_path}")
