"""Entry point for triple riding detection on an image.

Optional feature included:
- CLI argument support for image path and intermediate output saving.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from detector import DetectorConfig, detect_triple_riding, draw_detection_result
from utils import ensure_directory, load_image, resolve_input_image_path, save_image


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Triple Riding Detection (Overloaded Bike Detection) using classical CV"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image path. If relative, it is resolved inside input/ folder.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory where result images are saved.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="result.jpg",
        help="Name of final output image.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate images (gray, edges, morphed, roi).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display final output in a window.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full image-based triple riding detection pipeline."""
    args = parse_arguments()

    project_dir = Path(__file__).resolve().parent
    input_dir = project_dir / args.input_dir
    output_dir = project_dir / args.output_dir

    ensure_directory(input_dir)
    ensure_directory(output_dir)

    image_path = resolve_input_image_path(args.image, input_dir)
    original_image = load_image(image_path)

    config = DetectorConfig()
    result = detect_triple_riding(original_image, config)

    final_output = draw_detection_result(
        image=result["resized_image"],
        boxes=result["boxes"],
        person_count=result["person_count"],
        status=result["status"],
        roi_ratio=config.roi_ratio,
    )

    output_image_path = output_dir / args.output_name
    save_image(output_image_path, final_output)

    if args.save_intermediate:
        save_image(output_dir / "01_gray.jpg", result["gray"])
        save_image(output_dir / "02_edges.jpg", result["edges"])
        save_image(output_dir / "03_morphed.jpg", result["morphed"])
        save_image(output_dir / "04_roi_top50.jpg", result["roi"])

    print(f"Input Image: {image_path}")
    print(f"Detected People: {result['person_count']}")
    print(f"Status: {result['status']}")
    print(f"Output Saved: {output_image_path}")

    if args.show:
        cv2.imshow("Triple Riding Detection", final_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
