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


def process_image(
    image_path: Path,
    output_dir: Path,
    config: DetectorConfig,
    save_intermediate: bool = False,
    show: bool = False,
) -> None:
    """Process a single image and save results."""
    original_image = load_image(image_path)
    
    result = detect_triple_riding(original_image, config)

    final_output = draw_detection_result(
        image=result["resized_image"],
        boxes=result["boxes"],
        person_count=result["person_count"],
        status=result["status"],
    )

    # Generate output filename based on input filename
    output_filename = f"result_{image_path.stem}.jpg"
    output_image_path = output_dir / output_filename
    save_image(output_image_path, final_output)

    if save_intermediate:
        save_image(output_dir / f"{image_path.stem}_01_gray.jpg", result["gray"])

    print(f"\nInput Image: {image_path.name}")
    print(f"Detected People: {result['person_count']}")
    print(f"Status: {result['status']}")
    print(f"Output Saved: {output_image_path}")

    if show:
        cv2.imshow(f"Triple Riding Detection - {image_path.name}", final_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    """Run the full image-based triple riding detection pipeline."""
    args = parse_arguments()

    project_dir = Path(__file__).resolve().parent
    input_dir = project_dir / args.input_dir
    output_dir = project_dir / args.output_dir

    ensure_directory(input_dir)
    ensure_directory(output_dir)

    config = DetectorConfig()

    # If specific image is provided, process only that image
    if args.image:
        image_path = resolve_input_image_path(args.image, input_dir)
        process_image(image_path, output_dir, config, args.save_intermediate, args.show)
    else:
        # Process all images in the input directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(ext))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} image(s) to process...")
        
        for image_path in sorted(image_files):
            try:
                process_image(image_path, output_dir, config, args.save_intermediate, args.show)
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
        
        print(f"\n✓ Processed {len(image_files)} image(s) successfully!")


if __name__ == "__main__":
    main()
