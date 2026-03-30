# Project Implementation Report

## 1. Work Done Summary

This project was implemented as a complete beginner-friendly classical Computer Vision pipeline for overloaded bike detection (triple riding detection).

Completed work:

1. Built a modular image processing pipeline using OpenCV and NumPy.
2. Implemented strict rule-based rider/blob detection logic (no deep learning).
3. Added configurable parameters through a dedicated config class.
4. Added CLI-based execution with optional arguments.
5. Added optional intermediate-image saving for learning/debugging.
6. Added output visualization with boxes, count, and status.
7. Documented setup, usage, and limitations in README.

## 2. Requirements Coverage

All requested functional requirements are implemented:

1. Input handling:
   - Load image from input folder or CLI path
   - Resize to 640x480
2. Preprocessing:
   - Grayscale conversion
   - Gaussian blur (5x5)
3. Edge detection:
   - Canny (50, 150)
4. Morphological operations:
   - Dilation
   - Closing
   - Kernel (5x5)
5. ROI:
   - Top 50% only
6. Contour detection:
   - `cv2.findContours`
   - External contours
7. Contour filtering:
   - Area range [500, 5000]
   - Aspect ratio range (0.5, 1.5)
8. Count people:
   - Count valid contours
9. Decision logic:
   - Count > 2 -> Triple Riding Detected
   - Else -> Normal Riding
10. Visualization:

- Bounding boxes
- Person count text
- Status text
- Save output image

## 3. File-by-File Contents

### `main.py`

Purpose: Entry point and execution orchestration.

Contains:

- `parse_arguments()`:
  - Parses CLI options: `--image`, `--input-dir`, `--output-dir`, `--output-name`, `--save-intermediate`, `--show`.
- `main()`:
  - Resolves project/input/output paths.
  - Ensures required directories exist.
  - Loads image.
  - Calls detection pipeline from `detector.py`.
  - Draws annotations.
  - Saves final output image.
  - Optionally saves intermediate outputs.
  - Prints detection summary to terminal.
  - Optionally displays result in an OpenCV window.

### `detector.py`

Purpose: Core computer vision logic and classification.

Contains:

- `DetectorConfig` dataclass:
  - Stores all key pipeline parameters (resize size, blur, Canny thresholds, ROI ratio, contour filters).
- `resize_image()`:
  - Resizes image to fixed dimensions.
- `preprocess_image()`:
  - Converts to grayscale and applies Gaussian blur.
- `detect_edges()`:
  - Canny edge detector.
- `apply_morphological_operations()`:
  - Dilation + closing on edges.
- `extract_top_roi()`:
  - Crops top 50% ROI.
- `find_valid_person_boxes()`:
  - Finds external contours and applies area/aspect-ratio filtering.
- `classify_riding()`:
  - Converts person count to status string.
- `detect_triple_riding()`:
  - Full pipeline runner returning final and intermediate outputs.
- `draw_detection_result()`:
  - Draws bounding boxes, ROI divider, and overlay text on image.

### `utils.py`

Purpose: Helper functions for reusable file and image operations.

Contains:

- `SUPPORTED_IMAGE_EXTENSIONS`:
  - Allowed image types list.
- `ensure_directory()`:
  - Creates directory if missing.
- `list_images()`:
  - Lists valid image files in input folder.
- `resolve_input_image_path()`:
  - Resolves user image argument or defaults to first input image.
- `load_image()`:
  - Safe image loading with validation.
- `save_image()`:
  - Safe image saving with error handling.

### `README.md`

Purpose: User-facing documentation.

Contains:

- Project overview
- Pipeline explanation
- Root-level project structure
- Installation and run instructions
- CLI examples
- Output details
- Limitations and beginner notes
- Constraint compliance statement

### `requirements.txt`

Purpose: Python dependencies.

Contains:

- `opencv-python`
- `numpy`

### `input/`

Purpose: Store test images.

Usage:

- Put bike images here if running default command `python main.py`.

### `output/`

Purpose: Store generated results.

Usage:

- Final annotated result is saved here.
- Optional intermediate images are saved here when enabled.

## 4. Notes

- The project intentionally avoids deep learning and pre-trained detectors.
- This pipeline is educational and works best with clear images and side-view rider visibility.
- For better accuracy on new datasets, tune values in `DetectorConfig`.
