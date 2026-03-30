# Triple Riding Detection (Overloaded Bike Detection) using Classical Computer Vision

A beginner-friendly Computer Vision project to estimate how many riders are on a bike and flag potential triple riding.

This project uses only classical CV techniques:

- OpenCV
- NumPy
- Rule-based contour filtering

## Project Structure

```text
.
|-- input/              # input images
|-- output/             # output images
|-- main.py             # main execution script
|-- detector.py         # all detection logic
|-- utils.py            # helper functions
|-- README.md           # project explanation
|-- REPORT.md           # implementation report and file summary
|-- requirements.txt    # dependencies
```

## Objective

Given an input image:

1. Detect upper-body/head-like blobs in the top half of the scene.
2. Count the number of valid detected people.
3. Classify:
   - `count > 2` -> **Triple Riding Detected**
   - otherwise -> **Normal Riding**

## Pipeline (Step-by-Step)

1. Input handling
   - Load image from `input/` folder (or via `--image` argument)
   - Resize image to `640x480`

2. Preprocessing
   - Convert to grayscale
   - Apply Gaussian blur `(5, 5)`

3. Edge detection
   - Apply Canny edge detection with thresholds `(50, 150)`

4. Morphological operations
   - Dilation to connect broken edges
   - Closing to fill gaps
   - Kernel size `(5, 5)`

5. Region of interest
   - Process only top `50%` of image to reduce road/wheel noise

6. Contour detection
   - `cv2.findContours` with external contours only (`cv2.RETR_EXTERNAL`)

7. Contour filtering
   - Keep contour if:
     - `500 <= area <= 5000`
     - `0.5 < aspect_ratio < 1.5`

8. Counting and decision
   - Valid contours are treated as detected people
   - If people count is greater than `2`, classify as triple riding

9. Visualization
   - Draw bounding boxes over valid blobs
   - Draw ROI divider line
   - Overlay rider count and status
   - Save result in `output/`

## Installation

Run from repository root:

```bash
pip install -r requirements.txt
```

## How To Run

1. Put your test image in `input/`.
2. Run:

```bash
python main.py
```

### Useful CLI options

```bash
python main.py --image bike.jpg
python main.py --image /full/path/to/image.jpg
python main.py --output-name result1.jpg
python main.py --save-intermediate
python main.py --show
```

## Example Outputs

- Final annotated image: `output/result.jpg`
- If `--save-intermediate` is used:
  - `output/01_gray.jpg`
  - `output/02_edges.jpg`
  - `output/03_morphed.jpg`
  - `output/04_roi_top50.jpg`

## Limitations

- This is a rule-based method, so results depend on lighting, camera angle, and image quality.
- False positives can occur when background objects resemble rider blobs.
- Heavy occlusion may merge riders into fewer contours.
- Thresholds (`area`, `aspect ratio`, Canny values) may require tuning for different scenes.

## Notes for Beginners

- Start with clear side-view bike images.
- Use `--save-intermediate` to understand each processing stage.
- Tune `DetectorConfig` in `detector.py` to improve robustness on your data.

## Constraint Compliance

- No deep learning models used.
- No pre-trained person detector used.
- Only classical computer vision operations are used.
