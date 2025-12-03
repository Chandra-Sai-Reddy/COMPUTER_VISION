import os
from pathlib import Path

import cv2
import numpy as np

# ==========================
# CONFIG
# ==========================
# ==========================
# CONFIG: UPDATE PATHS HERE
# ==========================

DATASET_DIR = "/Users/donthireddychandrasaireddy/Desktop/COMPUTER_VISION/Module3/Mod_3_dataset"

OUTPUT_DIR = "/Users/donthireddychandrasaireddy/Desktop/COMPUTER_VISION/Module3/outputs_mod3"

# Sobel/LoG parameters
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0


# ==========================
# UTILS
# ==========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ==========================
# TASK 1 – GRADIENTS & LoG
# ==========================

def compute_gradients(gray):
    """
    Compute gradient magnitude and angle using Sobel filters.
    Returns images scaled to [0, 255] for visualization.
    """
    # Sobel in x and y
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude and angle (degrees)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Normalize for visualization
    mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    angle_vis = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return mag_vis, angle_vis


def compute_log(gray):
    """
    Laplacian of Gaussian:
      1) Gaussian blur
      2) Laplacian
    Returns an image scaled to [0, 255] for visualization.
    """
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    log_vis = cv2.convertScaleAbs(log)
    return log_vis


# ==========================
# TASK 2 – EDGE & CORNER KEYPOINTS
# ==========================

def detect_edge_keypoints(gray, step=5):
    """
    Simple EDGE keypoint detector:
      - Use Canny to find edges
      - Treat (subsampled) edge pixels as keypoints
    Returns:
      edges (binary image)
      list of (x, y) keypoints
    """
    edges = cv2.Canny(gray, 100, 200)

    ys, xs = np.where(edges > 0)
    points = list(zip(xs, ys))

    # Subsample to avoid drawing thousands of points
    if step > 1 and len(points) > 0:
        points = points[::step]

    return edges, points


def detect_corner_keypoints(gray):
    """
    Simple CORNER keypoint detector using Harris corner response.
    Returns:
      corner_response (float32 image)
      list of (x, y) keypoints
    """
    gray_f = np.float32(gray)
    # Harris parameters can be tuned
    harris = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)

    # Threshold relative to max response
    thresh = 0.01 * harris.max()
    ys, xs = np.where(harris > thresh)
    points = list(zip(xs, ys))

    return harris, points


def visualize_keypoints(img_bgr, edge_points, corner_points):
    """
    Draw edge keypoints (red) and corner keypoints (green) on a copy of the image.
    """
    vis = img_bgr.copy()

    # Edge keypoints: red
    for (x, y) in edge_points:
        cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)

    # Corner keypoints: green
    for (x, y) in corner_points:
        cv2.circle(vis, (x, y), 4, (0, 255, 0), 1)

    return vis


# ==========================
# TASK 3 – OBJECT BOUNDARY
# ==========================

def find_main_object_contour(gray):
    """
    Finds the largest external contour, assuming it is the main object.
    You may adjust this depending on your scene.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, edges

    # Take the largest contour by area
    main_contour = max(contours, key=cv2.contourArea)
    return main_contour, edges


def visualize_boundary(img_bgr, contour):
    """
    Draws the contour as the boundary of the object.
    """
    vis = img_bgr.copy()
    cv2.drawContours(vis, [contour], -1, (255, 0, 0), 2)
    return vis


# ==========================
# MAIN PIPELINE
# ==========================

def process_dataset():
    dataset_dir = Path(DATASET_DIR)
    output_root = Path(OUTPUT_DIR)

    # Create subfolders for each task
    grad_dir = output_root / "task1_gradients"
    log_dir = output_root / "task1_log"
    edge_dir = output_root / "task2_edges"
    corner_dir = output_root / "task2_corners"
    keypoints_vis_dir = output_root / "task2_keypoints_vis"
    boundary_edges_dir = output_root / "task3_edges"
    boundary_vis_dir = output_root / "task3_boundary_vis"

    for d in [grad_dir, log_dir, edge_dir, corner_dir,
              keypoints_vis_dir, boundary_edges_dir, boundary_vis_dir]:
        ensure_dir(d)

    # Iterate over images
    image_files = sorted([
        p for p in dataset_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    print(f"Found {len(image_files)} images.")

    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        img = load_image(img_path)
        gray = to_gray(img)

        stem = img_path.stem  # base filename without extension

        # ---------- TASK 1: GRADIENTS ----------
        mag_vis, angle_vis = compute_gradients(gray)
        log_vis = compute_log(gray)

        cv2.imwrite(str(grad_dir / f"{stem}_grad_mag.png"), mag_vis)
        cv2.imwrite(str(grad_dir / f"{stem}_grad_angle.png"), angle_vis)
        cv2.imwrite(str(log_dir / f"{stem}_log.png"), log_vis)

        # ---------- TASK 2: EDGE & CORNER KEYPOINTS ----------
        edges, edge_points = detect_edge_keypoints(gray)
        harris_response, corner_points = detect_corner_keypoints(gray)

        # Save edge and corner response maps (for analysis)
        cv2.imwrite(str(edge_dir / f"{stem}_edges.png"), edges)

        # Normalize Harris for visualization
        harris_norm = cv2.normalize(harris_response, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(corner_dir / f"{stem}_harris.png"), harris_norm)

        # Visualize keypoints overlayed on original
        keypoints_vis = visualize_keypoints(img, edge_points, corner_points)
        cv2.imwrite(str(keypoints_vis_dir / f"{stem}_keypoints.png"),
                    keypoints_vis)

        # ---------- TASK 3: OBJECT BOUNDARY ----------
        main_contour, edges_for_boundary = find_main_object_contour(gray)
        cv2.imwrite(str(boundary_edges_dir / f"{stem}_edges_for_boundary.png"),
                    edges_for_boundary)

        if main_contour is not None:
            boundary_vis = visualize_boundary(img, main_contour)
            cv2.imwrite(str(boundary_vis_dir / f"{stem}_boundary.png"),
                        boundary_vis)
        else:
            print(f"  [!] No contour found for {img_path.name}")

    print("Done.")


if __name__ == "__main__":
    process_dataset()
