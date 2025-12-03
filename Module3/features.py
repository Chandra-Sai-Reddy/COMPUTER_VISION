import cv2
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os

# --- SAM2 import guard ---
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    SAM2_IMPORT_ERROR = ""
except Exception as e:
    SAM2_AVAILABLE = False
    SAM2_IMPORT_ERROR = str(e)

# ==========================
# BASIC UTILITIES
# ==========================

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ==========================
# TASK 1 – GRADIENTS & LoG
# ==========================

GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0

def compute_gradients(gray):
    """
    Compute gradient magnitude and angle using Sobel filters.
    Returns images scaled to [0, 255] for visualization.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    angle_vis = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return mag_vis, angle_vis


def compute_log(gray):
    """
    Laplacian of Gaussian:
      1) Gaussian blur
      2) Laplacian
    Returns image scaled to [0, 255] for visualization.
    """
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    log_vis = cv2.convertScaleAbs(log)
    return log_vis


def get_gradient_and_log(img_bgr):
    """
    Wrapper for Streamlit:
    Takes a BGR image and returns:
      - gradient magnitude visualization
      - gradient angle visualization
      - LoG visualization
    """
    gray = to_gray(img_bgr)
    mag_vis, angle_vis = compute_gradients(gray)
    log_vis = compute_log(gray)
    return mag_vis, angle_vis, log_vis


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

    if step > 1 and len(points) > 0:
        points = points[::step]

    return edges, points


def detect_corner_keypoints(gray):
    """
    Harris corner keypoint detector.
    Returns:
      corner_response (float32 image)
      list of (x, y) keypoints
    """
    gray_f = np.float32(gray)
    harris = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)

    thresh = 0.01 * harris.max()
    ys, xs = np.where(harris > thresh)
    points = list(zip(xs, ys))

    return harris, points


def visualize_keypoints(img_bgr, edge_points, corner_points):
    """
    Draw edge keypoints (red) and corner keypoints (green).
    """
    vis = img_bgr.copy()

    for (x, y) in edge_points:
        cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)

    for (x, y) in corner_points:
        cv2.circle(vis, (x, y), 4, (0, 255, 0), 1)

    return vis


def get_keypoints(img_bgr):
    """
    Wrapper for Streamlit:
    Takes a BGR image and returns:
      - edges image (for Canny view)
      - BGR image with keypoints overlaid
    """
    gray = to_gray(img_bgr)

    edges, edge_points = detect_edge_keypoints(gray)
    harris_response, corner_points = detect_corner_keypoints(gray)

    keypoints_vis = visualize_keypoints(img_bgr, edge_points, corner_points)
    return edges, keypoints_vis


# ==========================
# TASK 3 – OBJECT BOUNDARY
# ==========================

def find_main_object_contour(gray):
    """
    Finds the largest external contour, assuming it is the main object.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, edges

    main_contour = max(contours, key=cv2.contourArea)
    return main_contour, edges


def visualize_boundary(img_bgr, contour):
    """
    Draws the contour as the boundary of the object.
    """
    vis = img_bgr.copy()
    cv2.drawContours(vis, [contour], -1, (255, 0, 0), 2)
    return vis


def get_boundary(img_bgr):
    """
    Wrapper for Streamlit:
    Takes a BGR image and returns:
      - mask (edges used for boundary detection)
      - BGR image with boundary drawn
    """
    gray = to_gray(img_bgr)
    main_contour, edges_for_boundary = find_main_object_contour(gray)

    if main_contour is not None:
        boundary_vis = visualize_boundary(img_bgr, main_contour)
    else:
        boundary_vis = img_bgr.copy()

    return edges_for_boundary, boundary_vis
# ==========================
# TASKS 4 & 5 – ArUco + SAM2
# ==========================

# --- ArUco helper: try multiple dictionaries ---
ARUCO_DICT_TYPES = [
    cv2.aruco.DICT_6X6_250,
    cv2.aruco.DICT_5X5_100,
    cv2.aruco.DICT_4X4_50,
]

def _detect_aruco_any(image_bgr):
    """
    Try several common ArUco dictionaries and return
    the first successful detection.
    """
    for dict_type in ARUCO_DICT_TYPES:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        corners, ids, _ = detector.detectMarkers(image_bgr)
        if corners and ids is not None and len(corners) > 0:
            print(f"[Module3.features] ArUco detected with dict {dict_type}")
            return corners, ids

    return [], None


# --- SAM2 setup (load once when module is imported) ---
SAM2_PREDICTOR = None
SAM2_DEVICE = "cpu"

def _init_sam2():
    """
    Lazy-load the SAM2 model on first use, reuse afterwards.
    """
    global SAM2_PREDICTOR, SAM2_DEVICE
    if SAM2_PREDICTOR is not None:
        return SAM2_PREDICTOR, SAM2_DEVICE

    if torch.cuda.is_available():
        SAM2_DEVICE = "cuda"
    else:
        SAM2_DEVICE = "cpu"

    print(f"[Module3.features] Loading SAM2 on device: {SAM2_DEVICE}")
    SAM2_PREDICTOR = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large",
        device=SAM2_DEVICE,
    )
    return SAM2_PREDICTOR, SAM2_DEVICE


def get_aruco_hull(img_bgr):
    """
    TASK 4:
    Detect ArUco markers in a single BGR image and draw the
    convex hull of all marker corners.

    Returns:
        hull_image (BGR)  or None if no markers.
        message (str)
    """
    image = img_bgr.copy()

    # Try all supported dictionaries
    corners, ids = _detect_aruco_any(image)

    if not corners or ids is None:
        return None, "No ArUco markers found in image (tried several dictionaries)."

    # Collect all marker corners
    all_corners = []
    for marker_corners in corners:
        for corner in marker_corners[0]:
            all_corners.append(corner)

    if not all_corners:
        return None, "No ArUco corners found."

    points = np.array(all_corners, dtype=np.int32)
    hull_indices = cv2.convexHull(points, returnPoints=False)
    hull_points = points[hull_indices.squeeze()]

    hull_image = image.copy()
    cv2.aruco.drawDetectedMarkers(hull_image, corners, ids)
    cv2.drawContours(hull_image, [hull_points], -1, (255, 0, 0), 3)  # Blue hull

    return hull_image, "ArUco convex hull drawn successfully."

def get_sam2_segmentation(img_bgr):
    """
    TASK 5:
    Use ArUco marker centers as prompts for SAM2 to segment the
    non-rectangular object.

    Returns:
        mask_image   : uint8 mask (0 or 255)
        overlay_image: BGR image with red overlay + green prompt dots
        message      : str
    """
    if img_bgr is None or img_bgr.size == 0:
        return None, None, "Input image is empty."

    if not SAM2_AVAILABLE:
        return None, None, f"SAM2 not available: {SAM2_IMPORT_ERROR}"

    image = img_bgr.copy()

    # 1. Detect ArUco markers (try several dictionaries)
    corners, ids = _detect_aruco_any(image)
    if not corners or ids is None:
        return None, None, "No ArUco markers found – cannot prompt SAM2 (tried several dictionaries)."

    # 2. Compute marker centers to use as prompts
    prompt_points = []
    for marker_corners in corners:
        M = cv2.moments(marker_corners[0])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            prompt_points.append([cX, cY])

    if not prompt_points:
        return None, None, "Could not compute marker centers."

    try:
        # 3. Init SAM2 (load model once)
        predictor, device = _init_sam2()

        # SAM2 expects RGB and NumPy points
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        point_coords = np.array(prompt_points, dtype=np.float32)   # (N, 2)
        point_labels = np.ones(len(prompt_points), dtype=np.int32) # (N,)

        # 4. Run SAM2 – NOTE: using .predict, NOT .predict_torch
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # scores is a NumPy array, pick best mask by score
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx]            # (H, W) bool or 0/1
        best_mask = best_mask.astype(bool)

        # 5. Build mask image (single-channel 0/255)
        mask_image = (best_mask.astype(np.uint8) * 255)

        # 6. Build overlay image
        overlay = image.copy()
        overlay[best_mask] = (0, 0, 255)  # Red where mask is True
        final_vis = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        # Draw green dots at marker centers
        for pt in prompt_points:
            cv2.circle(final_vis, tuple(pt), 5, (0, 255, 0), -1)

        return mask_image, final_vis, "SAM2 segmentation computed successfully."

    except Exception as e:
        # If anything goes wrong, don't crash Streamlit
        return None, None, f"SAM2 segmentation failed: {e}"
