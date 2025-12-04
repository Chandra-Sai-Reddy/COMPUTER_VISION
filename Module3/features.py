import cv2
import numpy as np

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

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
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
# TASK 4 – ARUCO BOUNDARY (NON-RECTANGULAR)
# ==========================

def get_aruco_hull(img_bgr):
    """
    Detects ArUco markers and draws the convex hull around them.
    Returns:
      - BGR image with hull + markers
      - status message
    """
    if img_bgr is None:
        return None, "No image provided."

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    except Exception as e:
        return None, f"ArUco not available in this OpenCV build: {e}"

    corners, ids, _ = detector.detectMarkers(gray)
    if not corners:
        return None, "No ArUco markers found."

    all_pts = np.vstack([c[0] for c in corners]).astype(np.int32)

    hull = cv2.convexHull(all_pts)

    result = img_bgr.copy()
    cv2.drawContours(result, [hull], -1, (0, 255, 0), 3)
    cv2.aruco.drawDetectedMarkers(result, corners, ids)

    return result, "ArUco hull detected."


# ==========================
# TASK 5 – SAM2 PLACEHOLDER (CLOUD)
# ==========================

def get_sam2_segmentation(img_bgr):
    """
    Placeholder for SAM2 on Streamlit Cloud.
    We do NOT run the model here (no torch).
    The app instead lets the user upload precomputed SAM2 masks.
    """
    return None, None, "SAM2 cannot run on Streamlit Cloud. Upload a SAM2 mask instead."
