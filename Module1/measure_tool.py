import cv2
import numpy as np

def get_calibration_data(calib_path):
    """
    Load calibration data and compute focal length f.
    Directly based on your original measure.py:

        data = np.load("calibration_data_mac.npz")
        mtx, dist = data["mtx"], data["dist"]
        f = (mtx[0,0] + mtx[1,1]) / 2
    """
    try:
        data = np.load(calib_path)
        mtx = data["mtx"]
        dist = data["dist"]
        f = (mtx[0, 0] + mtx[1, 1]) / 2.0
        return mtx, dist, f
    except Exception:
        # app.py treats f=None as "calibration not found"
        return None, None, None


def undistort_image(img, mtx, dist):
    """
    OPTION B: do NOT undistort – just return a copy of the original image.

    We keep this function so existing calls still work, but we disable
    the geometric correction because the calibration does not match
    the camera used for your measurement images.
    """
    # If you ever want to re-enable true undistortion, replace this body
    # with the original OpenCV code again.
    return img.copy()


def calculate_real_distance(p1, p2, Z, f):
    """
    Same formula as in your click_event:

        px_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
        real_size = (Z * px_dist) / f
    """
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    px_dist = float(np.linalg.norm(p1 - p2))
    real_size = (Z * px_dist) / float(f)

    return real_size
