import cv2
import numpy as np
from .ransac_homography import ransac_homography

# ---------- PARAMETERS (from your our_sift.py) ----------
NUM_OCTAVES = 4
SCALES_PER_OCTAVE = 3  # real SIFT uses 3+3
SIGMA = 1.6
CONTRAST_THRESHOLD = 0.03
EDGE_THRESHOLD = 10  # not used directly but kept from your code


# ==========================
# CORE SIFT IMPLEMENTATION
# ==========================

def build_gaussian_pyramid(img_gray):
    # Normalize to float32 [0..1]
    img = img_gray.astype(np.float32) / 255.0
    octaves = []

    for o in range(NUM_OCTAVES):
        scales = []
        k = 2 ** (1.0 / SCALES_PER_OCTAVE)
        sigmas = [SIGMA * (k ** i) for i in range(SCALES_PER_OCTAVE + 3)]
        base = img if o == 0 else cv2.pyrDown(octaves[o - 1][SCALES_PER_OCTAVE])

        for s in sigmas:
            blurred = cv2.GaussianBlur(base, (0, 0), s)
            scales.append(blurred)

        octaves.append(scales)

    return octaves


def build_dog_pyramid(gaussian_pyr):
    dog_pyr = []
    for octv in gaussian_pyr:
        dogs = []
        for i in range(1, len(octv)):
            dogs.append(octv[i] - octv[i - 1])
        dog_pyr.append(dogs)
    return dog_pyr


def is_extremum(dog_prev, dog, dog_next, y, x, thresh):
    val = dog[y, x]
    if abs(val) < thresh:
        return False

    patch_prev = dog_prev[y - 1:y + 2, x - 1:x + 2]
    patch_curr = dog[y - 1:y + 2, x - 1:x + 2]
    patch_next = dog_next[y - 1:y + 2, x - 1:x + 2]

    val_max = max(val, patch_prev.max(), patch_curr.max(), patch_next.max())
    val_min = min(val, patch_prev.min(), patch_curr.min(), patch_next.min())

    return val == val_max or val == val_min


def find_keypoints(gaussian_pyr, dog_pyr):
    keypoints = []  # (octave, scale, y, x)

    for o, dogs in enumerate(dog_pyr):
        for s in range(1, len(dogs) - 1):
            dog_prev = dogs[s - 1]
            dog = dogs[s]
            dog_next = dogs[s + 1]
            h, w = dog.shape

            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if is_extremum(dog_prev, dog, dog_next, y, x, CONTRAST_THRESHOLD):
                        keypoints.append((o, s, y, x))

    return keypoints


def compute_orientation(gaussian_pyr, keypoints):
    oriented_kps = []  # (x_img, y_img, scale, orientation)

    for (o, s, y, x) in keypoints:
        img = gaussian_pyr[o][s]
        h, w = img.shape
        if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
            continue

        # Gradient
        dx = img[y, x + 1] - img[y, x - 1]
        dy = img[y - 1, x] - img[y + 1, x]
        mag = np.sqrt(dx * dx + dy * dy)
        angle = np.degrees(np.arctan2(dy, dx)) % 360

        # Map to original image scale
        scale_factor = 2 ** o
        x_img = x * scale_factor
        y_img = y * scale_factor

        oriented_kps.append((x_img, y_img, s, angle, mag))

    return oriented_kps


def compute_descriptors(gaussian_pyr, oriented_kps):
    descriptors = []
    keypoints_out = []

    for (x, y, s, angle, mag) in oriented_kps:
        # Use first octave image as reference for descriptor
        img = gaussian_pyr[0][s]
        h, w = img.shape
        x0, y0 = int(round(x)), int(round(y))
        if x0 < 8 or x0 >= w - 8 or y0 < 8 or y0 >= h - 8:
            continue

        # Simple 4x4 grid, 8-bin HOG
        desc = []
        patch = img[y0 - 8:y0 + 8, x0 - 8:x0 + 8]
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mag_patch = np.sqrt(gx * gx + gy * gy)
        ori_patch = (np.degrees(np.arctan2(gy, gx)) - angle) % 360

        for i in range(4):
            for j in range(4):
                cell_mag = mag_patch[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4].flatten()
                cell_ori = ori_patch[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4].flatten()

                hist = np.zeros(8, dtype=np.float32)
                for m, o in zip(cell_mag, cell_ori):
                    bin_idx = int(o // 45) % 8
                    hist[bin_idx] += m
                desc.extend(hist)

        desc = np.array(desc, dtype=np.float32)
        # Normalize
        norm = np.linalg.norm(desc)
        if norm > 1e-6:
            desc /= norm
        descriptors.append(desc)
        keypoints_out.append((x, y))

    return np.array(keypoints_out, dtype=np.float32), np.array(descriptors, dtype=np.float32)


def run_from_scratch_sift(img_gray):
    """
    Main entry: given a grayscale image, return:
      pts:  (N, 2) array of (x, y) positions
      desc: (N, D) array of descriptors
    """
    gauss_pyr = build_gaussian_pyramid(img_gray)
    dog_pyr = build_dog_pyramid(gauss_pyr)
    basic_kps = find_keypoints(gauss_pyr, dog_pyr)
    oriented_kps = compute_orientation(gauss_pyr, basic_kps)
    pts, desc = compute_descriptors(gauss_pyr, oriented_kps)
    return pts, desc


# ==========================
# MATCHING + RANSAC + DRAWING
# ==========================

def match_features(desc1, desc2):
    """
    Use BFMatcher + Lowe's ratio, like your match_descriptors.
    Returns a list of cv2.DMatch objects.
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches_knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def from_scratch_ransac(pts1, pts2, matches, num_iter=1000, thresh=3.0):
    """
    Run your RANSAC homography on matched keypoints.
    pts1, pts2: (N, 2) float32 arrays
    matches: list of cv2.DMatch
    Returns:
        H  (3x3) or None
        inlier_indices (np.array of indices into 'matches')
    """
    if len(matches) < 4:
        return None, []

    src_pts = np.float32([pts1[m.queryIdx] for m in matches])
    dst_pts = np.float32([pts2[m.trainIdx] for m in matches])

    H, inliers = ransac_homography(src_pts, dst_pts, num_iter=num_iter, thresh=thresh)
    return H, inliers


def draw_matches_custom(img1, pts1, img2, pts2, matches, inliers):
    """
    Visualize inlier matches between img1 and img2.

    img1, img2: BGR images
    pts1, pts2: (N, 2) arrays of keypoint coordinates
    matches:    list of cv2.DMatch
    inliers:    indices (into 'matches') returned by from_scratch_ransac
    """
    # Create side-by-side canvas
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    W = w1 + w2

    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[0:h1, 0:w1] = img1
    vis[0:h2, w1:w1 + w2] = img2

    if inliers is None or len(inliers) == 0:
        return vis

    for idx in inliers:
        m = matches[idx]
        x1, y1 = pts1[m.queryIdx]
        x2, y2 = pts2[m.trainIdx]

        p1 = (int(x1), int(y1))
        p2 = (int(x2) + w1, int(y2))  # shift second image x by w1

        cv2.circle(vis, p1, 3, (0, 255, 0), -1)
        cv2.circle(vis, p2, 3, (0, 255, 0), -1)
        cv2.line(vis, p1, p2, (0, 255, 0), 1)

    return vis
