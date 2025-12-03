import cv2
import numpy as np

print(f"[Panorama] Using OpenCV version: {cv2.__version__}")


def stitch_images(img_bottom, img_top):
    """
    Stitches two images together vertically.
    img_bottom is the base image,
    img_top is the image to be warped and added above.

    Returns the stitched and cropped BGR image, or None if stitching fails.
    """

    # --- Step 2: Find Keypoints and Descriptors (same as your script) ---
    sift = cv2.SIFT_create()
    kp_top, des_top = sift.detectAndCompute(img_top, None)
    kp_bottom, des_bottom = sift.detectAndCompute(img_bottom, None)

    if des_top is None or des_bottom is None:
        print("[Panorama] Error: No descriptors found for one or both images.")
        return None

    # --- Step 3: Match Features ---
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_bottom, des_top, k=2)

    # --- Step 4: Lowe's Ratio Test ---
    good_matches = []
    if not matches or len(matches) < 2:
        print("[Panorama] Warning: Not enough matches found.")
    else:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    print(f"[Panorama] Found {len(good_matches)} good matches after ratio test.")

    MIN_MATCH_COUNT = 4
    if len(good_matches) <= MIN_MATCH_COUNT:
        print(f"[Panorama] Not enough good matches - {len(good_matches)}/{MIN_MATCH_COUNT}")
        return None

    # --- Step 5: Find Homography ---
    src_pts = np.float32([kp_bottom[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_top[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("[Panorama] Homography estimation failed.")
        return None

    # --- Step 6: Warp and Stitch ---
    h_top, w_top = img_top.shape[:2]
    h_bottom, w_bottom = img_bottom.shape[:2]

    canvas_height = h_top + h_bottom
    canvas_width = max(w_top, w_bottom)

    dst = cv2.warpPerspective(img_bottom, H, (canvas_width, canvas_height))
    dst[0:h_top, 0:w_top] = img_top

    # --- Step 7: Crop black padding (same idea as your script) ---
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_result = dst[y:y + h, x:x + w]
        return cropped_result
    else:
        # If contour finding fails, return un-cropped
        return dst


def stitch_collection(images_list):
    """
    High-level function for Streamlit.
    Takes a list of BGR images in order [bottom, ..., top]
    and stitches them one-by-one using stitch_images().

    Returns:
        stitched_image (BGR) or None
        message (str)
    """
    if not images_list or len(images_list) < 2:
        return None, "Need at least 2 images to stitch."

    # Start with the bottom-most image
    current_panorama = images_list[0].copy()
    print("[Panorama] Starting stitching with first image as base.")

    for idx, img_top in enumerate(images_list[1:], start=1):
        print(f"[Panorama] Stitching image {idx + 1}/{len(images_list)}...")
        if img_top is None:
            return None, f"Image at index {idx} is None."

        result = stitch_images(current_panorama, img_top)

        if result is None:
            return None, f"Stitching failed at image index {idx + 1}."

        current_panorama = result

    msg = f"Successfully stitched {len(images_list)} images."
    return current_panorama, msg
