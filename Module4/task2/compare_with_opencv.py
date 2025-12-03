import cv2
import numpy as np
from our_sift import run_from_scratch_sift
from ransac_homography import ransac_homography

def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

if __name__ == "__main__":
    img1 = cv2.imread("images/img1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("images/img2.jpg", cv2.IMREAD_GRAYSCALE)

    # ---------- OUR SIFT ----------
    pts1, desc1 = run_from_scratch_sift(img1)
    pts2, desc2 = run_from_scratch_sift(img2)
    print(f"Our SIFT: {len(pts1)} keypoints in image1, {len(pts2)} in image2")

    matches = match_descriptors(desc1, desc2)
    print(f"Our SIFT good matches: {len(matches)}")

    src_pts = np.float32([pts1[m.queryIdx] for m in matches])
    dst_pts = np.float32([pts2[m.trainIdx] for m in matches])

    H_our, inliers_our = ransac_homography(src_pts, dst_pts)
    print(f"Our SIFT + RANSAC inliers: {len(inliers_our)}")

    # ---------- OPENCV SIFT ----------
    sift = cv2.SIFT_create()
    kp1_cv, des1_cv = sift.detectAndCompute(img1, None)
    kp2_cv, des2_cv = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_cv = bf.knnMatch(des1_cv, des2_cv, k=2)
    good_cv = []
    for m, n in matches_cv:
        if m.distance < 0.75 * n.distance:
            good_cv.append(m)

    print(f"OpenCV SIFT: {len(kp1_cv)} kps in img1, {len(kp2_cv)} in img2")
    print(f"OpenCV SIFT good matches: {len(good_cv)}")

    src_pts_cv = np.float32([kp1_cv[m.queryIdx].pt for m in good_cv])
    dst_pts_cv = np.float32([kp2_cv[m.trainIdx].pt for m in good_cv])
    H_cv, inliers_cv = ransac_homography(src_pts_cv, dst_pts_cv)
    print(f"OpenCV SIFT + RANSAC inliers: {len(inliers_cv)}")

    # ---------- VISUALIZE ----------
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Draw our SIFT keypoints
    for (x, y) in pts1:
        cv2.circle(img1_color, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite("our_sift_keypoints.jpg", img1_color)

    # Draw OpenCV SIFT keypoints
    img1_cv_kps = cv2.drawKeypoints(img1, kp1_cv, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite("opencv_sift_keypoints.jpg", img1_cv_kps)
