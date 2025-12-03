import cv2
import numpy as np

# --- Load calibration data ---
data = np.load("calibration_data_mac.npz")
mtx, dist = data["mtx"], data["dist"]
f = (mtx[0,0] + mtx[1,1]) / 2
print(f"Using focal length f = {f:.2f} pixels")

# --- Load and undistort image ---
img = cv2.imread("object.jpg")
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

cv2.imwrite("object_mac_undistorted.jpg", undistorted)
print("Saved undistorted image → object_mac_undistorted.jpg")

# --- Click two points to measure pixel distance ---
points = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(undistorted, (x, y), 5, (0,0,255), -1)
        cv2.imshow("Measure", undistorted)
        if len(points) == 2:
            cv2.line(undistorted, points[0], points[1], (0,255,0), 2)
            px_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
            print(f"Pixel distance = {px_dist:.2f}")
            Z = float(input("Enter camera-to-object distance (cm): "))
            real_size = (Z * px_dist) / f
            print(f"Estimated real size = {real_size:.2f} cm")

cv2.imshow("Measure", undistorted)
cv2.setMouseCallback("Measure", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
