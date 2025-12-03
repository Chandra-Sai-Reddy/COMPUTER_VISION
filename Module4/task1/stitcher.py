import cv2
import numpy as np
import sys

print(f"Using OpenCV version: {cv2.__version__}")

def stitch_images(img_bottom, img_top):
    """
    Stitches two images together vertically.
    img_bottom is the base image, img_top is the image to be warped and added above.
    Returns the stitched and cropped image, or None if stitching fails.
    """
    
    print(f"Stitching new pair...")
    
    # --- Step 2: Find Keypoints and Descriptors ---
    sift = cv2.SIFT_create()
    kp_top, des_top = sift.detectAndCompute(img_top, None)
    kp_bottom, des_bottom = sift.detectAndCompute(img_bottom, None)

    print(f"Found {len(kp_top)} keypoints in top image")
    print(f"Found {len(kp_bottom)} keypoints in bottom image")

    # --- Step 3: Match Features ---
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    if des_top is None or des_bottom is None:
        print("Error: No descriptors found for one or both images.")
        return None
        
    matches = bf.knnMatch(des_bottom, des_top, k=2)

    # --- Step 4: Filter Matches with Lowe's Ratio Test ---
    good_matches = []
    if not matches or len(matches) < 2:
        print("Warning: Not enough matches found.")
    else:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
    print(f"Found {len(good_matches)} good matches after ratio test.")

    MIN_MATCH_COUNT = 4
    if len(good_matches) <= MIN_MATCH_COUNT:
        print(f"Not enough good matches found - {len(good_matches)}/{MIN_MATCH_COUNT}")
        return None

    # --- Step 5: Find Homography ---
    src_pts = np.float32([ kp_bottom[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp_top[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print("Homography matrix found.")

    # --- Step 6: Warp and Stitch ---
    h_top, w_top = img_top.shape[:2]
    h_bottom, w_bottom = img_bottom.shape[:2]
    
    canvas_height = h_top + h_bottom
    canvas_width = max(w_top, w_bottom) 
    
    dst = cv2.warpPerspective(img_bottom, H, (canvas_width, canvas_height))
    dst[0:h_top, 0:w_top] = img_top
    
    # --- Step 7: Crop black padding ---
    print("Cropping...")
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_result = dst[y:y+h, x:x+w]
        return cropped_result
    else:
        return dst # Return uncropped if contour finding fails

# --- Main Program Logic ---

# Define the order of images from bottom to top
image_paths = [
    'images/image1.jpeg', # Bottom
    'images/image2.jpeg',
    'images/image3.jpeg',
    'images/image4.jpeg'  # Top
]

# Read the first (bottom-most) image
current_panorama = cv2.imread(image_paths[0])
if current_panorama is None:
    print(f"Error: Could not read base image {image_paths[0]}")
    sys.exit()

print(f"Loaded base image: {image_paths[0]}")

# Loop through the rest of the images and stitch them on top
for i in range(1, len(image_paths)):
    path_top = image_paths[i]
    img_top = cv2.imread(path_top)
    
    if img_top is None:
        print(f"Error: Could not read top image {path_top}")
        continue
        
    print(f"\n--- Stitching {path_top} onto panorama ---")
    
    # We call our function
    # current_panorama is the "bottom", img_top is the "top"
    result = stitch_images(current_panorama, img_top)
    
    if result is not None:
        print("Stitch successful.")
        current_panorama = result # The result becomes the new "bottom"
        cv2.imwrite(f"temp_pano_{i}.jpg", current_panorama) # Save intermediate steps
    else:
        print(f"Stitching failed for {path_top}. Aborting.")
        sys.exit()

# Save the final result
cv2.imwrite("final_panorama.jpg", current_panorama)
print("\n---")
print("All images stitched! Final result saved as 'final_panorama.jpg'")