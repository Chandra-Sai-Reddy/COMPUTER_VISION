import cv2
import numpy as np
import sys

# --- 1. LOAD CALIBRATION ---
try:
    # Change filename if needed
    calibration_data = np.load('calibration_data_mac.npz') 
    camera_matrix = calibration_data['mtx']
    FOCAL_LENGTH_PX = camera_matrix[0, 0]
    calibration_data.close()
except Exception as e:
    print(f"Calibration Error: {e}")
    # Fallback for testing if file missing (Remove this for real submission)
    FOCAL_LENGTH_PX = 1250.0 
    print("WARNING: Using dummy focal length 1250.0 for testing.")

# --- CONFIGURATION ---
BASELINE_CM = 28.0  # <--- CHECK THIS
IMG_PATH_LEFT = 'data/left.jpg'   # <--- CHECK PATHS
IMG_PATH_RIGHT = 'data/right.jpg' 

# Global State
points_left = []
points_right = []
step_counter = 0 # 0 to 5
img_left_display = None
img_right_display = None

def redraw_images():
    """Updates the images with text and circles based on current state"""
    global img_left_display, img_right_display
    
    # specific instructions based on step
    msg_L = ""
    msg_R = ""
    
    if step_counter == 0: msg_L = "1. Click Top-Left"
    elif step_counter == 1: msg_R = "2. Click Top-Left"
    elif step_counter == 2: msg_L = "3. Click Top-Right"
    elif step_counter == 3: msg_R = "4. Click Top-Right"
    elif step_counter == 4: msg_L = "5. Click Bottom-Left"
    elif step_counter == 5: msg_R = "6. Click Bottom-Left"
    elif step_counter == 6: msg_L = "DONE! See Console."; msg_R = "DONE! See Console."

    # Copy originals so we don't draw over and over
    img_L_temp = img_left_display.copy()
    img_R_temp = img_right_display.copy()

    # Draw Points
    for pt in points_left: cv2.circle(img_L_temp, pt, 5, (0, 255, 0), -1)
    for pt in points_right: cv2.circle(img_R_temp, pt, 5, (0, 0, 255), -1)

    # Draw Text
    cv2.putText(img_L_temp, msg_L, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(img_R_temp, msg_R, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Left Image", img_L_temp)
    cv2.imshow("Right Image", img_R_temp)

def mouse_callback(event, x, y, flags, param):
    global step_counter
    
    if event == cv2.EVENT_LBUTTONDOWN:
        window_name = param
        
        # Logic: Only allow click if it's the correct window for the current step
        valid_click = False
        
        # Steps 0, 2, 4 must be LEFT image
        if window_name == "left" and step_counter in [0, 2, 4]:
            points_left.append((x, y))
            step_counter += 1
            valid_click = True
            
        # Steps 1, 3, 5 must be RIGHT image
        elif window_name == "right" and step_counter in [1, 3, 5]:
            points_right.append((x, y))
            step_counter += 1
            valid_click = True
            
        if valid_click:
            redraw_images()
            # If we just finished the 6th click (step 5 -> 6)
            if step_counter == 6:
                calculate_results()

def calculate_results():
    print("\n" + "="*30)
    print("CALCULATING RESULTS...")
    print("="*30)
    
    try:
        # Unpack points
        (xL1, yL1) = points_left[0]; (xR1, yR1) = points_right[0] # Top-Left
        (xL2, yL2) = points_left[1]; (xR2, yR2) = points_right[1] # Top-Right
        (xL3, yL3) = points_left[2]; (xR3, yR3) = points_right[2] # Bottom-Left
        
        # Disparities
        d1 = abs(xL1 - xR1)
        d2 = abs(xL2 - xR2)
        d3 = abs(xL3 - xR3)
        
        if 0 in [d1, d2, d3]:
            print("ERROR: Disparity is 0 in one of the points. Try again.")
            return

        # Depths
        z1 = (FOCAL_LENGTH_PX * BASELINE_CM) / d1
        z2 = (FOCAL_LENGTH_PX * BASELINE_CM) / d2
        z3 = (FOCAL_LENGTH_PX * BASELINE_CM) / d3
        z_avg = (z1 + z2 + z3) / 3
        
        print(f"Average Depth (Z): {z_avg:.2f} cm")
        
        # Pixel Dimensions (using Left Image)
        # Width: Distance P1 -> P2
        pix_w = np.sqrt((xL1 - xL2)**2 + (yL1 - yL2)**2)
        # Height: Distance P1 -> P3
        pix_h = np.sqrt((xL1 - xL3)**2 + (yL1 - yL3)**2)
        
        # Real Dimensions
        real_w = (pix_w * z_avg) / FOCAL_LENGTH_PX
        real_h = (pix_h * z_avg) / FOCAL_LENGTH_PX
        
        print(f"Real Width:  {real_w:.2f} cm")
        print(f"Real Height: {real_h:.2f} cm")
        print("\nPress 'r' to reset and measure again, or 'q' to quit.")

    except Exception as e:
        print(f"Calculation Error: {e}")

# --- MAIN ---
img_left_display = cv2.imread(IMG_PATH_LEFT)
img_right_display = cv2.imread(IMG_PATH_RIGHT)

if img_left_display is None or img_right_display is None:
    print("Error: Could not load images. Check paths.")
else:
    cv2.imshow("Left Image", img_left_display)
    cv2.imshow("Right Image", img_right_display)
    
    # Pass "left" or "right" as param to know which window was clicked
    cv2.setMouseCallback("Left Image", mouse_callback, param="left")
    cv2.setMouseCallback("Right Image", mouse_callback, param="right")
    
    # Initial Draw
    redraw_images()
    
    print("Controls:")
    print("  Click image to select points (Follow on-screen text)")
    print("  'r' - Reset points")
    print("  'q' - Quit")

    while True:
        key = cv2.waitKey(50) & 0xFF # Wait 50ms (more responsive)
        
        if key == ord('q'):
            print("Exiting...")
            break
            
        if key == ord('r'):
            print("\n--- RESET ---")
            points_left = []
            points_right = []
            step_counter = 0
            redraw_images()

    cv2.destroyAllWindows()
    