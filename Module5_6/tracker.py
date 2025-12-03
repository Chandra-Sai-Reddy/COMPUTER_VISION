import cv2
import numpy as np

# --- 1. MARKER-BASED TRACKER (Placeholder) ---
# In modules/Module5_6/tracker.py

def track_markers_aruco(frame):
    """
    Detects ArUco markers (DICT_4X4_50), draws their borders,
    and tracks the center of the first detected marker.
    """
    if frame is None: return frame

    # 1. Convert to Grayscale (ArUco detection works on gray images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Define the ArUco Dictionary
    # We use 4x4 markers (50 possible IDs). This is standard and fast.
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # 3. Create Detector (New OpenCV 4.7+ syntax)
    # If this fails with AttributeError, use cv2.aruco.detectMarkers(...)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # 4. Detect Markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # 5. Draw Results
    if ids is not None:
        # Draw the square boundary around markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Draw the Center Point of the first marker found
        # corners shape: (N, 1, 4, 2)
        c = corners[0][0] # Get the 4 corners of the first marker
        
        # Calculate center (average of 4 corners)
        cx = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
        cy = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)

        # Draw a crosshair at the center
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"ID: {ids[0][0]} (x={cx}, y={cy})", (cx + 10, cy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Optional: Display 'Tracking Locked' status
        cv2.putText(frame, "Tracking Locked", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Marker Detected", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame

# --- 2. MARKER-LESS TRACKER (CSRT) ---
class MarkerlessTracker:
    def __init__(self, frame, bbox):
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)

    def update(self, frame):
        success, box = self.tracker.update(frame)
        return success, box

# --- 3. SAM2 SEGMENTATION OVERLAY (Fixed for Empty Frames) ---
def overlay_sam2_mask(frame, frame_idx, mask_data):
    key = str(frame_idx)
    
    # 1. Safety Check: Valid Frame
    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
        return frame

    if key in mask_data:
        try:
            mask = mask_data[key]
            
            # 2. Safety Check: Valid Mask Data
            if mask is None or mask.size == 0:
                return frame

            # Fix Dimensions (Squeeze 3D to 2D)
            mask = np.squeeze(mask)
            if mask.ndim > 2: mask = mask[:, :, 0]
            if mask.ndim < 2: return frame

            # Resize to match Frame
            h, w = frame.shape[:2]
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)

            # 3. CRITICAL FIX: The "Dog Out of Frame" Check
            mask_bool = mask_resized > 0
            
            # Count how many pixels are in the mask
            pixel_count = np.count_nonzero(mask_bool)
            
            # If ZERO pixels are found (Dog is gone), STOP immediately.
            if pixel_count == 0:
                return frame

            # 4. Prepare Overlay
            color_mask = np.zeros_like(frame)
            color_mask[:, :] = [0, 255, 0] # Green
            
            # 5. Blend
            # We now know for a fact that 'frame[mask_bool]' is NOT empty
            frame[mask_bool] = cv2.addWeighted(frame[mask_bool], 0.6, color_mask[mask_bool], 0.4, 0)
            
            cv2.putText(frame, f"SAM2 Mask: {key}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            # If anything else goes wrong, print it but DO NOT CRASH
            print(f"Skipping frame {key} error: {e}")
            pass
        
    return frame