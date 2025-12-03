import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# 1. CALIBRATION LOADING (from stereo_measure.py)
# ==========================================

def load_calibration(calib_path):
    """
    Load calibration data and return focal length f in pixels.

    Expects an npz file with:
        'mtx' : 3x3 camera matrix
        'dist': distortion coefficients (ignored here)

    If loading fails, falls back to 1250.0 (like your stereo_measure.py).
    """
    try:
        calibration_data = np.load(calib_path)
        camera_matrix = calibration_data["mtx"]
        f = camera_matrix[0, 0]  # or average of mtx[0,0], mtx[1,1]
        calibration_data.close()
        print(f"[stereo_pose] Loaded calibration. F = {f:.2f} px")
        return f
    except Exception as e:
        print(f"[stereo_pose] Calibration Error: {e}")
        # Fallback like your original code
        f = 1250.0
        print("[stereo_pose] WARNING: Using dummy focal length 1250.0 for testing.")
        return f


# ==========================================
# 2. STEREO SIZE ESTIMATION (from stereo_measure.py)
# ==========================================

def calculate_stereo_metrics(points_L, points_R, baseline_cm, f):
    """
    Given:
        points_L: [(xL1,yL1), (xL2,yL2), (xL3,yL3)] in LEFT image
        points_R: [(xR1,yR1), (xR2,yR2), (xR3,yR3)] in RIGHT image
        baseline_cm: distance between the two cameras (in cm)
        f: focal length in pixels

    Compute:
        - Average depth Z (using disparities of all 3 points)
        - Real width (between P1->P2 in left image)
        - Real height (between P1->P3 in left image)

    Returns:
        (results_dict, message)
        where results_dict has keys: 'z_avg', 'real_w', 'real_h'
    """
    try:
        if f is None:
            return None, "Focal length is None. Load calibration first."

        if len(points_L) != 3 or len(points_R) != 3:
            return None, "Need exactly 3 points in each image."

        # Unpack points similar to your stereo_measure.py
        (xL1, yL1) = points_L[0]
        (xR1, yR1) = points_R[0]  # Top-Left

        (xL2, yL2) = points_L[1]
        (xR2, yR2) = points_R[1]  # Top-Right

        (xL3, yL3) = points_L[2]
        (xR3, yR3) = points_R[2]  # Bottom-Left

         # Disparities
        d1 = abs(xL1 - xR1)
        d2 = abs(xL2 - xR2)
        d3 = abs(xL3 - xR3)

        eps = 1e-6  # small threshold for numerical safety

        depths = []
        for d in [d1, d2, d3]:
            if d > eps:
                depths.append((f * baseline_cm) / d)

        if len(depths) < 2:
            return None, "Not enough valid disparities (too close to 0). Select better points."

        z_avg = sum(depths) / len(depths)


        if 0 in [d1, d2, d3]:
            return None, "ERROR: Disparity is 0 for one of the points. Try again."

        # Depth per point: Z = f * B / d
        z1 = (f * baseline_cm) / d1
        z2 = (f * baseline_cm) / d2
        z3 = (f * baseline_cm) / d3
        z_avg = (z1 + z2 + z3) / 3.0

        # Pixel distances in LEFT image
        pix_w = np.sqrt((xL1 - xL2) ** 2 + (yL1 - yL2) ** 2)  # width P1->P2
        pix_h = np.sqrt((xL1 - xL3) ** 2 + (yL1 - yL3) ** 2)  # height P1->P3

        # Real dimensions
        real_w = (pix_w * z_avg) / f
        real_h = (pix_h * z_avg) / f

        results = {
            "z_avg": round(float(z_avg), 2),
            "real_w": round(float(real_w), 2),
            "real_h": round(float(real_h), 2),
        }
        msg = "Stereo metrics computed successfully."
        return results, msg

    except Exception as e:
        return None, f"Calculation Error: {e}"


# ==========================================
# 3. MEDIAPIPE POSE TRACKING (from pose_estimation.py)
# ==========================================

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Precompute landmark names for pose and hands (like your scripts)
POSE_LANDMARK_NAMES = [name.name for name in mp_pose.PoseLandmark]
HAND_LANDMARK_NAMES = [name.name for name in mp_hands.HandLandmark]


def process_pose(frame_bgr, pose_model):
    """
    Process a single BGR frame with a MediaPipe Pose model.

    Args:
        frame_bgr : BGR image (OpenCV frame)
        pose_model: an instance of mp.solutions.pose.Pose

    Returns:
        annotated_frame_bgr : frame with pose landmarks drawn
        data_rows           : list of dicts containing:
            'landmark_id', 'landmark_name', 'x', 'y', 'z', 'visibility'
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = pose_model.process(frame_rgb)
    frame_rgb.flags.writeable = True

    annotated = frame_bgr.copy()
    data_rows = []

    if results.pose_landmarks:
        # Collect data (similar to pose_estimation.py)
        for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = POSE_LANDMARK_NAMES[landmark_id]
            x = landmark.x
            y = landmark.y
            z = landmark.z
            visibility = landmark.visibility

            data_rows.append({
                "type": "pose",
                "landmark_id": landmark_id,
                "landmark_name": landmark_name,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "visibility": float(visibility),
            })

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(245, 66, 230), thickness=2, circle_radius=2
            ),
        )

    return annotated, data_rows


# ==========================================
# 4. MEDIAPIPE HAND TRACKING (from hand_tracking.py)
# ==========================================

def process_hands(frame_bgr, hands_model):
    """
    Process a single BGR frame with a MediaPipe Hands model.

    Args:
        frame_bgr  : BGR image (OpenCV frame)
        hands_model: an instance of mp.solutions.hands.Hands

    Returns:
        annotated_frame_bgr : frame with hand landmarks drawn
        data_rows           : list of dicts containing:
            'hand_id', 'handedness', 'landmark_id',
            'landmark_name', 'x', 'y', 'z'
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = hands_model.process(frame_rgb)
    frame_rgb.flags.writeable = True

    annotated = frame_bgr.copy()
    data_rows = []

    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Handedness (Left/Right), like in your hand_tracking.py
            handedness_str = "Unknown"
            if results.multi_handedness and len(results.multi_handedness) > hand_id:
                handedness_obj = results.multi_handedness[hand_id]
                handedness_str = handedness_obj.classification[0].label

            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                landmark_name = HAND_LANDMARK_NAMES[landmark_id]
                x = landmark.x
                y = landmark.y
                z = landmark.z

                data_rows.append({
                    "type": "hand",
                    "hand_id": hand_id,
                    "handedness": handedness_str,
                    "landmark_id": landmark_id,
                    "landmark_name": landmark_name,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                })

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=2
                ),
            )

    return annotated, data_rows


