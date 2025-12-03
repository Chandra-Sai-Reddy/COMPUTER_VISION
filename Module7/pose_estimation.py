import cv2
import mediapipe as mp
import csv
import time

# --- Setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Webcam input
cap = cv2.VideoCapture(0)

# CSV file setup
csv_file_name = 'pose_data.csv'
frame_id = 0

# --- CSV Header ---
# This is the explanation of the data required by the assignment
csv_header = [
    'frame_id',    # The video frame number
    'landmark_id', # The ID of the pose landmark (0-32)
    'landmark_name', # The human-readable name (e.g., 'NOSE', 'LEFT_SHOULDER')
    'x',           # Normalized x-coordinate (0.0 to 1.0)
    'y',           # Normalized y-coordinate (0.0 to 1.0)
    'z',           # Normalized z-coordinate (approx. depth, 0 is at hips)
    'visibility'   # Visibility of the landmark (0.0 to 1.0)
]

# Get the landmark names from Mediapipe
landmark_names = [name.name for name in mp_pose.PoseLandmark]

with open(csv_file_name, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header) # Write the header row

    # --- Main Loop ---
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        frame_id += 1

        # To improve performance, optionally mark the image as not writeable
        image.flags.writeable = False
        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find pose
        results = pose.process(image)
        
        # Convert the image back to BGR for drawing
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- Data Saving ---
        if results.pose_landmarks:
            for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                # Get the landmark name
                landmark_name = landmark_names[landmark_id]
                
                # Extract data
                x = landmark.x
                y = landmark.y
                z = landmark.z
                visibility = landmark.visibility
                
                # Write data to CSV
                writer.writerow([
                    frame_id,
                    landmark_id,
                    landmark_name,
                    x, y, z,
                    visibility
                ])

            # --- Visual Output ---
            # Draw the pose annotation on the image.
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Mediapipe Pose', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"Done. Pose data saved to {csv_file_name}")