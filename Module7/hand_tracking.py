import cv2
import mediapipe as mp
import csv
import time

# --- Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, # Track up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Webcam input
cap = cv2.VideoCapture(0)

# CSV file setup
csv_file_name = 'hand_data.csv'
frame_id = 0

# --- CSV Header ---
csv_header = [
    'frame_id',    # The video frame number
    'hand_id',     # The ID of the hand (0 or 1)
    'handedness',  # 'Left' or 'Right'
    'landmark_id', # The ID of the hand landmark (0-20)
    'landmark_name', # The human-readable name (e.g., 'WRIST', 'THUMB_TIP')
    'x',           # Normalized x-coordinate (0.0 to 1.0)
    'y',           # Normalized y-coordinate (0.0 to 1.0)
    'z',           # Normalized z-coordinate (approx. depth, 0 is at wrist)
]

# Get the landmark names from Mediapipe
landmark_names = [name.name for name in mp_hands.HandLandmark]

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
        # **** THIS LINE IS FIXED ****
        image.flags.writeable = False
        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        results = hands.process(image)
        
        # Convert the image back to BGR for drawing
        # **** THIS LINE IS FIXED ****
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- Data Saving ---
        if results.multi_hand_landmarks:
            # Loop through each detected hand
            for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness (Left/Right)
                handedness_obj = results.multi_handedness[hand_id]
                handedness_str = handedness_obj.classification[0].label
                
                # Loop through the 21 landmarks of this hand
                for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                    # Get the landmark name
                    landmark_name = landmark_names[landmark_id]
                    
                    # Extract data
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    
                    # Write data to CSV
                    writer.writerow([
                        frame_id,
                        hand_id,
                        handedness_str,
                        landmark_id,
                        landmark_name,
                        x, y, z
                    ])

                # --- Visual Output ---
                # Draw the hand annotations on the image.
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    )

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Mediapipe Hands', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"Done. Hand data saved to {csv_file_name}")