import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace

# Folders to store embeddings and temporary images
DATA_DIR = "face_db"
TMP_DIR = "temps"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "embeddings.pkl")

def capture_images(num=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return []

    print(f"Press 'c' to capture {num} images of your face.")
    images = []

    while len(images) < num:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Register - press 'c' to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            img_path = os.path.join(TMP_DIR, f"capture_{len(images)}.jpg")
            cv2.imwrite(img_path, frame)
            images.append(img_path)
            print(f"Captured image {len(images)}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return images

def get_embedding(img_path, model_name="Facenet"):
    try:
        rep = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=True)
        if isinstance(rep, list) and len(rep) > 0:
            rep = rep[0]
        if isinstance(rep, dict) and "embedding" in rep:
            return np.array(rep["embedding"])
        elif isinstance(rep, (list, tuple, np.ndarray)):
            return np.array(rep)
    except Exception as e:
        print(f"Warning: Could not process {img_path} -> {e}")
    return None

def save_embedding(name, embedding):
    db = {}
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            db = pickle.load(f)
    db[name] = embedding.tolist()
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)
    print(f"Saved embedding for {name}")

if __name__ == "__main__":
    username = input("Enter your name for registration: ").strip()
    images = capture_images(num=5)
    if not images:
        print("No images captured. Exiting.")
        exit(1)

    embeddings = []
    for img in images:
        emb = get_embedding(img)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        print("Failed to generate embeddings. Try again with better lighting.")
        exit(1)

    mean_embedding = np.mean(np.vstack(embeddings), axis=0)
    save_embedding(username, mean_embedding)
    print("Registration complete!")
