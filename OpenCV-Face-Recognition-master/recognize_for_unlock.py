import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace
import time
import sys

DB_PATH = "face_db/embeddings.pkl"
TMP_DIR = "temps"
os.makedirs(TMP_DIR, exist_ok=True)

THRESHOLD = 0.45  # adjust if recognition is too strict/loose

def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
    # convert lists back to numpy arrays
    for k in db:
        db[k] = np.array(db[k])
    return db

def get_embedding(img_path, model_name="Facenet"):
    try:
        rep = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=True)
        if isinstance(rep, list) and len(rep) > 0:
            rep = rep[0]
        if isinstance(rep, dict) and "embedding" in rep:
            return np.array(rep["embedding"])
        elif isinstance(rep, (list, tuple, np.ndarray)):
            return np.array(rep)
    except Exception:
        return None
    return None

def verify_embedding(emb, db):
    if emb is None or not db:
        return None, None
    best_name, best_dist = None, float("inf")
    for name, d_emb in db.items():
        dist = np.linalg.norm(emb - d_emb)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name, best_dist

def main(timeout_seconds=7):
    db = load_db()
    if not db:
        print("No registered faces found. Run registration first.")
        return 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return 1

    start = time.time()
    print("Scanning for registered face... (look at the camera)")
    while time.time() - start < timeout_seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        tmp_path = os.path.join(TMP_DIR, "attempt.jpg")
        cv2.imwrite(tmp_path, frame)
        emb = get_embedding(tmp_path)
        if emb is not None:
            name, dist = verify_embedding(emb, db)
            if name is not None and dist < THRESHOLD:
                print(f"Verified: {name}")
                cap.release()
                cv2.destroyAllWindows()
                return 0  # success
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("No match.")
    return 1

if __name__ == "__main__":
    rc = main(timeout_seconds=7)
    sys.exit(rc)
