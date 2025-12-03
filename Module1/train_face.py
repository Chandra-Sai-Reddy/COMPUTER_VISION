import cv2, os, numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Create folder for your face samples
os.makedirs("faces", exist_ok=True)

# Capture ~25 images of yourself
cap = cv2.VideoCapture(0)
print("📸 Collecting your face samples. Press ESC to stop.")
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"faces/{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("Collecting Samples", frame)
    if cv2.waitKey(1) & 0xFF == 27 or count >= 25:
        break
cap.release()
cv2.destroyAllWindows()

# Train recognizer and save model
samples, labels = [], []
for f in os.listdir("faces"):
    img = cv2.imread(os.path.join("faces", f), cv2.IMREAD_GRAYSCALE)
    samples.append(img)
    labels.append(1)  # 1 = your face

recognizer.train(samples, np.array(labels))
recognizer.save("my_face_model.xml")
print("✅ Model trained and saved as my_face_model.xml")
