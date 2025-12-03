from flask import Flask, render_template, request, jsonify
import cv2, os, numpy as np
import base64

app = Flask(__name__)

# Load calibration
calib = np.load("calibration_data_mac.npz")
mtx = calib["mtx"]
f = (mtx[0,0] + mtx[1,1]) / 2

# LBPH face recognizer
face_cascade = cv2.CascadeClassifier("static/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists("my_face_model.xml"):
    raise FileNotFoundError("Face model missing. Run training first!")

recognizer.read("my_face_model.xml")

# Route: verification page
@app.route("/")
def verify_page():
    return render_template("verify.html")

# Route: receive frame for verification
@app.route("/verify_face", methods=["POST"])
def verify_face():
    data = request.get_json()
    img_data = base64.b64decode(data["image"].split(",")[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    authorized = False
    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if id_ == 1 and conf < 60:
            authorized = True
            break
    return jsonify({"authorized": authorized})

# Route: main measurement page
@app.route("/home")
def home():
    return render_template("index.html")

# Route: measurement API
@app.route("/measure", methods=["POST"])
def measure():
    data = request.get_json()
    Z, px = float(data["Z"]), float(data["px"])
    real_size = (Z * px) / f
    return jsonify({"real_size": round(real_size, 2)})

if __name__ == "__main__":
    app.run(debug=True)
