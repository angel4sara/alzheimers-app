import time
import joblib
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from skimage.feature import hog

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "SavedFiles", "model.pkl")


#MODEL_PATH = "../SavedFiles/model.pkl"
IMAGE_HEIGHT, IMAGE_WIDTH = 150, 150
ROI_TOP_LEFT = (50, 50)
ROI_BOTTOM_RIGHT = (100, 100)
HOG_ARGS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=False,
    channel_axis=2,
)

app = Flask(__name__, static_folder="static", template_folder="templates")
model = joblib.load(MODEL_PATH)
model_type = type(model).__name__

def preprocess_image(file_bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image")
    resized = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    x1, y1 = ROI_TOP_LEFT
    x2, y2 = ROI_BOTTOM_RIGHT
    roi = resized[y1:y2, x1:x2]
    hog_feature = hog(roi, **HOG_ARGS)
    X_hog = hog_feature.reshape(1, -1)
    X_roi = roi.reshape(1, -1)
    feats = np.hstack((X_hog, X_roi))
    return feats

def class_label(i: int) -> str:
    return "DEMENTED" if i == 0 else "MILD DEMENTED" if i == 1 else "NON DEMENTED"

# ---------- home page
@app.get("/")                     # Home/intro page
def home():
    return render_template("home.html")

@app.get("/predict")              # Prediction form page
def predict_page():
    return render_template("index.html", model_type=model_type)

@app.post("/predict")
def predict_form():
    f = request.files.get("image")
    if not f:
        return render_template("index.html", model_type=model_type, error="No file uploaded.")
    try:
        t0 = time.time()
        feats = preprocess_image(f.read())
        pred = model.predict(feats)
        label = class_label(int(pred[0]))
        elapsed = round(time.time() - t0, 4)
        return render_template("index.html", model_type=model_type, result=label, elapsed=elapsed)
    except Exception as e:
        return render_template("index.html", model_type=model_type, error=str(e)), 400

@app.post("/api/predict")
def predict_api():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No file uploaded"), 400
    try:
        t0 = time.time()
        feats = preprocess_image(f.read())
        pred = model.predict(feats)
        label = class_label(int(pred[0]))
        elapsed = round(time.time() - t0, 4)
        return jsonify(prediction=label, elapsed_seconds=elapsed, model=model_type)
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == "__main__":
    app.run(debug=True)
