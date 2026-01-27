import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------- LOAD MODEL ----------------
model = load_model("driver_model.keras")

TEST_DIR = "dataset/test"
IMG_SIZE = 224


# ---------------- EXPLAINABILITY (NO MEDIAPIPE) ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def analyze_distraction_reason(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    reasons = []

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ---- CASE 1: FACE NOT RELIABLY DETECTED ----
    if len(faces) == 0:
        return "Driver state unclear (possible occlusion / sunglasses)"

    for (x, y, fw, fh) in faces:
        face_center_x = x + fw / 2
        image_center_x = w / 2

        # ---- HEAD TURN CHECK (ONLY IF FACE EXISTS) ----
        if abs(face_center_x - image_center_x) > 90:
            reasons.append("Head turned away from road")

        roi_gray = gray[y:y+fh, x:x+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # ---- EYE VISIBILITY (WEAK SIGNAL) ----
        if len(eyes) == 0:
            reasons.append("Eyes not clearly visible")

        # ---- PHONE / HAND HEURISTIC (MOST IMPORTANT FOR YOUR DATASET) ----
        if y + fh < h * 0.45:
            reasons.append("Hand near face (possible phone usage)")

    # ---- FINAL DECISION ----
    if "Hand near face (possible phone usage)" in reasons:
        return "Phone-related distraction"

    if "Head turned away from road" in reasons:
        return "Driver not facing road"

    if reasons:
        return " & ".join(set(reasons))

    return "General distraction detected"



# ---------------- PREDICTION FUNCTION ----------------
def predict_image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    pred = model.predict(img_input, verbose=0)[0][0]

    if pred >= 0.75:
        label = "Distracted"
    elif pred <= 0.35:
        label = "Attentive"
    else:
        label = "Uncertain"

    confidence = pred if label == "Distracted" else (1 - pred)

    reason = "Monitoring driver"
    if label == "Distracted":
        reason = analyze_distraction_reason(img)

    return img_rgb, label, confidence, pred, reason


# ---------------- RUN DEMO ----------------
test_images = os.listdir(TEST_DIR)[100:110]

for img_name in test_images:
    path = os.path.join(TEST_DIR, img_name)

    img, label, conf, score, reason = predict_image(path)

    plt.imshow(img)
    plt.title(
        f"{label} | Score: {score:.2f}\nReason: {reason}"
    )
    plt.axis("off")
    plt.show()
