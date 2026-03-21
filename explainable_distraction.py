import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ------------------ LOAD MODEL ------------------
model = load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\driver_model.keras",
    compile=False
)


IMG_SIZE = 224
TEST_DIR = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\test"

# ------------------ LOAD HAAR CASCADES ------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ------------------ EXPLAINABLE LOGIC ------------------
def analyze_distraction_reason(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    reasons = []

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "Driver not facing road"

    for (x, y, fw, fh) in faces:
        face_center_x = x + fw / 2
        image_center_x = w / 2

        # Head direction
        if abs(face_center_x - image_center_x) > 80:
            reasons.append("Head turned away")

        roi_gray = gray[y:y+fh, x:x+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            reasons.append("Eyes not clearly visible")

        # Heuristic: hand near face (talking)
        face_bottom = y + fh
        if face_bottom < h * 0.45:
            reasons.append("Hand near face (talking / phone use)")

        # Heuristic: one-hand driving (arm angle proxy)
        if fw < 0.25 * w:
            reasons.append("Possible one-hand driving")

    if not reasons:
        return "General distraction detected"

    return " & ".join(set(reasons))


# ------------------ PREDICTION FUNCTION ------------------
def predict_image(img_path):
    img = cv2.imread(img_path)
    original_img = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]

    label = "Distracted" if prediction > 0.5 else "Attentive"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    reason = None
    if label == "Distracted":
        reason = analyze_distraction_reason(original_img)

    return label, confidence, reason, original_img

# ------------------ TEST MULTIPLE IMAGES ------------------
test_images = os.listdir(TEST_DIR)[0:1]

for img_name in test_images:
    path = os.path.join(TEST_DIR, img_name)

    label, conf, reason, img = predict_image(path)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    title = f"{label} ({conf*100:.2f}%)"
    if reason:
        title += f"\nReason: {reason}"

    plt.title(title)
    plt.axis("off")
    plt.show()
