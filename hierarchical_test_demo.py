import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ---------------- LOAD MODELS ----------------
binary_model = tf.keras.models.load_model("driver_model.keras")
multi_model = tf.keras.models.load_model("multiclass_model.keras")

IMG_SIZE = 224
TEST_DIR = "dataset/test"

# Class labels for stage 2
class_names = [
    "texting - right",          # c1
    "talking on phone - right", # c2
    "texting - left",           # c3
    "talking on phone - left",  # c4
    "operating radio",          # c5
    "drinking",                 # c6
    "reaching behind",          # c7
    "hair and makeup",          # c8
    "talking to passenger"      # c9
]

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ---------------- PIPELINE ----------------
img_name = os.listdir(TEST_DIR)[0]
img_path = os.path.join(TEST_DIR, img_name)

img = cv2.imread(img_path)
input_img = preprocess(img)

# ---------- STAGE 1: NORMAL vs DISTRACTED ----------
binary_pred = binary_model.predict(input_img)[0][0]

if binary_pred < 0.5:
    label = "Normal"
    confidence = (1 - binary_pred) * 100
    final_text = f"Normal ({confidence:.2f}%)"

else:
    # ---------- STAGE 2: DISTRACTION TYPE ----------
    multi_pred = multi_model.predict(input_img)[0]
    class_id = np.argmax(multi_pred)
    confidence = multi_pred[class_id] * 100
    label = class_names[class_id]
    final_text = f"Distracted → {label} ({confidence:.2f}%)"

# ---------------- DISPLAY ----------------
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(final_text)
plt.axis("off")
plt.show()
