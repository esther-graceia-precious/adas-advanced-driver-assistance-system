import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------- LOAD MODELS ----------------
binary_model = tf.keras.models.load_model("driver_model.keras", compile=False)
multi_model  = tf.keras.models.load_model("multiclass_model.keras", compile=False)

IMG_SIZE = 224
TEST_DIR = "dataset/test"

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

# ---------- STAGE 1 ----------
binary_pred = binary_model.predict(input_img)[0][0]

if binary_pred < 0.5:
    final_text = f"Normal ({(1 - binary_pred)*100:.2f}%)"
else:
    multi_pred = multi_model.predict(input_img)[0]
    class_id = np.argmax(multi_pred)
    final_text = f"Distracted → {class_names[class_id]} ({multi_pred[class_id]*100:.2f}%)"

# ---------------- DISPLAY ----------------
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(final_text)
plt.axis("off")
plt.show()
