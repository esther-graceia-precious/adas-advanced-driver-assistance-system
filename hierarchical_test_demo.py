import os


import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------- LOAD MODELS ----------------
binary_model = tf.keras.models.load_model("driver_model.keras", compile=False)
multi_model  = tf.keras.models.load_model("multiclass_model.keras", compile=False)

IMG_SIZE = 224
TEST_DIR = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\test"
OUTPUT_DIR = "ppt_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class_names = [
    "Texting (Right Hand)",          # c1
    "Talking on Phone (Right)",      # c2
    "Texting (Left Hand)",           # c3
    "Talking on Phone (Left)",       # c4
    "Operating Radio",               # c5
    "Drinking",                      # c6
    "Reaching Behind",               # c7
    "Hair and Makeup",               # c8
    "Talking to Passenger"           # c9
]

# ---------------- PREPROCESS ----------------
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ---------------- RUN PIPELINE ----------------
img_name = os.listdir(TEST_DIR)[0]   # change index for more samples
img_path = os.path.join(TEST_DIR, img_name)

img = cv2.imread(img_path)
input_img = preprocess(img)

# ---------- STAGE 1: BINARY ----------
binary_pred = binary_model.predict(input_img)[0][0]

if binary_pred < 0.5:
    final_text = f"Attentive ({(1 - binary_pred)*100:.2f}%)"
else:
    multi_pred = multi_model.predict(input_img)[0]
    class_id = np.argmax(multi_pred)
    final_text = f"Distracted → {class_names[class_id]} ({multi_pred[class_id]*100:.2f}%)"

# ---------------- DISPLAY & SAVE ----------------
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(final_text, fontsize=12)
plt.axis("off")

save_path = os.path.join(OUTPUT_DIR, f"output_{img_name}")
plt.savefig(save_path, bbox_inches="tight")
plt.show()

print(f"✅ Output saved for PPT at: {save_path}")
