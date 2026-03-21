import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\driver_model.keras"
TEST_DIR = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset"   # ✅ use test or validation folder
IMG_SIZE = 224
BATCH_SIZE = 32

class_names = ["Attentive", "Distracted"]

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ---------------- LOAD DATA ----------------
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ---------------- PREDICTIONS ----------------
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen).ravel()
y_pred = (y_pred_probs > 0.5).astype(int)

# ---------------- CLASSIFICATION REPORT ----------------
print("\n📊 Binary Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix – Binary Driver Distraction")

# ✅ SAVE FOR PPT / REPORT
plt.savefig("binary_confusion_matrix.png", dpi=300)
plt.show()

print("✅ Confusion matrix saved as binary_confusion_matrix.png")
