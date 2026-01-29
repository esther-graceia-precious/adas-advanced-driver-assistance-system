import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIG ----------------
MODEL_PATH = "driver_model.keras"
TEST_DIR = "dataset"
   # or validation/test folder
IMG_SIZE = 224
BATCH_SIZE = 32

class_names = ["Normal", "Distracted"]

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

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

# ---------------- METRICS ----------------
print("\n📊 Binary Classification Report:\n")
print(classification_report(
    y_true, y_pred,
    target_names=class_names
))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Binary Driver Distraction")
plt.tight_layout()
plt.show()
