import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ----------------
MODEL_PATH = "multiclass_model.keras"
DATA_DIR = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multiclass_dataset"
IMG_SIZE = 224
BATCH_SIZE = 32

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ---------------- DATA GENERATOR ----------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2   # MUST MATCH TRAINING
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ---------------- PREDICTIONS ----------------
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=val_gen.class_indices.keys(),
    yticklabels=val_gen.class_indices.keys(),
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Multiclass Confusion Matrix")
plt.savefig("results/multiclass_confusion_matrix.png")
plt.show()

# ---------------- CLASSIFICATION REPORT ----------------
print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=val_gen.class_indices.keys()
))
