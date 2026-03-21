# generate_results.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.makedirs("video_results", exist_ok=True)

# ================================
# FOCAL LOSS
# ================================
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

# ================================
# LOAD MODEL
# ================================
model = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\video_model_best.h5",
    custom_objects={'loss': focal_loss()},
    compile=False
)
print("✅ Model loaded!")

# ================================
# LOAD VALIDATION DATA
# ================================
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_generator = val_datagen.flow_from_directory(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\new_dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("Class indices:", val_generator.class_indices)
# Should be: {'attentive': 0, 'distracted': 1}

# ================================
# GET PREDICTIONS
# ================================
print("Getting predictions...")
y_pred_probs = model.predict(val_generator, verbose=1)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = val_generator.classes

# ================================
# 1. CONFUSION MATRIX
# ================================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Attentive', 'Distracted'],
            yticklabels=['Attentive', 'Distracted'])
plt.title('Confusion Matrix - Video Model', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('video_results/confusion_matrix.png', dpi=150)
plt.close()
print("✅ Confusion matrix saved!")

# ================================
# 2. CLASSIFICATION REPORT
# ================================
report = classification_report(
    y_true, y_pred,
    target_names=['Attentive', 'Distracted']
)
print("\nClassification Report:")
print(report)
with open('video_results/classification_report.txt', 'w') as f:
    f.write(report)
print("✅ Classification report saved!")

# ================================
# 3. ROC CURVE
# ================================
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Video Model')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('video_results/roc_curve.png', dpi=150)
plt.close()
print("✅ ROC curve saved!")

# ================================
# 4. SAMPLE PREDICTIONS
# ================================
import cv2
import random

dataset_path = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\new_dataset"
classes = ['attentive', 'distracted']

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Sample Predictions - Video Model', fontsize=16)

for row, cls in enumerate(classes):
    folder = os.path.join(dataset_path, cls)
    images = random.sample(os.listdir(folder), 5)
    for col, img_name in enumerate(images):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))
        img_input = img_resized / 255.0
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
        prob = float(model.predict(img_input, verbose=0)[0][0])
        pred = "Distracted" if prob > 0.5 else "Attentive"
        correct = pred.lower() == cls
        color = 'green' if correct else 'red'
        axes[row][col].imshow(img_rgb)
        axes[row][col].set_title(f"Actual: {cls}\nPred: {pred}\n{prob:.2f}",
                                  color=color, fontsize=8)
        axes[row][col].axis('off')

plt.tight_layout()
plt.savefig('video_results/sample_predictions.png', dpi=150)
plt.close()
print("✅ Sample predictions saved!")

print("\n✅ ALL RESULTS GENERATED!")
print("Check your results/ folder:")
print("  - confusion_matrix.png")
print("  - classification_report.txt")
print("  - roc_curve.png")
print("  - sample_predictions.png")