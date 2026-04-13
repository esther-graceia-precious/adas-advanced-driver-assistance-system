# test_driver5_unseen.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import json

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT"
TEST_DATA_PATH = os.path.join(BASE_DIR, "Custom_Dataset", "Driver_5")
MODEL_PATH = os.path.join(BASE_DIR, "video_model_custom_final.h5")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "driver5_unseen_test")

os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ═══════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════════
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)  # Fix dtype issues
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

# ═══════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════
print("="*70)
print("DRIVER 5 UNSEEN SUBJECT TEST")
print("="*70)
print("\n🚀 Loading Custom Model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'loss': focal_loss()},
    compile=False
)
print(f"✅ Model loaded: {os.path.basename(MODEL_PATH)}")

# ═══════════════════════════════════════════════════════════════════════
# LOAD TEST DATA (Flatten all distracted sub-folders)
# ═══════════════════════════════════════════════════════════════════════
print("\n📂 Loading Driver 5 test data...")

X_test = []
y_true = []
image_paths = []

categories = {
    'attentive': 0,
    'distracted': 1
}

for cat_name, label in categories.items():
    cat_path = os.path.join(TEST_DATA_PATH, cat_name)
    
    if not os.path.exists(cat_path):
        print(f"⚠️ Warning: {cat_path} not found — skipping")
        continue
    
    # Walk through all sub-folders (c0_attentive, c1_phone, c6_drinking, etc.)
    for root, dirs, files in os.walk(cat_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                
                try:
                    # Load and preprocess
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(IMG_SIZE)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    X_test.append(img_array)
                    y_true.append(label)
                    image_paths.append(img_path)
                    
                except Exception as e:
                    print(f"❌ Error loading {img_path}: {e}")
                    continue

X_test = np.array(X_test)
y_true = np.array(y_true)

print(f"✅ Loaded {len(X_test)} images")
print(f"   Attentive (Class 0): {np.sum(y_true == 0)}")
print(f"   Distracted (Class 1): {np.sum(y_true == 1)}")

if len(X_test) == 0:
    print("\n❌ ERROR: No test images found!")
    print(f"Check if images exist in: {TEST_DATA_PATH}")
    exit(1)

# ═══════════════════════════════════════════════════════════════════════
# RUN PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n📊 Running predictions...")

probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1).flatten()
y_pred = (probs >= 0.5).astype(int)

# ═══════════════════════════════════════════════════════════════════════
# CALCULATE METRICS
# ═══════════════════════════════════════════════════════════════════════
accuracy = accuracy_score(y_true, y_pred)

# ROC curve
fpr, tpr, _ = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"\nConfusion Matrix:")
print(f"                Pred: Attentive    Pred: Distracted")
print(f"True: Attentive        {cm[0,0]:<15} {cm[0,1]}")
print(f"True: Distracted       {cm[1,0]:<15} {cm[1,1]}")

# Classification report
print("\n" + classification_report(y_true, y_pred, 
                                    target_names=['Attentive', 'Distracted'],
                                    digits=4))

# ═══════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════
# 1. JSON metrics
metrics = {
    'accuracy': float(accuracy * 100),
    'roc_auc': float(roc_auc),
    'confusion_matrix': cm.tolist(),
    'total_samples': int(len(X_test)),
    'attentive_samples': int(np.sum(y_true == 0)),
    'distracted_samples': int(np.sum(y_true == 1))
}

with open(os.path.join(RESULTS_DIR, "driver5_metrics.json"), 'w') as f:
    json.dump(metrics, f, indent=2)

# 2. Text report
with open(os.path.join(RESULTS_DIR, "driver5_report.txt"), 'w') as f:
    f.write("="*70 + "\n")
    f.write("DRIVER 5 UNSEEN SUBJECT TEST RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Model: {os.path.basename(MODEL_PATH)}\n")
    f.write(f"Test Data: {TEST_DATA_PATH}\n")
    f.write(f"Total Images: {len(X_test)}\n")
    f.write(f"  Attentive: {np.sum(y_true == 0)}\n")
    f.write(f"  Distracted: {np.sum(y_true == 1)}\n\n")
    f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"                Pred: Attentive    Pred: Distracted\n")
    f.write(f"True: Attentive        {cm[0,0]:<15} {cm[0,1]}\n")
    f.write(f"True: Distracted       {cm[1,0]:<15} {cm[1,1]}\n\n")
    f.write(classification_report(y_true, y_pred, 
                                   target_names=['Attentive', 'Distracted'],
                                   digits=4))

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n📈 Generating visualizations...")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Attentive', 'Distracted'],
            yticklabels=['Attentive', 'Distracted'],
            cbar_kws={'label': 'Count'})
plt.title('Driver 5 Confusion Matrix\n(Unseen Subject — Cross-Person Generalization)', 
          fontsize=13, fontweight='bold')
plt.ylabel('True Label', fontsize=11)
plt.xlabel('Predicted Label', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "driver5_confusion_matrix.png"), dpi=150)
plt.close()

# 2. ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#1F4E79', linewidth=2, 
         label=f'Custom Model (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.500)')
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('ROC Curve — Driver 5 (Unseen Subject)', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "driver5_roc_curve.png"), dpi=150)
plt.close()

# 3. Prediction Samples Grid
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

# Get random sample indices (balanced if possible)
att_indices = np.where(y_true == 0)[0]
dis_indices = np.where(y_true == 1)[0]

sample_indices = []
if len(att_indices) >= 6:
    sample_indices.extend(np.random.choice(att_indices, 6, replace=False))
else:
    sample_indices.extend(att_indices)
    
if len(dis_indices) >= 6:
    sample_indices.extend(np.random.choice(dis_indices, 6, replace=False))
else:
    sample_indices.extend(dis_indices)

np.random.shuffle(sample_indices)
sample_indices = sample_indices[:12]

for i, idx in enumerate(sample_indices):
    axes[i].imshow(X_test[idx])
    
    true_label = "Attentive" if y_true[idx] == 0 else "Distracted"
    pred_label = "Attentive" if y_pred[idx] == 0 else "Distracted"
    conf = probs[idx] if y_pred[idx] == 1 else (1 - probs[idx])
    
    correct = (y_true[idx] == y_pred[idx])
    color = '#27AE60' if correct else '#E74C3C'
    
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({conf*100:.1f}%)",
                      color=color, fontsize=10, fontweight='bold')
    axes[i].axis('off')

plt.suptitle('Driver 5 Prediction Samples\n(Green = Correct, Red = Incorrect)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "driver5_prediction_samples.png"), dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# ANALYZE ERRORS
# ═══════════════════════════════════════════════════════════════════════
print("\n🔍 Analyzing errors...")

false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]

print(f"\nFalse Positives (Attentive predicted as Distracted): {len(false_positives)}")
if len(false_positives) > 0:
    print("  Sample files:")
    for idx in false_positives[:5]:
        print(f"    {os.path.basename(image_paths[idx])} (conf: {probs[idx]:.2f})")

print(f"\nFalse Negatives (Distracted predicted as Attentive): {len(false_negatives)}")
if len(false_negatives) > 0:
    print("  Sample files:")
    for idx in false_negatives[:5]:
        print(f"    {os.path.basename(image_paths[idx])} (conf: {probs[idx]:.2f})")

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)
print(f"\nResults saved to: {RESULTS_DIR}")
print("\nGenerated files:")
print("  - driver5_metrics.json")
print("  - driver5_report.txt")
print("  - driver5_confusion_matrix.png")
print("  - driver5_roc_curve.png")
print("  - driver5_prediction_samples.png")
print("\n" + "="*70)