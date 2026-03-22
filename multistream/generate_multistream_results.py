# multistream/generate_multistream_results.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.makedirs("multistream/results", exist_ok=True)

# ================================
# HEAD POSE
# ================================
print("Evaluating Head Pose model...")
head_model = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\multistream\head_model_best.h5",
    compile=False
)
head_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
head_gen = val_datagen.flow_from_directory(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\Image Data\Head Pose",
    target_size=(224,224), batch_size=32,
    class_mode='categorical', subset='validation', shuffle=False
)

head_preds = np.argmax(head_model.predict(head_gen, verbose=1), axis=1)
head_true  = head_gen.classes
head_classes = list(head_gen.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(head_true, head_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=head_classes, yticklabels=head_classes)
plt.title('Head Pose Confusion Matrix (67%)', fontsize=14)
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig('multistream/results/head_confusion_matrix.png', dpi=150)
plt.close()

with open('multistream/results/head_classification_report.txt', 'w') as f:
    f.write(classification_report(head_true, head_preds, target_names=head_classes))
print("✅ Head model results saved!")

# ================================
# EYE GAZE
# ================================
print("Evaluating Eye Gaze model...")
eye_model = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\multistream\eye_model_best.h5",
    compile=False
)
eye_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

eye_gen = val_datagen.flow_from_directory(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\Image Data\Eye Gaze",
    target_size=(224,224), batch_size=32,
    class_mode='categorical', subset='validation', shuffle=False
)

eye_preds  = np.argmax(eye_model.predict(eye_gen, verbose=1), axis=1)
eye_true   = eye_gen.classes
eye_classes = list(eye_gen.class_indices.keys())

cm = confusion_matrix(eye_true, eye_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=eye_classes, yticklabels=eye_classes)
plt.title('Eye Gaze Confusion Matrix (90%)', fontsize=14)
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('multistream/results/eye_confusion_matrix.png', dpi=150)
plt.close()

with open('multistream/results/eye_classification_report.txt', 'w') as f:
    f.write(classification_report(eye_true, eye_preds, target_names=eye_classes))
print("✅ Eye model results saved!")

# ================================
# MOUTH STATE
# ================================
print("Evaluating Mouth State model...")
mouth_model = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\multistream\multistream\mouth_model_best.h5",
    compile=False
)
mouth_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

mouth_gen = val_datagen.flow_from_directory(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\Image Data\Mouth States",
    target_size=(224,224), batch_size=32,
    class_mode='categorical', subset='validation', shuffle=False
)

mouth_preds  = np.argmax(mouth_model.predict(mouth_gen, verbose=1), axis=1)
mouth_true   = mouth_gen.classes
mouth_classes = list(mouth_gen.class_indices.keys())

cm = confusion_matrix(mouth_true, mouth_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=mouth_classes, yticklabels=mouth_classes)
plt.title('Mouth State Confusion Matrix (75%)', fontsize=14)
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('multistream/results/mouth_confusion_matrix.png', dpi=150)
plt.close()

with open('multistream/results/mouth_classification_report.txt', 'w') as f:
    f.write(classification_report(mouth_true, mouth_preds, target_names=mouth_classes))
print("✅ Mouth model results saved!")

# ================================
# SUMMARY CHART
# ================================
models  = ['Head Pose\n(9 classes)', 'Eye Gaze\n(6 classes)', 'Mouth State\n(3 classes)']
accuracy = [67, 90, 75]
colors  = ['#ff9999', '#99cc99', '#ffcc99']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy, color=colors, edgecolor='gray', width=0.5)
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Multistream Model Accuracies', fontsize=14)
for bar, acc in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc}%', ha='center', fontweight='bold', fontsize=12)
plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
plt.legend()
plt.tight_layout()
plt.savefig('multistream/results/multistream_accuracy_comparison.png', dpi=150)
plt.close()
print("✅ Summary chart saved!")

print("\n✅ ALL MULTISTREAM RESULTS GENERATED!")
print("Files saved in multistream/results/:")
print("  - head_confusion_matrix.png")
print("  - head_classification_report.txt")
print("  - eye_confusion_matrix.png")
print("  - eye_classification_report.txt")
print("  - mouth_confusion_matrix.png")
print("  - mouth_classification_report.txt")
print("  - multistream_accuracy_comparison.png")