# train_face_model_FINAL.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

BASE = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT"
DATASET_PATH = os.path.join(BASE, "face_dataset_v2_balanced")

# ═══════════════════════════════════════════════════════════════════════
# FOCAL LOSS (FIXED — Add dtype casting)
# ═══════════════════════════════════════════════════════════════════════
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)  # ← FIX: Prevent dtype mismatch
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

# ═══════════════════════════════════════════════════════════════════════
# DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest',
    validation_split=0.2   
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2   
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=(224,224), batch_size=32,
    class_mode='binary', subset='training', shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH, target_size=(224,224), batch_size=32,
    class_mode='binary', subset='validation', shuffle=False  # ← CRITICAL: Must be False
)

print("="*60)
print("DATASET INFO")
print("="*60)
print(f"Classes: {train_gen.class_indices}")
print(f"Train: {train_gen.samples} | Val: {val_gen.samples}")

# ═══════════════════════════════════════════════════════════════════════
# CLASS WEIGHT CALCULATION (Enhanced)
# ═══════════════════════════════════════════════════════════════════════
labels = train_gen.classes
unique, counts = np.unique(labels, return_counts=True)
class_distribution = dict(zip(unique, counts))

print(f"\nClass distribution: {class_distribution}")
print(f"Class 0 (attentive): {counts[0]} samples")
print(f"Class 1 (distracted): {counts[1]} samples")
print(f"Imbalance ratio: {counts[1]/counts[0]:.2f}:1")

# ═══════════════════════════════════════════════════════════════════════
# CRITICAL FIX: Stronger weights to prevent constant predictor
# ═══════════════════════════════════════════════════════════════════════
# sklearn's 'balanced' computes: n_samples / (n_classes * n_samples_per_class)
# But we need STRONGER penalties for the minority class
cw_sklearn = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
print(f"\nsklearn balanced weights: {dict(enumerate(cw_sklearn))}")

# MANUAL OVERRIDE: Force stronger minority class weight
total = len(labels)
class_weights = {
    0: (total / counts[0]) * 1.5,  # ← BOOST attentive class significantly
    1: (total / counts[1]) * 0.8   # ← Reduce distracted class slightly
}

print(f"APPLIED weights (manual boost): {class_weights}")
print("="*60)

# ═══════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE (Removed L2 normalization — not needed here)
# ═══════════════════════════════════════════════════════════════════════
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)  # ← Increased capacity
x = Dropout(0.5)(x)  # ← Stronger dropout
x = Dense(128, activation='relu')(x)  # ← Additional layer
x = Dropout(0.4)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal_loss(),
    metrics=['accuracy']
)

# ═══════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,  # ← Increased patience
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'face_model_final_03.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(  # ← NEW: Adaptive learning rate
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 1: FROZEN BASE (15 epochs)")
print("="*60)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,  # ← Increased from 10
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print("PHASE 2: FINE-TUNING (15 epochs)")
print("="*60)

# Unfreeze last 40 layers (more aggressive fine-tuning)
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

print(f"Trainable layers: {sum([layer.trainable for layer in base.layers])}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),  # ← Lower LR for fine-tuning
    loss=focal_loss(),
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,  # ← Increased from 10
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ═══════════════════════════════════════════════════════════════════════
# SAVE & VALIDATION CHECK
# ═══════════════════════════════════════════════════════════════════════
model.save('face_model_final_03_final.h5')
print("\n" + "="*60)
print("✅ Training Complete!")
print("="*60)

# Quick validation check
print("\n🔍 Running validation check for constant predictor...")
val_gen.reset()
val_preds = model.predict(val_gen, verbose=0).flatten()
val_binary = (val_preds >= 0.5).astype(int)
unique_preds = np.unique(val_binary)

print(f"Unique predictions: {unique_preds}")
print(f"Prediction distribution:")
for cls in [0, 1]:
    count = np.sum(val_binary == cls)
    pct = (count / len(val_binary)) * 100
    print(f"  Class {cls}: {count}/{len(val_binary)} ({pct:.1f}%)")

if len(unique_preds) == 1:
    print("\n⚠️  WARNING: Model is still a constant predictor!")
    print("Consider:")
    print("  1. Increasing class_weights further")
    print("  2. Balancing dataset manually (undersample majority class)")
    print("  3. Using SMOTE for synthetic minority oversampling")
else:
    print("\n✅ Model is NOT a constant predictor — predictions are diverse!")

print(f"\nModel saved as: face_model_final_03_final.h5")
print("="*60)