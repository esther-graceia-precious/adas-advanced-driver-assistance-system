# model.py - FIXED VERSION
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

from preprocessing import train_generator, val_generator

os.makedirs("results", exist_ok=True)

# -----------------------------
# FOCAL LOSS
# -----------------------------
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# -----------------------------
# MODEL
# -----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(),
    metrics=['accuracy']
)

# -----------------------------
# CALLBACKS
# -----------------------------
callbacks = [
    CSVLogger("results/training_log.csv", append=False),
    EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
    
    # ✅ THIS IS THE KEY FIX - saves best model as .h5
    ModelCheckpoint(
        "driver_model_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks
)

# -----------------------------
# SAVE FINAL MODEL - .h5 FORMAT ✅
# -----------------------------
model.save("driver_model_final.h5")
print("✅ Model saved as driver_model_final.h5")

# -----------------------------
# GRAPHS
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/accuracy_curve.png')

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/loss_curve.png')

model.summary()