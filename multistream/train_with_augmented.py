# multistream/train_with_augmented.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("multistream/results", exist_ok=True)

# Use augmented dataset
DATASET_PATH = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\video_augmented"

# Replace the datagen section with this:
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.3, 1.3],
    channel_shift_range=30.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Train on augmented dataset
train_generator = train_datagen.flow_from_directory(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\video_augmented",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

# Validate on ORIGINAL video frames (not augmented!)
val_generator = val_datagen.flow_from_directory(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\new_dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print("Class indices:", train_generator.class_indices)
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Focal loss
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

# Class weights
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(),
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'multistream/video_model_nst_v2_best.h5',
        monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        'multistream/results/nst_training_log.csv'
    )
]

# Phase 1
print("\n--- Phase 1: Training top layers ---")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# Phase 2
print("\n--- Phase 2: Fine tuning ---")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=focal_loss(),
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save('multistream/video_model_nst_v2_final.h5')
print("✅ NST v2 augmented model saved!")
# Plot
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
plt.figure()
plt.plot(acc, label='Train')
plt.plot(val_acc, label='Val')
plt.title('NST Augmented Model Accuracy')
plt.legend()
plt.savefig('multistream/results/nst_accuracy.png')
print("✅ Graph saved!")
