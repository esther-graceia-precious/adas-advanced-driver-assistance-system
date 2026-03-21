# train_video_model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

os.makedirs("results", exist_ok=True)

DATASET_PATH = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\new_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ================================
# AUGMENTATION
# ================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.3, 1.3],
    channel_shift_range=30.0,
    shear_range=0.1,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("Class indices:", train_generator.class_indices)

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
# CLASS WEIGHTS
# ================================
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# ================================
# MODEL
# ================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=output)

# ================================
# COMPILE
# ================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(),
    metrics=['accuracy']
)

# ================================
# CALLBACKS
# ================================
callbacks = [
    CSVLogger("results/video_model_training_log.csv", append=False),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "video_model_86_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ================================
# PHASE 1 - Train top layers
# ================================
print("\n--- Phase 1: Training top layers ---")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# ================================
# PHASE 2 - Fine tune
# ================================
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

# ================================
# SAVE
# ================================
model.save("video_model_86_final.h5")
print("✅ Saved as video_model_86_best.h5 and video_model_86_final.h5")

# ================================
# GRAPHS
# ================================
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

plt.figure()
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.title('Video Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/video_model_accuracy.png')
print("✅ Graph saved!")
