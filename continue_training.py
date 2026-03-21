# continue_training.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

model = tf.keras.models.load_model(
    r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\video_model_86_best.h5",
    custom_objects={'loss': focal_loss()},
    compile=False
)
print("✅ Model loaded - continuing from 81.9%")

# Unfreeze last 50 layers of the full model
# Total layers = 158, so freeze first 108, unfreeze last 50
for layer in model.layers[:108]:
    layer.trainable = False
for layer in model.layers[108:]:
    layer.trainable = True

trainable = sum([1 for l in model.layers if l.trainable])
print(f"Trainable layers: {trainable}")

# Data
train_datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2,
    rotation_range=15, zoom_range=0.2,
    horizontal_flip=True, brightness_range=[0.3, 1.3],
    channel_shift_range=30.0, fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

DATASET = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\new_dataset"

train_generator = train_datagen.flow_from_directory(
    DATASET, target_size=(224, 224), batch_size=32,
    class_mode='binary', subset='training', shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    DATASET, target_size=(224, 224), batch_size=32,
    class_mode='binary', subset='validation', shuffle=False
)

# Compile with very low lr
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),
    loss=focal_loss(),
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "video_model_86_best.h5",
        monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5,
        patience=3, verbose=1, min_lr=1e-8
    )
]

print("\n--- Continuing fine tuning from 81.9% ---")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks
)

model.save("video_model_86_continued.h5")
print("✅ Saved as video_model_86_continued.h5")