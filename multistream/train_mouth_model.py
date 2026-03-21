# multistream/train_mouth_model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

os.makedirs("multistream/results", exist_ok=True)

DATASET_PATH = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset\Image Data\Mouth States"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 3

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.15,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Classes:", train_generator.class_indices)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('multistream/mouth_model_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\n--- Phase 1: Training top layers ---")
history1 = model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=callbacks)

print("\n--- Phase 2: Fine tuning ---")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=callbacks)

model.save('multistream/mouth_model_final.h5')
print("✅ Mouth model saved!")

acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
plt.figure()
plt.plot(acc, label='Train')
plt.plot(val_acc, label='Val')
plt.title('Mouth State Model Accuracy')
plt.legend()
plt.savefig('multistream/results/mouth_accuracy.png')
print("✅ Graph saved!")