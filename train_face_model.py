# train_face_model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from tensorflow.keras.layers import Lambda

BASE = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT"
DATASET_PATH = os.path.join(BASE, "face_dataset_v2")

def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

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
    class_mode='binary', subset='validation', shuffle=False
)

print("Classes:", train_gen.class_indices)
print(f"Train: {train_gen.samples} | Val: {val_gen.samples}")

labels = train_gen.classes
cw = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(cw))

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False
x = base.output
x = GlobalAveragePooling2D()(x)

embedding = Dense(128, activation='relu', name="embedding_layer")(x)
embedding = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(embedding)
x = Dropout(0.4)(embedding)

out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base.input, outputs=out)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=focal_loss(), metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'face_model_best_02.h5',
        monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
]

print("\n--- Phase 1: Frozen base ---")
model.fit(train_gen, validation_data=val_gen, epochs=10,
          class_weight=class_weights, callbacks=callbacks)

print("\n--- Phase 2: Fine-tuning ---")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss=focal_loss(), metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=10,
          class_weight=class_weights, callbacks=callbacks)

model.save('face_model_final_02.h5')
print("✅ Face model saved!")