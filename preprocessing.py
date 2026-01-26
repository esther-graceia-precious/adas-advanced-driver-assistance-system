import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image parameters (CPU friendly)
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Data augmentation (light – safe for CPU)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

# Training data
train_data = train_datagen.flow_from_directory(
    "dataset/",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

# Validation data
val_data = train_datagen.flow_from_directory(
    "dataset/",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

print("Class indices:", train_data.class_indices)
