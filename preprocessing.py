import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH = 32

# Training generator with augmentation + split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2   # <-- IMPORTANT
)

# Validation generator (NO augmentation)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_generator = train_datagen.flow_from_directory(
    "dataset/",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary",
    subset="training"
)

# Validation data
val_generator = val_datagen.flow_from_directory(
    "dataset/",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary",
    subset="validation"
)

print("Class indices:", train_generator.class_indices)
