import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- CONFIG ---
DATASET_PATH = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\Custom_Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

# --- DATA PREPARATION ---
def load_custom_data(base_path):
    images = []
    labels = []
    
    # Iterate through Driver_1, Driver_2, etc.
    for driver_folder in os.listdir(base_path):
        driver_dir = os.path.join(base_path, driver_folder)
        if not os.path.isdir(driver_dir): continue
        
        # Iterate through sub-categories (c0, c1, etc.)
        for category in os.listdir(driver_dir):
            cat_dir = os.path.join(driver_dir, category)
            
            # Map labels: c0 is Attentive (0), everything else is Distracted (1)
            label = 0 if 'c0_attentive' in category else 1
            
            for img_path in glob.glob(os.path.join(cat_dir, "*.jpg")):
                try:
                    img = load_img(img_path, target_size=IMG_SIZE)
                    img_array = img_to_array(img) / 255.0  # Normalize
                    images.append(img_array)
                    labels.append(label)
                except:
                    continue
                    
    return np.array(images), np.array(labels)

print("📂 Loading and mapping dataset...")
X, y = load_custom_data(DATASET_PATH)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Loaded {len(X)} images. Train: {len(X_train)}, Val: {len(X_val)}")

# --- FOCAL LOSS (FIXED FOR TYPE MISMATCH) ---
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        # Cast y_true to float32 to match y_pred
        y_true = tf.cast(y_true, tf.float32)
        
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

# --- MODEL BUILDING ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False # Start with frozen base

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer=Adam(learning_rate=1e-4), loss=focal_loss(), metrics=['accuracy'])

# --- TRAINING ---
# We use a higher weight for 'Attentive' to prevent the constant predictor bug
class_weights = {0: 2.5, 1: 1.0}

print("🚀 Starting Training...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights
)

# --- SAVE ---
model.save("video_model_custom_final.h5")
print("✅ Saved as video_model_custom_final.h5")