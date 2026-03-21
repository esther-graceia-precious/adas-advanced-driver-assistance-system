# multistream/neural_style_augment.py
import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm

# ================================
# NEURAL STYLE TRANSFER AUGMENTATION
# ================================

def gram_matrix(tensor):
    """Compute Gram matrix for style representation"""
    shape = tf.shape(tensor)
    channels = shape[-1]
    flattened = tf.reshape(tensor, [-1, channels])
    gram = tf.matmul(flattened, flattened, transpose_a=True)
    return gram / tf.cast(tf.size(flattened), tf.float32)

def get_style_model():
    """Use VGG19 to extract style features"""
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        weights='imagenet'
    )
    vgg.trainable = False

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    outputs = [vgg.get_layer(name).output for name in style_layers]
    return tf.keras.Model([vgg.input], outputs)

def get_style_features(style_model, img):
    """Extract style features from image"""
    img_tensor = tf.expand_dims(img, 0)
    img_tensor = tf.keras.applications.vgg19.preprocess_input(img_tensor * 255.0)
    outputs = style_model(img_tensor)
    return [gram_matrix(o[0]) for o in outputs]

def apply_style_transfer(content_img, style_features, style_model,
                          steps=100, style_weight=1e-2):
    """Apply style transfer to content image"""
    content_img = tf.cast(content_img, tf.float32)
    generated = tf.Variable(content_img)

    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            img_tensor = tf.expand_dims(generated, 0)
            img_tensor = tf.keras.applications.vgg19.preprocess_input(
                img_tensor * 255.0
            )
            gen_features = style_model(img_tensor)

            style_loss = 0
            for gen_feat, style_feat in zip(gen_features, style_features):
                gen_gram = gram_matrix(gen_feat[0])
                style_loss += tf.reduce_mean((gen_gram - style_feat) ** 2)

            style_loss *= style_weight

        grads = tape.gradient(style_loss, generated)
        optimizer.apply_gradients([(grads, generated)])
        generated.assign(tf.clip_by_value(generated, 0.0, 1.0))

    for _ in range(steps):
        train_step()

    return generated.numpy()

# ================================
# FAST STYLE AUGMENTATION
# (Lighter version using color/texture statistics)
# ================================
def fast_style_augment(content_img, style_img):
    """
    Fast style augmentation using AdaIN
    (Adaptive Instance Normalization)
    Transfers color and texture statistics without heavy computation
    """
    # Convert to float
    content = content_img.astype(np.float32) / 255.0
    style   = style_img.astype(np.float32) / 255.0

    # Match mean and std of each channel
    result = np.zeros_like(content)
    for c in range(3):
        c_mean = np.mean(content[:,:,c])
        c_std  = np.std(content[:,:,c]) + 1e-5
        s_mean = np.mean(style[:,:,c])
        s_std  = np.std(style[:,:,c]) + 1e-5

        # Normalize content then scale to style statistics
        normalized = (content[:,:,c] - c_mean) / c_std
        result[:,:,c] = normalized * s_std + s_mean

    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)

# ================================
# AUGMENT DATASET
# ================================
def augment_dataset_with_style(
    dataset_path,
    style_images_path,
    output_path,
    augment_ratio=0.3
):
    """
    Augment dataset using style transfer
    
    dataset_path:      your new_dataset folder
    style_images_path: folder with video frames (different style)
    output_path:       where to save augmented dataset
    augment_ratio:     what fraction of images to augment (0.3 = 30%)
    """
    import random

    # Load style images (video frames)
    style_images = []
    for img_name in os.listdir(style_images_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(style_images_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                style_images.append(img)

    if not style_images:
        print("❌ No style images found!")
        return

    print(f"✅ Loaded {len(style_images)} style images")

    # Process each class
    for class_name in ['attentive', 'distracted']:
        src_folder = os.path.join(dataset_path, class_name)
        dst_folder = os.path.join(output_path, class_name)
        os.makedirs(dst_folder, exist_ok=True)

        images = [f for f in os.listdir(src_folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\n📁 Processing {class_name}: {len(images)} images")

        for img_name in tqdm(images):
            src_path = os.path.join(src_folder, img_name)
            content_img = cv2.imread(src_path)

            if content_img is None:
                continue

            content_img = cv2.resize(content_img, (224, 224))

            # Copy original
            dst_path = os.path.join(dst_folder, img_name)
            cv2.imwrite(dst_path, content_img)

            # Augment with style transfer
            if random.random() < augment_ratio:
                style_img = random.choice(style_images)
                augmented = fast_style_augment(content_img, style_img)
                aug_name  = f"aug_{img_name}"
                aug_path  = os.path.join(dst_folder, aug_name)
                cv2.imwrite(aug_path, augmented)

    print(f"\n✅ Augmented dataset saved to: {output_path}")
    print(f"Original + augmented images ready for training!")


if __name__ == "__main__":
    # Style images = video frames (the target domain)
    # Extract some normal driving frames to use as style
    STYLE_FRAMES = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\new_dataset\attentive"
    DATASET      = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\dataset"
    OUTPUT       = r"C:\Users\A Esther Graceia\Documents\ADAS_PROJECT\augmented_dataset"

    augment_dataset_with_style(
        dataset_path=DATASET,
        style_images_path=STYLE_FRAMES,
        output_path=OUTPUT,
        augment_ratio=0.3  # augment 30% of images
    )