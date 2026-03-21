# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import os
import tempfile
import base64

app = Flask(__name__)
CORS(app)

# ================================
# LOAD MODEL
# ================================
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * ce
    return loss

MODEL_PATH = os.path.join(os.path.dirname(__file__), "video_model_86_best.h5")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'loss': focal_loss()},
    compile=False
)
print("✅ Model loaded!")

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

last_conv_layer = get_last_conv_layer(model)
print(f"✅ Last conv layer: {last_conv_layer}")

# ================================
# GRAD-CAM
# ================================
def generate_gradcam(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def apply_gradcam_overlay(frame, heatmap):
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
    return overlay

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def get_risk_level(distracted_pct):
    if distracted_pct < 20:
        return "LOW"
    elif distracted_pct < 50:
        return "MEDIUM"
    else:
        return "HIGH"

# ================================
# IMAGE ANALYSIS
# ================================
@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    img_array = np.frombuffer(image_file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Cannot read image'}), 400

    img_input = preprocess(frame)
    prob = float(model.predict(img_input, verbose=0)[0][0])
    label = "Distracted" if prob > 0.5 else "Attentive"
    risk = get_risk_level(prob * 100)

    # Grad-CAM
    gradcam_data = None
    try:
        heatmap = generate_gradcam(img_input, model, last_conv_layer)
        overlay = apply_gradcam_overlay(frame, heatmap)
        gradcam_data = {
            'original': frame_to_base64(frame),
            'gradcam': frame_to_base64(overlay),
            'confidence': round(prob * 100, 1)
        }
    except Exception as e:
        print(f"Grad-CAM error: {e}")

    return jsonify({
        'label': label,
        'confidence': round(prob * 100, 1),
        'risk_level': risk,
        'gradcam': gradcam_data
    })

# ================================
# VIDEO ANALYSIS
# ================================
@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video_file = request.files['video']
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(tmp.name)
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open video'}), 400

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    confidences = []
    labels = []
    timestamps = []
    frame_count = 0
    sample_every = max(1, int(fps / 2))

    max_prob = 0
    most_distracted_frame = None

    # For processed video frames
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_every == 0:
            img_input = preprocess(frame)
            prob = float(model.predict(img_input, verbose=0)[0][0])
            label = "Distracted" if prob > 0.5 else "Attentive"
            confidences.append(round(prob, 4))
            labels.append(label)
            timestamps.append(round(frame_count / fps, 2) if fps > 0 else frame_count)

            # Draw overlay on frame
            color = (0, 0, 255) if prob > 0.5 else (0, 255, 0)
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (10, 10), (400, 60), (0, 0, 0), -1)
            cv2.putText(display_frame, f"{label} ({prob*100:.1f}%)",
                       (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Save frame as base64
            if len(processed_frames) < 50:  # limit frames sent to frontend
                processed_frames.append(frame_to_base64(display_frame))

            if prob > max_prob:
                max_prob = prob
                most_distracted_frame = frame.copy()

        frame_count += 1

    cap.release()
    os.unlink(tmp.name)

    # Summary
    total = len(labels)
    distracted_count = labels.count("Distracted")
    attentive_count = labels.count("Attentive")
    distracted_pct = round((distracted_count / total) * 100, 1) if total > 0 else 0
    attentive_pct = round((attentive_count / total) * 100, 1) if total > 0 else 0
    risk_level = get_risk_level(distracted_pct)

    # Alerts
    alerts = []
    in_alert = False
    alert_start = 0
    ALERT_THRESHOLD = 5
    consecutive = 0

    for i, label in enumerate(labels):
        if label == "Distracted":
            consecutive += 1
            if consecutive >= ALERT_THRESHOLD and not in_alert:
                in_alert = True
                alert_start = timestamps[i - ALERT_THRESHOLD + 1]
        else:
            if in_alert:
                alerts.append({'start': alert_start, 'end': timestamps[i - 1]})
            in_alert = False
            consecutive = 0

    if in_alert:
        alerts.append({'start': alert_start, 'end': timestamps[-1]})

    # Grad-CAM
    gradcam_data = None
    if most_distracted_frame is not None:
        try:
            img_input = preprocess(most_distracted_frame)
            heatmap = generate_gradcam(img_input, model, last_conv_layer)
            gradcam_overlay = apply_gradcam_overlay(most_distracted_frame, heatmap)
            gradcam_data = {
                'original': frame_to_base64(most_distracted_frame),
                'gradcam': frame_to_base64(gradcam_overlay),
                'confidence': round(max_prob * 100, 1)
            }
        except Exception as e:
            print(f"Grad-CAM error: {e}")

    return jsonify({
        'summary': {
            'total_frames_analyzed': total,
            'duration_seconds': round(duration, 2),
            'attentive_pct': attentive_pct,
            'distracted_pct': distracted_pct,
            'alert_count': len(alerts),
            'overall_status': 'DISTRACTED' if distracted_pct > 50 else 'ATTENTIVE',
            'risk_level': risk_level
        },
        'frame_data': {
            'timestamps': timestamps,
            'confidences': confidences,
            'labels': labels
        },
        'processed_frames': processed_frames,
        'alerts': alerts,
        'gradcam': gradcam_data
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)