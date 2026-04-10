# yolo_server.py
# Run in yolo_env: python yolo_server.py (port 5001)

from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import base64

app = Flask(__name__)

model = YOLO('yolov8n.pt')

# Warm up — forces model to fully load before first real request
_dummy = np.zeros((480, 640, 3), dtype=np.uint8)
model(_dummy, verbose=False)
print("✅ YOLO warmed up and ready")

TARGETS = {
    67: 'cell phone',
    39: 'bottle',
    41: 'cup',
}

@app.route('/detect', methods=['POST'])
def detect():
    detections = []  # always initialized first — prevents UnboundLocalError

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'phone': False, 'drinking': False, 'raw': []}), 400

    try:
        img_bytes = base64.b64decode(data['image'])
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("❌ Frame decode failed — bad base64?")
            return jsonify({'phone': False, 'drinking': False, 'raw': []}), 200

        results = model(frame, verbose=False, classes=list(TARGETS.keys()), conf=0.25)

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            detections.append({
                'class': TARGETS.get(cls_id, 'unknown'),
                'conf':  round(conf, 2),
            })

    except Exception as e:
        print(f"❌ YOLO error: {e}")
        return jsonify({'phone': False, 'drinking': False, 'raw': [], 'error': str(e)}), 500

    # Always prints — confirms requests are arriving
    print(f"🔍 Found: {detections}")

    return jsonify({
        'phone':    any(d['class'] == 'cell phone' for d in detections),
        'drinking': any(d['class'] in ['bottle', 'cup'] for d in detections),
        'raw':      detections
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)