# yolo_server.py — UPDATED VERSION
from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import base64

app = Flask(__name__)

model = YOLO('yolov8n.pt')

# Warm up
_dummy = np.zeros((480, 640, 3), dtype=np.uint8)
model(_dummy, verbose=False)
print("✅ YOLO warmed up and ready")

# VERIFIED COCO class IDs
# VERIFIED COCO class IDs (Expanded for robustness)
TARGETS = {
    67: 'cell phone',
    39: 'bottle',
    41: 'cup',      # Very common misidentification for bottles
    52: 'hot dog'  # Your logs showed this is a frequent confusion
}

@app.route('/detect', methods=['POST'])
def detect():
    detections = []

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'phone': False, 'drinking': False, 'raw': []}), 400

    try:
        img_bytes = base64.b64decode(data['image'])
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("❌ Frame decode failed")
            return jsonify({'phone': False, 'drinking': False, 'raw': []}), 200

        # CRITICAL FIX: Remove class filter to see ALL detections first
        results = model(frame, verbose=False, conf=0.30)  # Lowered confidence

        # Log ALL detections (debugging)
        all_detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = results[0].names[cls_id]  # Get actual class name
            all_detections.append({
                'id': cls_id,
                'class': cls_name,
                'conf': round(conf, 2),
            })

        print(f"🔍 ALL detections: {all_detections}")

        # Filter only target objects
        for det in all_detections:
            if det['id'] in TARGETS:
                detections.append({
                    'class': TARGETS[det['id']],
                    'conf': det['conf']
                })

    except Exception as e:
        print(f"❌ YOLO error: {e}")
        return jsonify({'phone': False, 'drinking': False, 'raw': [], 'error': str(e)}), 500

    print(f"✅ Filtered detections: {detections}")

    return jsonify({
        'phone': any(d['class'] == 'cell phone' for d in detections),
        'drinking': any(d['class'] in ['bottle', 'cup'] for d in detections),
        'raw': detections
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)