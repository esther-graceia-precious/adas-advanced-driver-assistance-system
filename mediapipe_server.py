from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import mediapipe as mp

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh

def calculate_ear(landmarks, w, h):
    # Left eye
    left_eye = [33, 160, 158, 133, 153, 144]
    # Right eye  
    right_eye = [362, 385, 387, 263, 373, 380]
    
    def eye_ear(indices):
        pts = [(int(landmarks.landmark[i].x * w),
                int(landmarks.landmark[i].y * h)) for i in indices]
        p1, p2, p3, p4, p5, p6 = pts
        return (np.linalg.norm(np.array(p2)-np.array(p6)) +
                np.linalg.norm(np.array(p3)-np.array(p5))) / \
               (2 * np.linalg.norm(np.array(p1)-np.array(p4)))
    
    return (eye_ear(left_eye) + eye_ear(right_eye)) / 2.0

def calculate_mar(landmarks, w, h):
    # Top, bottom (vertical), left, right (horizontal)
    mouth_pts = [13, 14, 17, 0, 78, 308]
    pts = [(int(landmarks.landmark[i].x * w),
            int(landmarks.landmark[i].y * h)) for i in mouth_pts]
    
    top, bottom, chin, nose, left, right = pts
    vertical = np.linalg.norm(np.array(top) - np.array(bottom))
    horizontal = np.linalg.norm(np.array(left) - np.array(right))
    return vertical / (horizontal + 1e-6)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()

        # Decode image
        img_bytes = base64.b64decode(data['image'])
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image data', 'ear': None, 'mar': None}), 400

        # Create a NEW FaceMesh instance for each request
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,  # Changed to True for stateless processing
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        ) as face_mesh:
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                return jsonify({'ear': None, 'mar': None})

            landmarks = result.multi_face_landmarks[0]
            h, w, _ = frame.shape

            ear = calculate_ear(landmarks, w, h)
            mar = calculate_mar(landmarks, w, h)

            return jsonify({'ear': float(ear), 'mar': float(mar)})

    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        return jsonify({'error': str(e), 'ear': None, 'mar': None}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(port=5002, debug=False)