import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def detect_face_and_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detections = []

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_gray)

        detections.append({
            "face_box": (x, y, w, h),
            "eyes": eyes,
            "face_roi": face_roi
        })

    return detections
