import cv2
import mediapipe as mp
import joblib
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

def get_face_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    return points.flatten()

print("[INFO] Загрузка модели...")
clf = joblib.load('Lesson 19/Homework/face_classifier.pkl')
label_to_name = joblib.load('Lesson 19/Homework/label_to_name.pkl')

cap = cv2.VideoCapture(0)

print("[INFO] Запуск камеры... Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = get_face_landmarks(frame)
    if landmarks is not None:
        prediction = clf.predict([landmarks])
        name = label_to_name[prediction[0]]
        proba = clf.predict_proba([landmarks]).max()
        text = f"{name} ({proba*100:.1f}%)"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()