import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Kamera tidak terbaca")
else:
    cv2.imwrite("hasil_capture.jpg", frame)
    try:
        hasil = DeepFace.analyze(
            frame,
            actions=["age", "emotion"],
            enforce_detection=False,
            detector_backend="opencv"
        )
        print(hasil)
    except Exception as e:
        print("ERROR:", e)