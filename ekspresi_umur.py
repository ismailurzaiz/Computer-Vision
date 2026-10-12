import cv2
import threading
from collections import deque
from ultralytics import YOLO
from deepface import DeepFace

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

label_terakhir = {}
riwayat_umur = {}
sedang_analisis = {}

def analisis_wajah(potongan, track_id):
    try:
        hasil = DeepFace.analyze(
            potongan,
            actions=["age", "emotion"],
            enforce_detection=False,
            detector_backend="mtcnn"
        )
        umur_baru = hasil[0]["age"]
        ekspresi = hasil[0]["dominant_emotion"]

        if track_id not in riwayat_umur:
            riwayat_umur[track_id] = deque(maxlen=5)
        riwayat_umur[track_id].append(umur_baru)

        umur_rata2 = round(sum(riwayat_umur[track_id]) / len(riwayat_umur[track_id]))
        label_terakhir[track_id] = f"{ekspresi}, {umur_rata2} th"
    except Exception:
        pass
    sedang_analisis[track_id] = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera tidak terbaca")
        break

    hasil = model.track(frame, persist=True, verbose=False, classes=[0])

    if hasil[0].boxes.id is not None:
        for box, track_id in zip(hasil[0].boxes, hasil[0].boxes.id):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            potongan = frame[y1:y2, x1:x2].copy()

            if potongan.size != 0 and not sedang_analisis.get(track_id, False):
                sedang_analisis[track_id] = True
                threading.Thread(target=analisis_wajah, args=(potongan, track_id), daemon=True).start()

            label = label_terakhir.get(track_id, "Menganalisis...")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Deteksi Ekspresi & Umur", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()