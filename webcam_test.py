import cv2
from ultralytics import YOLO

# Load model YOLO (bisa pakai custom 'best.pt' kalau sudah training)
model = YOLO("yolov8n.pt")  

# Buka webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows pakai CAP_DSHOW, Linux bisa tanpa

# 🔹 Atur resolusi webcam (ubah sesuai kebutuhan)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # contoh: 1280, bisa 640 atau 1920
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # contoh: 720, bisa 480 atau 1080

# 🔹 Supaya window bisa dibesarkan manual
cv2.namedWindow("YOLO Webcam Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Webcam Live", 1280, 720)

if not cap.isOpened():
    print("❌ Webcam gagal dibuka")
else:
    print("✅ Webcam berhasil dibuka")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame")
            break

        # Deteksi objek dengan YOLO
        results = model(frame, stream=True)

        # Loop setiap hasil deteksi
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Ambil koordinat kotak
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Ambil confidence dan kelas
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                # Gambar kotak
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Teks label + confidence
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

        # Tampilkan hasil
        cv2.imshow("YOLO Webcam Live", frame)

        # Tekan Q untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
