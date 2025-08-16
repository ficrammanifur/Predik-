import cv2
import numpy as np
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
from datetime import datetime
import time
import logging
import platform
from collections import deque

# ======================
# Logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ======================
# Flask & CORS
# ======================
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://ficrammanifur.github.io",
            "http://localhost:3000",
            "http://localhost:5000",
            "*",
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "ngrok-skip-browser-warning"]
    }
})

# ======================
# Globals
# ======================
model = None
detection_history = []
current_detections = 0
last_detection_time = None
system_status = "Initializing"

# Default capture params (ubah kalau mau)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 15

# ======================
# Model Loader
# ======================
def load_model():
    """Load YOLO model (custom jika ada, kalau tidak pakai pretrained)."""
    global model, system_status
    try:
        # Cek beberapa lokasi umum hasil training
        candidate_paths = [
            'runs/detect/monyet/weights/best.pt',            # nama run yang disarankan
            'runs/detect/monkey_detection/weights/best.pt',  # nama run lama
        ]
        custom_path = next((p for p in candidate_paths if os.path.exists(p)), None)

        if custom_path:
            model = YOLO(custom_path)
            logging.info("Loaded custom trained model: %s", custom_path)
        else:
            model = YOLO('yolov8n.pt')
            logging.warning("Custom model not found. Fallback to pretrained yolov8n.pt")

        # Paksa ke CPU kalau tidak ada CUDA
        try:
            model.to('cuda')
        except Exception:
            model.to('cpu')
            logging.info("Using CPU for inference.")

        logging.info("Model classes: %s", getattr(model, "names", {}))
        system_status = "Active"
    except Exception as e:
        logging.error("Error loading model: %s", e)
        system_status = "Error"

load_model()

# ======================
# Camera utils
# ======================
def backend_for_os():
    """Pilih backend sesuai OS."""
    if platform.system().lower().startswith('win'):
        return cv2.CAP_DSHOW
    elif platform.system().lower().startswith('linux'):
        # V4L2 umumnya paling stabil di Linux
        return cv2.CAP_V4L2 if hasattr(cv2, 'CAP_V4L2') else 0
    else:
        return 0

def open_capture(index: int, width: int, height: int, fps: int):
    """Buka kamera index tertentu dengan setting resolusi & fps."""
    cap = cv2.VideoCapture(index, backend_for_os())
    if not cap.isOpened():
        cap.release()
        return None

    # Set resolusi & fps (tidak semua webcam menurut)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Tes baca 1 frame
    time.sleep(0.2)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def get_camera(max_attempts=3, max_indices=5, width=TARGET_WIDTH, height=TARGET_HEIGHT, fps=TARGET_FPS):
    """Coba beberapa index kamera dan retry beberapa kali."""
    for attempt in range(1, max_attempts + 1):
        for index in range(max_indices):
            try:
                cap = open_capture(index, width, height, fps)
                if cap:
                    logging.info("Webcam opened on index %d (%dx%d @ ~%dfps)", index, width, height, fps)
                    return cap
                else:
                    logging.warning("Failed to open webcam on index %d", index)
            except Exception as e:
                logging.error("Error opening webcam index %d: %s", index, e)
        logging.info("Attempt %d/%d failed, retrying in 2s...", attempt, max_attempts)
        time.sleep(2)
    logging.error("Failed to initialize webcam after all attempts")
    return None

# ======================
# Detection
# ======================
def detect_monkeys(frame):
    """Deteksi monyet dan gambar kotaknya."""
    global current_detections, last_detection_time

    if model is None:
        logging.error("Model not loaded, skipping detection")
        return frame, 0

    try:
        # conf bisa disesuaikan 0.25–0.6
        results = model(frame, conf=0.5, verbose=False)
        monkey_count = 0

        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names.get(cls_id, str(cls_id)).lower()
                # Hanya gambar untuk kelas 'monyet' (atau kalau pakai pretrained: monkey)
                if 'monyet' in class_name or 'monkey' in class_name:
                    monkey_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])

                    # Kotak + label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Monyet {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        current_detections = monkey_count
        if monkey_count > 0:
            last_detection_time = datetime.now()
            detection_entry = {
                'time': last_detection_time.strftime('%Y-%m-%d %H:%M:%S'),
                'count': monkey_count,
                'location': 'Webcam'
            }
            detection_history.append(detection_entry)
            # Batasi panjang history
            if len(detection_history) > 500:
                detection_history[:] = detection_history[-250:]

        return frame, monkey_count

    except Exception as e:
        logging.error("Detection error: %s", e)
        return frame, 0

# ======================
# Streaming generator
# ======================
def generate_frames():
    """Hasilkan stream MJPEG dengan anotasi YOLO."""
    camera = get_camera()
    if camera is None:
        # Kirim 1 dummy frame lalu stop
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy, 'No Webcam Available', (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ok, buf = cv2.imencode('.jpg', dummy, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return

    # FPS estimator
    times = deque(maxlen=30)
    try:
        while True:
            start_t = time.time()
            ok, frame = camera.read()
            if not ok:
                logging.warning("Failed to read frame, trying to reopen camera...")
                camera.release()
                camera = get_camera()
                if camera is None:
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(dummy, 'Webcam Disconnected', (110, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    ok2, buf2 = cv2.imencode('.jpg', dummy, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ok2:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buf2.tobytes() + b'\r\n')
                    return
                continue

            # Pastikan ukuran sesuai target (kalau device nggak nurut)
            if frame.shape[1] != TARGET_WIDTH or frame.shape[0] != TARGET_HEIGHT:
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            # Deteksi + anotasi
            frame, count = detect_monkeys(frame)

            # Hitung FPS
            times.append(time.time() - start_t)
            fps = (1.0 / (sum(times) / len(times))) if times else 0.0

            # Overlay info
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, f'Waktu: {ts}', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f'Monyet Terdeteksi: {count}', (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f'Res: {frame.shape[1]}x{frame.shape[0]}  FPS: {fps:.1f}', (10, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Encode ke JPEG
            ok_enc, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok_enc:
                logging.error("Failed to encode frame")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

            # Batasi rate (sekitar TARGET_FPS)
            # Jika ingin benar-benar mengunci FPS, gunakan sleep max(0, (1/TARGET_FPS - elapsed))
            time.sleep(0.001)

    except GeneratorExit:
        pass
    except Exception as e:
        logging.error("Frame generation error: %s", e)
    finally:
        if camera:
            camera.release()
            logging.info("Webcam released")

# ======================
# Routes
# ======================
@app.route('/')
def index():
    return jsonify({
        'message': 'Monkey Detection System Backend',
        'status': system_status,
        'endpoints': {
            'video_feed': '/video_feed',
            'api_status': '/api/status',
            'api_history': '/api/history',
            'api_clear_history': '/api/clear_history',
            'test_webcam': '/test_webcam'
        }
    })

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    today = datetime.now().strftime('%Y-%m-%d')
    return jsonify({
        'current_detections': current_detections,
        'last_detection': last_detection_time.strftime('%Y-%m-%d %H:%M:%S') if last_detection_time else 'Belum ada deteksi',
        'system_status': system_status,
        'total_detections_today': len([d for d in detection_history if d['time'].startswith(today)])
    })

@app.route('/api/history')
def api_history():
    return jsonify({
        'history': detection_history[-50:],
        'total_count': len(detection_history)
    })

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    global detection_history
    detection_history = []
    return jsonify({'message': 'Riwayat berhasil dihapus'})

@app.route('/test_webcam')
def test_webcam():
    cam = get_camera()
    if cam is None:
        return jsonify({'status': 'error', 'message': 'Gagal menginisialisasi webcam'}), 500
    try:
        ok, _ = cam.read()
        if ok:
            return jsonify({'status': 'success', 'message': 'Webcam berhasil diinisialisasi dan frame terbaca'})
        else:
            return jsonify({'status': 'error', 'message': 'Gagal membaca frame dari webcam'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error pengujian webcam: {str(e)}'}), 500
    finally:
        cam.release()

# ======================
# Main
# ======================
if __name__ == '__main__':
    logging.info("Starting Monkey Detection System...")
    logging.info("Access:  http://localhost:5000")
    logging.info("Video:   http://localhost:5000/video_feed")
    logging.info("Status:  http://localhost:5000/api/status")
    logging.info("History: http://localhost:5000/api/history")
    logging.info("Test:    http://localhost:5000/test_webcam")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
