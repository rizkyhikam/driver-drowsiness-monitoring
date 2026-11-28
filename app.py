
import cv2
import numpy as np
import pickle
import time
import urllib.request
import os
import threading

from flask import Flask, render_template, Response, jsonify, request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from supabase import create_client, Client

app = Flask(__name__)

# =====================================================
# 1. LOAD RANDOM FOREST MODEL
# =====================================================

with open('drowsiness_rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

print("[INFO] Random Forest model berhasil dimuat.")


# =====================================================
# 2. SETUP MEDIAPIPE FACE LANDMARKER
# =====================================================

MODEL_PATH = 'face_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print("[INFO] Mengunduh model Face Landmarker...")

    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/"
        "face_landmarker.task"
    )

    urllib.request.urlretrieve(url, MODEL_PATH)

    print("[INFO] Model Face Landmarker berhasil diunduh.")

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

print("[INFO] MediaPipe Face Landmarker berhasil diinisialisasi.")


# =====================================================
# 3. KONFIGURASI SUPABASE
# =====================================================


# =====================================================
# 3. KONFIGURASI SUPABASE
# =====================================================

SUPABASE_URL = "https://gwvdtoeuirnnttulnycv.supabase.co"
SUPABASE_KEY = "sb_publishable_MwrxZsL3dQlwTVeiQDdaRQ_1hwoJ4xq"


def _sanitize_proxy_env():
    # httpx gagal mem-parsing entri IPv6 "[::1]" di NO_PROXY Windows
    for var in ("NO_PROXY", "no_proxy"):
        if var in os.environ:
            os.environ[var] = ",".join(
                part for part in os.environ[var].split(",")
                if part.strip() not in ("[::1]", "::1")
            )


_sanitize_proxy_env()

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[INFO] Berhasil terkoneksi ke Supabase Database.")
except Exception as e:
    supabase = None
    print(f"[WARNING] Gagal koneksi ke Supabase: {e}")

last_supabase_error = 0.0

# =====================================================
# 4. GLOBAL MONITORING CONFIGURATION
# =====================================================

is_monitoring = False

monitoring_lock = threading.Lock()

telemetry_data = {
    "status": "STOPPED",
    "ear": 0.0,
    "mar": 0.0,
    "pitch": 0.0,
    "yaw": 0.0
}

last_log_time = time.time()


# =====================================================
# 5. GEOMETRY CALCULATION
# =====================================================

def calculate_ear(landmarks, eye_indices):

    p2_p6 = np.linalg.norm(
        landmarks[eye_indices[1]] -
        landmarks[eye_indices[5]]
    )

    p3_p5 = np.linalg.norm(
        landmarks[eye_indices[2]] -
        landmarks[eye_indices[4]]
    )

    p1_p4 = np.linalg.norm(
        landmarks[eye_indices[0]] -
        landmarks[eye_indices[3]]
    )

    if p1_p4 == 0:
        return 0.0

    return (p2_p6 + p3_p5) / (2.0 * p1_p4)


def calculate_mar(landmarks, mouth_indices):

    p2_p8 = np.linalg.norm(
        landmarks[mouth_indices[1]] -
        landmarks[mouth_indices[7]]
    )

    p3_p7 = np.linalg.norm(
        landmarks[mouth_indices[2]] -
        landmarks[mouth_indices[6]]
    )

    p4_p6 = np.linalg.norm(
        landmarks[mouth_indices[3]] -
        landmarks[mouth_indices[5]]
    )

    p1_p5 = np.linalg.norm(
        landmarks[mouth_indices[0]] -
        landmarks[mouth_indices[4]]
    )

    if p1_p5 == 0:
        return 0.0

    return (p2_p8 + p3_p7 + p4_p6) / (3.0 * p1_p5)


# =====================================================
# 6. HEAD POSE ESTIMATION
# =====================================================

def estimate_head_pose(transformation_matrix):

    # MediaPipe sudah memberi matriks transformasi kepala (canonical face
    # -> kamera). Kalau tidak ada, anggap kepala menghadap depan.
    if transformation_matrix is None:
        return 0.0, 0.0

    rmat = np.asarray(
        transformation_matrix,
        dtype=np.float64
    )[:3, :3]

    # Ubah matriks rotasi menjadi sudut Euler dalam derajat
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)

    pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))

    yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))

    return float(pitch), float(yaw)


# =====================================================
# 7. VIDEO STREAMING & PROCESSING PIPELINE
# =====================================================

def generate_frames():

    global telemetry_data
    global last_log_time
    global is_monitoring
    global last_supabase_error

    cap = None

    LEFT_EYE = [362, 385, 387, 263, 373, 380]

    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    MOUTH = [61, 81, 13, 311, 291, 402, 14, 178]

    while True:

        if is_monitoring:

            # -----------------------------------------
            # OPEN CAMERA
            # -----------------------------------------

            if cap is None or not cap.isOpened():

                # Backend MSMF default di Windows butuh ~20 detik untuk
                # membuka kamera; DirectShow jauh lebih cepat
                if os.name == 'nt':
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(0)

                if not cap.isOpened():

                    print("[ERROR] Kamera tidak dapat diakses.")

                    cap.release()
                    cap = None

                    time.sleep(1)
                    continue

                print("[INFO] Kamera berhasil diaktifkan.")

            success, frame = cap.read()

            if not success:

                print("[WARNING] Gagal membaca frame kamera.")

                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)

            h, w, c = frame.shape

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            detection_result = landmarker.detect(mp_image)

            # -----------------------------------------
            # FACE DETECTION
            # -----------------------------------------

            if detection_result.face_landmarks:

                face_landmarks = detection_result.face_landmarks[0]

                landmarks = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in face_landmarks
                ])

                # EAR
                left_ear = calculate_ear(
                    landmarks,
                    LEFT_EYE
                )

                right_ear = calculate_ear(
                    landmarks,
                    RIGHT_EYE
                )

                avg_ear = (
                    left_ear + right_ear
                ) / 2.0

                # MAR
                avg_mar = calculate_mar(
                    landmarks,
                    MOUTH
                )

                # HEAD POSE
                head_matrix = (
                    detection_result.facial_transformation_matrixes[0]
                    if detection_result.facial_transformation_matrixes
                    else None
                )

                pitch, yaw = estimate_head_pose(
                    head_matrix
                )

                # -------------------------------------
                # RANDOM FOREST PREDICTION
                # -------------------------------------

                features = np.array([[
                    avg_ear,
                    avg_mar,
                    pitch,
                    yaw
                ]])

                pred_status = rf_model.predict(
                    features
                )[0]

                # -------------------------------------
                # HYBRID RULE-BASED OVERRIDE
                # -------------------------------------

                if avg_ear <= 0.22 or avg_mar >= 0.35:

                    pred_status = 'DROWSY'

                elif abs(pitch) > 15.0 or abs(yaw) > 15.0:

                    pred_status = 'DISTRACTED'

                # -------------------------------------
                # UPDATE TELEMETRY
                # -------------------------------------

                telemetry_data = {
                    "status": str(pred_status),
                    "ear": float(avg_ear),
                    "mar": float(avg_mar),
                    "pitch": float(pitch),
                    "yaw": float(yaw)
                }

                # -------------------------------------
                # SUPABASE LOGGING EVERY 4 SECONDS
                # -------------------------------------

                if time.time() - last_log_time > 4.0:

                    if supabase:

                        try:

                            supabase.table(
                                "drowsiness_logs"
                            ).insert({

                                "status": str(pred_status),

                                "ear": round(
                                    float(avg_ear), 3
                                ),

                                "mar": round(
                                    float(avg_mar), 3
                                ),

                                "pitch": round(
                                    float(pitch), 2
                                ),

                                "yaw": round(
                                    float(yaw), 2
                                )

                            }).execute()

                            print(
                                f"[SUPABASE LOGGED] Status: {pred_status}"
                            )

                        except Exception as err:

                            if time.time() - last_supabase_error > 60.0:

                                print(
                                    f"[SUPABASE ERROR] {err}"
                                )

                                last_supabase_error = time.time()

                    last_log_time = time.time()

                # -------------------------------------
                # DRAW STATUS ON CAMERA
                # -------------------------------------

                if pred_status == 'ALERT':

                    color = (0, 255, 0)

                elif pred_status == 'DROWSY':

                    color = (0, 0, 255)

                else:

                    color = (0, 255, 255)

                cv2.putText(
                    frame,
                    f"STATUS: {pred_status}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"EAR: {avg_ear:.2f} | MAR: {avg_mar:.2f}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"PITCH: {pitch:.1f} | YAW: {yaw:.1f}",
                    (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            else:

                telemetry_data = {
                    "status": "NO FACE",
                    "ear": 0.0,
                    "mar": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0
                }

                cv2.putText(
                    frame,
                    "WAJAH TIDAK TERDETEKSI",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )

            # -----------------------------------------
            # ENCODE FRAME
            # -----------------------------------------

            ret, buffer = cv2.imencode(
                '.jpg',
                frame
            )

            if ret:

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    buffer.tobytes() +
                    b'\r\n'
                )

        else:

            # -----------------------------------------
            # STOP CAMERA
            # -----------------------------------------

            if cap is not None:

                if cap.isOpened():
                    cap.release()

                cap = None

                print("[INFO] Kamera dinonaktifkan.")

            telemetry_data = {
                "status": "STOPPED",
                "ear": 0.0,
                "mar": 0.0,
                "pitch": 0.0,
                "yaw": 0.0
            }

            blank_frame = np.zeros(
                (480, 640, 3),
                np.uint8
            )

            cv2.putText(
                blank_frame,
                "KAMERA NONAKTIF",
                (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (100, 116, 139),
                2
            )

            cv2.putText(
                blank_frame,
                "Klik Mulai Pemantauan untuk memulai",
                (110, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 150),
                1
            )

            ret, buffer = cv2.imencode(
                '.jpg',
                blank_frame
            )

            if ret:

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    buffer.tobytes() +
                    b'\r\n'
                )

            time.sleep(0.2)


# =====================================================
# 8. FLASK ROUTES
# =====================================================

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')


@app.route('/history')
def history():
    return render_template('history.html')


@app.route('/guide')
def guide():
    return render_template('guide.html')


@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/get_telemetry')
def get_telemetry():

    return jsonify(telemetry_data)


# =====================================================
# HISTORY API — ambil log dari Supabase
# =====================================================

@app.route('/get_history')
def get_history():
    """
    Query parameter yang didukung:
      ?page=1          → nomor halaman (default 1, 50 baris per halaman)
      ?status=DROWSY   → filter kolom status (opsional)
      ?date_from=YYYY-MM-DD  → filter tanggal mulai (opsional)
      ?date_to=YYYY-MM-DD    → filter tanggal akhir (opsional)
    """

    if supabase is None:
        return jsonify({
            "error": "Database tidak terhubung. Periksa konfigurasi Supabase.",
            "data": [],
            "total": 0
        }), 503

    try:

        PAGE_SIZE = 50

        page       = max(int(request.args.get('page', 1)), 1)
        status_filter  = request.args.get('status', '').strip().upper()
        date_from  = request.args.get('date_from', '').strip()
        date_to    = request.args.get('date_to', '').strip()

        offset = (page - 1) * PAGE_SIZE

        # --- hitung total (untuk pagination) ---
        count_query = supabase.table('drowsiness_logs').select(
            '*', count='exact'
        )

        if status_filter:
            count_query = count_query.eq('status', status_filter)

        if date_from:
            count_query = count_query.gte('created_at', date_from + 'T00:00:00')

        if date_to:
            count_query = count_query.lte('created_at', date_to + 'T23:59:59')

        count_result = count_query.execute()
        total = count_result.count if count_result.count is not None else 0

        # --- ambil data halaman ini ---
        data_query = supabase.table('drowsiness_logs').select(
            'id, created_at, status, ear, mar, pitch, yaw'
        ).order('created_at', desc=True).range(offset, offset + PAGE_SIZE - 1)

        if status_filter:
            data_query = data_query.eq('status', status_filter)

        if date_from:
            data_query = data_query.gte('created_at', date_from + 'T00:00:00')

        if date_to:
            data_query = data_query.lte('created_at', date_to + 'T23:59:59')

        result = data_query.execute()

        return jsonify({
            "data":      result.data,
            "total":     total,
            "page":      page,
            "page_size": PAGE_SIZE,
            "pages":     max(1, -(-total // PAGE_SIZE))   # ceil division
        })

    except Exception as err:
        print(f"[HISTORY ERROR] {err}")
        return jsonify({
            "error": str(err),
            "data": [],
            "total": 0
        }), 500


# =====================================================
# HISTORY STATS API — ringkasan per-status
# =====================================================

@app.route('/get_history_stats')
def get_history_stats():

    if supabase is None:
        return jsonify({"error": "Database tidak terhubung."}), 503

    try:

        result = supabase.table('drowsiness_logs').select(
            'status', count='exact'
        ).execute()

        # hitung per-status
        counts = {"ALERT": 0, "DROWSY": 0, "DISTRACTED": 0, "NO FACE": 0}
        total  = 0

        all_rows = supabase.table('drowsiness_logs').select(
            'status'
        ).execute()

        for row in (all_rows.data or []):
            s = row.get('status', '')
            counts[s] = counts.get(s, 0) + 1
            total += 1

        return jsonify({
            "total":      total,
            "alert":      counts.get('ALERT', 0),
            "drowsy":     counts.get('DROWSY', 0),
            "distracted": counts.get('DISTRACTED', 0),
            "no_face":    counts.get('NO FACE', 0),
        })

    except Exception as err:
        print(f"[STATS ERROR] {err}")
        return jsonify({"error": str(err)}), 500


# =====================================================
# 9. START MONITORING API
# =====================================================

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring_api():

    global is_monitoring
    global last_log_time

    with monitoring_lock:

        is_monitoring = True
        last_log_time = time.time()

    print("[INFO] Monitoring dimulai.")

    return jsonify({
        "status": "started"
    })


# =====================================================
# 10. STOP MONITORING API
# =====================================================

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring_api():

    global is_monitoring

    with monitoring_lock:

        is_monitoring = False

        telemetry_data.update({
            "status": "STOPPED",
            "ear": 0.0,
            "mar": 0.0,
            "pitch": 0.0,
            "yaw": 0.0
        })

    print("[INFO] Monitoring dihentikan.")

    return jsonify({
        "status": "stopped"
    })


# =====================================================
# 11. RUN FLASK SERVER
# =====================================================

if __name__ == '__main__':

    print("[INFO] Menjalankan Server Web Flask...")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False,
        threaded=True
    )