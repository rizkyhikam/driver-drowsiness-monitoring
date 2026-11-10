import cv2
import numpy as np
import pickle
import time
import urllib.request
import os
from flask import Flask, render_template, Response, jsonify
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from supabase import create_client, Client

app = Flask(__name__)

# 1. LOAD MODEL RANDOM FOREST
with open('drowsiness_rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# 2. AUTO-DOWNLOAD & SETUP MEDIAPIPE FACE LANDMARKER (API TERBARU FOR PYTHON 3.13)
MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("[INFO] Mengunduh file model Face Landmarker MediaPipe...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# 3. KONFIGURASI SUPABASE
SUPABASE_URL = "https://gwvdtoeuirnnttulnycv.supabase.co"
SUPABASE_KEY = "sb_publishable_MwrxZsL3dQlwTVeiQDdaRQ_1hwoJ4xq"

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[INFO] Berhasil terkoneksi ke Supabase Database.")
except Exception as e:
    supabase = None
    print(f"[WARNING] Gagal koneksi ke Supabase: {e}")

telemetry_data = {
    "status": "ALERT",
    "ear": 0.30,
    "mar": 0.15,
    "pitch": 0.0,
    "yaw": 0.0
}
last_log_time = time.time()

# 4. RUMUS KALKULASI GEOMETRI (EAR, MAR, HEAD POSE)
def calculate_ear(landmarks, eye_indices):
    p2_p6 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    p3_p5 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    p1_p4 = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def calculate_mar(landmarks, mouth_indices):
    p2_p8 = np.linalg.norm(landmarks[mouth_indices[1]] - landmarks[mouth_indices[7]])
    p3_p7 = np.linalg.norm(landmarks[mouth_indices[2]] - landmarks[mouth_indices[6]])
    p4_p6 = np.linalg.norm(landmarks[mouth_indices[3]] - landmarks[mouth_indices[5]])
    p1_p5 = np.linalg.norm(landmarks[mouth_indices[0]] - landmarks[mouth_indices[4]])
    return (p2_p8 + p3_p7 + p4_p6) / (3.0 * p1_p5)

def estimate_head_pose(landmarks, image_shape):
    h, w, _ = image_shape
    face_3d, face_2d = [], []
    indices = [1, 152, 263, 33, 287, 57]
    
    for idx in indices:
        x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
        face_2d.append([x, y])
        face_3d.append([x, y, landmarks[idx][2]])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * w
    cam_matrix = np.array([[focal_length, 0, h / 2], [0, focal_length, w / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles[0] * 360, angles[1] * 360

# 5. STREAMING WEBCAM & PIPELINE PROSES
def generate_frames():
    global telemetry_data, last_log_time
    cap = cv2.VideoCapture(0)

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [61, 81, 13, 311, 291, 402, 14, 178]

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])
            
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            avg_mar = calculate_mar(landmarks, MOUTH)
            pitch, yaw = estimate_head_pose(landmarks, frame.shape)

            # 1. Prediksi Awal via Model ML Random Forest
            features = np.array([[avg_ear, avg_mar, pitch, yaw]])
            pred_status = rf_model.predict(features)[0]

            # 2. Hybrid Override Rule (Sangat Responsif Real-Time)
            if avg_ear <= 0.22 or avg_mar >= 0.35:
                pred_status = 'DROWSY'
            elif abs(pitch) > 15.0 or abs(yaw) > 15.0:
                pred_status = 'DISTRACTED'

            telemetry_data = {
                "status": pred_status,
                "ear": float(avg_ear),
                "mar": float(avg_mar),
                "pitch": float(pitch),
                "yaw": float(yaw)
            }

            # Auto Push Log ke Supabase setiap 4 detik
            if time.time() - last_log_time > 4.0:
                if supabase:
                    try:
                        supabase.table("drowsiness_logs").insert({
                            "status": pred_status,
                            "ear": round(float(avg_ear), 3),
                            "mar": round(float(avg_mar), 3),
                            "pitch": round(float(pitch), 2),
                            "yaw": round(float(yaw), 2)
                        }).execute()
                        print(f"[SUPABASE LOGGED] Status: {pred_status}")
                    except Exception as err:
                        print(f"[SUPABASE ERROR] {err}")
                last_log_time = time.time()

            color = (0, 255, 0) if pred_status == 'ALERT' else ((0, 0, 255) if pred_status == 'DROWSY' else (0, 255, 255))
            cv2.putText(frame, f"STATUS: {pred_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f} | MAR: {avg_mar:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_telemetry')
def get_telemetry():
    return jsonify(telemetry_data)

if __name__ == '__main__':
    print("[INFO] Menjalankan Server Web Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)