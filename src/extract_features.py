# src/extract_features.py (Hızlı optimize edilmiş sürüm 🚀)
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path

mp_face_mesh = mp.solutions.face_mesh
mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def eye_open_ratio(face):
    """Göz açıklığı tahmini (üst-alt farkı)"""
    left_top = face.landmark[159].y
    left_bottom = face.landmark[145].y
    right_top = face.landmark[386].y
    right_bottom = face.landmark[374].y
    ratio = ((left_bottom - left_top) + (right_bottom - right_top)) / 2
    return max(0.0, min(1.0, ratio * 50))

def yaw_pitch(face):
    """Kafa yön tahmini (yaw/pitch açıları)"""
    nose = face.landmark[1]
    left_eye = face.landmark[33]
    right_eye = face.landmark[263]
    yaw = (left_eye.x - right_eye.x) * 100
    pitch = (nose.y - (left_eye.y + right_eye.y) / 2) * 100
    return yaw, pitch

def extract_one(video_path: Path, sample_fps=1):
    """Belirtilen videodan frame bazlı özellik çıkarımı"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int((fps // sample_fps) * 2))  # Hız artışı: 2 saniyede 1 frame al
    rows, idx, processed = [], 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            if res.multi_face_landmarks:
                f = res.multi_face_landmarks[0]
                eyes_open = eye_open_ratio(f)
                yaw, pitch = yaw_pitch(f)
                rows.append(dict(frame=idx, yaw=yaw, pitch=pitch, eyes_open=eyes_open, face_visible=1))
            else:
                rows.append(dict(frame=idx, yaw=0, pitch=0, eyes_open=0, face_visible=0))
            processed += 1

            # Her 100 frame'de bir ilerleme bildir
            if processed % 100 == 0:
                print(f"⏳ {vid.name} → {processed} frame işlendi...")

        idx += 1

    cap.release()
    return pd.DataFrame(rows)

if __name__ == "__main__":
    data_dir = Path("data/raw_videos/models/src")
    dfs = []

    for vid in sorted(data_dir.glob("*.MOV")):
        print(f"🎥 İşleniyor: {vid.name}")
        df = extract_one(vid, sample_fps=1)  # saniyede 1 örnek al, hızlı mod
        df["video"] = vid.name
        dfs.append(df)

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        all_df.to_csv("data/features_raw.csv", index=False)
        print("✅ features_raw.csv yeniden oluşturuldu (hızlı mod kuzen 🔥)")
    else:
        print("⚠️ Hiç video bulunamadı, yolunu kontrol et kuzen.")
