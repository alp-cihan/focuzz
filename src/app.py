# src/app_v2_ui.py
import os
import base64
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")

with open(logo_path, "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:15px; margin-bottom:5px;">
        <img src="data:image/jpeg;base64,{logo_base64}"
             width="65" height="65"
             style="border-radius:50%; box-shadow:0 0 15px #00b4d8;">
        <h1 style="margin:0; color:white;">Focuzz - TurkAI</h1>
    </div>
    <p style="color:#aaa; margin-top:-10px;">AI powered focus dashboard.</p>
    """,
    unsafe_allow_html=True
)


# 🌈 ---- CUSTOM UI THEME (Dashboard Tarzı Geliştirilmiş) ----
import datetime

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Helvetica:wght@700&display=swap');

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
    font-family: 'Helvetica', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

h1 {
    font-family: 'Helvetica', sans-serif;
    font-weight: 900;
    text-align: center;
    color: #EAF2F8;
    font-size: 42px;
    letter-spacing: 1px;
    margin-bottom: 0;
}

.subtext {
    text-align: center;
    font-family: 'Helvetica', sans-serif;
    font-size: 16px;
    color: #D6DBDF;
    margin-top: 4px;
    margin-bottom: 25px;
}

div[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px;
}

div[data-testid="stSidebar"] h2 {
    text-align: center;
    font-family: 'Helvetica', sans-serif;
    font-weight: 700;
    color: #00ADB5;
}

.sidebar-item {
    background-color: rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 10px 15px;
    margin-bottom: 10px;
}

.sidebar-item h4 {
    margin-bottom: 5px;
    color: #F8F9F9;
    font-family: 'Helvetica', sans-serif;
    font-weight: 700;
    font-size: 18px;
}

.sidebar-item p {
    color: #D0D3D4;
    font-size: 15px;
}

.stTextArea textarea {
    background-color: rgba(255,255,255,0.15);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px;
}

.cam-frame {
    border: 5px solid black;
    border-radius: 10px;
    overflow: hidden;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 🎯 Başlık ve alt metin
#st.markdown("<h1> Focuzz - TurkAI</h1>", unsafe_allow_html=True)
#st.markdown("<p class='subtext'>AI powered focus dashboard.</p>", unsafe_allow_html=True)

# 🧭 --- DASHBOARD SAĞ PANEL ---
with st.sidebar:
    st.markdown("## 🕓 Dashboard")

    # 🕒 Aktif saat (her dakika yenilenir)
    now = datetime.datetime.now().strftime("%H:%M")
    st.markdown(f"<div class='sidebar-item'><h4>🕒 Time</h4><p style='font-family:Helvetica;font-weight:700;font-size:22px;color:white;'>{now}</p></div>", unsafe_allow_html=True)

    # 🌤️ Hava Durumu (dummy)
    st.markdown("""
    <div class='sidebar-item'>
        <h4>🌤️ Weather Forecast</h4>
        <p>Adana, Türkiye</p>
        <p>☀️ 23°C | Sunny</p>
    </div>
    """, unsafe_allow_html=True)

    # 📝 Notlar Alanı
    st.markdown("""
    <div class='sidebar-item'>
        <h4>📝 Notes</h4>
        <p>Stay away from your phone when focusing! <br>Do not forget to give some break 😎</p>
    </div>
    """, unsafe_allow_html=True)

    # Kullanıcı not girişi
    st.text_area("Add your notes:", placeholder="To Do List...")

    # Dashboard çizgisi
    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; color:gray; text-align:center;'>Focuzz v1.0 © TurkAI</p>", unsafe_allow_html=True)


# 🧭 --- DASHBOARD SAĞ PANEL ---
# 🧭 --- DASHBOARD SAĞ PANEL (tek ve ortalanmış) ---

# 🎯 Setup
mp_face_mesh = mp.solutions.face_mesh
mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
yolo = YOLO("yolov8n.pt")

# 🎬 UI düzeni
col1, col2 = st.columns([2, 1])
FRAME_WINDOW = col1.image([])
col2.markdown("### 🔍 Live Data")
focus_text = col2.empty()
status_text = col2.empty()
phone_text = col2.empty()
record_status = col2.empty()
summary_box = st.empty()

record_btn = st.button("🎬 Start Recording / Stop")

# 🧠 State
if "recording" not in st.session_state:
    st.session_state.recording = False
    st.session_state.start_time = None
    st.session_state.frames = []
    st.session_state.phone_frames = 0
    st.session_state.phone_start = None

cap = cv2.VideoCapture(0)

def eye_direction(face):
    left_eye_indices = [33, 133, 159, 145]
    right_eye_indices = [263, 362, 386, 374]
    lx = np.mean([face.landmark[i].x for i in left_eye_indices])
    rx = np.mean([face.landmark[i].x for i in right_eye_indices])
    return (lx + rx) / 2

# 🎬 Kayıt başlat / durdur
if record_btn:
    if not st.session_state.recording:
        st.session_state.recording = True
        st.session_state.start_time = time.time()
        st.session_state.frames = []
        st.session_state.phone_frames = 0
        st.session_state.phone_start = None
        summary_box.empty()
        st.success("🔴 Recording successful!")
    else:
            st.session_state.recording = False
            duration = time.time() - st.session_state.start_time
            total = len(st.session_state.frames)
            if total > 0:
                arr = np.array(st.session_state.frames)
                f = (arr == "FOCUSED").sum() / total * 100
                p = (arr == "PARTIAL").sum() / total * 100
                d = (arr == "DISTRACTED").sum() / total * 100
                phone_time = st.session_state.phone_frames / 20  # 20fps tahmini

                # 🎭 Emoji’li feedback
                if f >= 75:
                    feedback = "😎 Perfect! Legendary focus."
                elif f >= 50:
                    feedback = "🙂 Not bad, try to focus more."
                else:
                    feedback = "😴 Low focus ratio, should we give a break??"

                summary_box.success(
                    f"📊 **Record completed!**\n\n"
                    f"🕒 Süre: {int(duration // 60)} dk {int(duration % 60)} sn\n"
                    f"🧠 Focus: **%{f:.1f} focused**\n"
                    f"😐 Partial: **%{p:.1f}**\n"
                    f"😵 Distracted: **%{d:.1f}**\n"
                    f"📱 Phone on screen: **{phone_time:.1f} seconds!**\n\n"
                    f"{feedback}"
                )


            else:
                summary_box.warning("Empty recording! 😅")




# 🎥 Kamera döngüsü
while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Could not start camera. 😕")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)

    label = "No face"
    score = 0
    color = (255, 255, 255)
    phone_detected = False

    # 📱 YOLO tespiti
    results = yolo.predict(rgb, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if r.names[cls].lower() in ["cell phone", "mobile phone", "phone"]:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "PHONE!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # ⏱️ Telefon süresi ölçümü
    if st.session_state.recording:
        if phone_detected:
            st.session_state.phone_frames += 1

    # 👤 Focus analizi
    if res.multi_face_landmarks:
        f = res.multi_face_landmarks[0]
        left_top = f.landmark[159].y
        left_bottom = f.landmark[145].y
        right_top = f.landmark[386].y
        right_bottom = f.landmark[374].y
        eye_ratio = ((left_bottom - left_top) + (right_bottom - right_top)) / 2
        eyes_open = max(0.0, min(1.0, eye_ratio * 50))

        left_eye = f.landmark[33]
        right_eye = f.landmark[263]
        nose = f.landmark[1]
        mid_eye_y = (left_eye.y + right_eye.y) / 2

        # 🎯 Adaptif Yaw Normalizasyonu
        eye_distance = abs(left_eye.x - right_eye.x)

        if "baseline_eye_dist" not in st.session_state:
            st.session_state.baseline_eye_dist = eye_distance  # ilk frame referans
        else:
            # yavaş yavaş güncelle (smooth)
            st.session_state.baseline_eye_dist = (
                    st.session_state.baseline_eye_dist * 0.98 + eye_distance * 0.02
            )

        # yaw değeri artık fark oranına göre normalize
        yaw = abs(1 - (eye_distance / st.session_state.baseline_eye_dist)) * 50

        pitch = abs(nose.y - mid_eye_y) * 100

        eye_center = eye_direction(f)
        gaze_penalty = abs(0.5 - eye_center) * 200

        base_score = max(0, 100 - (yaw * 1.2) - (pitch * 0.9) - gaze_penalty) * eyes_open
        score = 0 if phone_detected else base_score

        if score > 55:
            label = "FOCUSED"
            color = (0, 255, 0)
        elif score > 35:
            label = "PARTIAL"
            color = (0, 255, 255)
        else:
            label = "DISTRACTED"
            color = (255, 0, 0)

        if st.session_state.recording:
            st.session_state.frames.append(label)

        cv2.putText(frame, f"{label} ({int(score)}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # 🧠 Görseller ve bilgiler
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    focus_text.markdown(f"**Status:** {label}")
    status_text.markdown(f"**Focus Score:** {int(score)}%")
    phone_text.markdown(f"**Phone Detected:** {'📱 YES' if phone_detected else '❌ NO'}")
    record_status.markdown(
        f"**Recording State:** {'🔴 Recording' if st.session_state.recording else '⚪ Awaiting'}"
    )

    time.sleep(0.05)

cap.release()
