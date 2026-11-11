# src/auto_label.py
import pandas as pd
from pathlib import Path

# Çıkış klasörünü oluştur (varsa hata vermez)
Path("data").mkdir(parents=True, exist_ok=True)

# Veri dosyasını oku
df = pd.read_csv("data/features_raw.csv")

labels = []
window = 60  # 2 saniyelik bloklar (30 fps varsayarsak)
for video, g in df.groupby("video"):
    for i in range(0, len(g), window):
        chunk = g.iloc[i:i+window]
        if len(chunk) < window * 0.5:
            continue

        yaw_ok = (chunk["yaw"].abs() < 12).mean()          # çok dönükse düş
        pitch_ok = (chunk["pitch"].abs() < 10).mean()      # eğilmişse düş
        eyes_open_ok = (chunk["eyes_open"] > 0.45).mean()  # düşükse uykulu say
        face_ok = (chunk["face_visible"] > 0.8).mean()     # yüz kaybolmuşsa distracted

        # Ortalama odak puanı (0-1 arası)
        focus_score = (yaw_ok + pitch_ok + eyes_open_ok + face_ok) / 4

        if focus_score > 0.75:
            label = "focused"
        elif focus_score > 0.55:
            label = "partially_focused"
        else:
            label = "distracted"

        labels.append({
            "video": chunk["video"].iloc[0],
            "start": chunk["frame"].iloc[0],
            "end": chunk["frame"].iloc[-1],
            "label": label
        })

# Çıktıyı kaydet
out_path = Path("data/labels.csv")
pd.DataFrame(labels).to_csv(out_path, index=False)

print("✅ labels.csv oluşturuldu (orta sertlikte odak eşiğiyle, kuzen 🔥)")
