# src/join_labels.py
import pandas as pd

# Dosyaların yolları
feat_path = "data/features_raw.csv"
labels_path = "data/labels.csv"
out_path = "data/features.csv"

# 1) Ham özellikler (frame bazlı)
df_f = pd.read_csv(feat_path)   # frame, yaw, pitch, eyes_open, face_visible, video
df_l = pd.read_csv(labels_path) # video, start, end, label

# Varsayılan label
df_f["label"] = "unknown"

# Her label aralığını ilgili framelere yaz
for row in df_l.itertuples():
    mask = (
        (df_f["video"] == row.video) &
        (df_f["frame"] >= row.start) &
        (df_f["frame"] <= row.end)
    )
    df_f.loc[mask, "label"] = row.label

# Sadece labeled olanları al (unknown olanları at)
df_f = df_f[df_f["label"] != "unknown"].reset_index(drop=True)

df_f.to_csv(out_path, index=False)
print("✅ features.csv oluşturuldu kuzen.")
