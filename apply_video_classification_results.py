import glob
import pandas as pd
import os
import shutil
from pathlib import Path
import tqdm


df = pd.read_csv(
    os.path.join("./video_classifier_logs", "240619_video_class_label.csv"),
    index_col=0
)
df = df[~df["video_class"].isna()]  # label이 없는 것들만 선택

df.patient_id = df.patient_id.astype(str)
df.study_date = df.study_date.astype(str)
df.video_class = df.video_class.astype(int)

print(df.head())
for i, row in df.iterrows():
    base_path = os.path.join(
        row["patient_id"],
        row["study_date"],
        row["modality"],
        row["file_name"],
    )
    if row["video_class"] == 2:
        continue

    src_path = os.path.join(
        "/mnt/nas/snubhcvc/raw/cag_ccta_1yr_all/data/",
        "2_unclassified",
        base_path
    )
    
    dest_path = os.path.join(
        "/mnt/nas/snubhcvc/raw/cag_ccta_1yr_all/data",
        str(row["video_class"]),
        base_path
    )
    path = Path(os.path.dirname(dest_path))
    path.mkdir(parents=True, exist_ok=True)
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        print(f"move {src_path} -----> {dest_path}.")
