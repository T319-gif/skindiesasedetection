import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- Config ---
IMG_DIR_1 = "HAM10000_images_part_1"
IMG_DIR_2 = "HAM10000_images_part_2"
METADATA = "HAM10000_metadata.csv"
OUTPUT_DIR = "data"
SPLIT = (0.7, 0.15, 0.15)  # train, val, test split ratios

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load metadata
df = pd.read_csv(r"C:\Users\TZZS\Downloads\archive\HAM10000_metadata.csv")
print(df.head())

# Get unique lesion_ids for patient-level split
unique_lesions = df["lesion_id"].unique()
train_lesions, temp_lesions = train_test_split(unique_lesions, test_size=(1 - SPLIT[0]), random_state=42)
val_lesions, test_lesions = train_test_split(temp_lesions, test_size=SPLIT[2] / (SPLIT[1] + SPLIT[2]), random_state=42)

def assign_split(row):
    if row["lesion_id"] in train_lesions:
        return "train"
    elif row["lesion_id"] in val_lesions:
        return "val"
    else:
        return "test"

df["split"] = df.apply(assign_split, axis=1)
print(df["split"].value_counts())

# Make directories for each class
for split in ["train", "val", "test"]:
    for cls in df["dx"].unique():
        Path(f"{OUTPUT_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

# Copy images to folders
img_dirs = [IMG_DIR_1, IMG_DIR_2]
for _, row in df.iterrows():
    image_file = row["image_id"] + ".jpg"
    split = row["split"]
    label = row["dx"]
    
    # find the image file in either part 1 or part 2
    src_path = None
    for d in img_dirs:
        candidate = Path(d) / image_file
        if candidate.exists():
            src_path = candidate
            break
    if src_path:
        dst_path = Path(f"{OUTPUT_DIR}/{split}/{label}/{image_file}")
        shutil.copy(src_path, dst_path)

print("✅ Dataset prepared in folder:", OUTPUT_DIR)
