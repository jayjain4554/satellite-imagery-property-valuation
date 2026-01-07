import os
import requests
import pandas as pd
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
TRAIN_PATH = "data/raw/train.xlsx"
TEST_PATH  = "data/raw/test.xlsx"

TRAIN_IMG_DIR = "data/images/train"
TEST_IMG_DIR  = "data/images/test"

IMG_SIZE = 256        # ESRI supports this cleanly
DELTA = 0.002         # bounding box size (~200m)

# ESRI World Imagery endpoint
ESRI_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/export"
)

# ===============================
# LOAD DATA
# ===============================
train_df = pd.read_excel(TRAIN_PATH)
test_df  = pd.read_excel(TEST_PATH)

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True)

# ===============================
# FETCH FUNCTION
# ===============================
def fetch_esri_image(lat, lon, save_path):
    bbox = f"{lon-DELTA},{lat-DELTA},{lon+DELTA},{lat+DELTA}"

    params = {
        "bbox": bbox,
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{IMG_SIZE},{IMG_SIZE}",
        "format": "png",
        "f": "image"
    }

    response = requests.get(ESRI_URL, params=params, timeout=20)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed at {lat},{lon}")

# ===============================
# DOWNLOAD LOOP
# ===============================
def download_images(df, folder, limit=None):
    if limit:
        df = df.head(limit)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        save_path = f"data/images/{folder}/{idx}.png"
        fetch_esri_image(row.lat, row.long, save_path)

# ⚠️ RECOMMENDED LIMITS
download_images(train_df, "train")
download_images(test_df, "test")

print("ESRI satellite images downloaded successfully")
