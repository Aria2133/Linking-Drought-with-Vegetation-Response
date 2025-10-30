import os
import subprocess

# 📂 مسیر پوشه‌ی GeoTIFF ها
folder = r"C:\path\to\output\clipped"

# 🗂️ اسم Asset Folder توی GEE
gee_asset_folder = "projects/gee-1-461220/assets/AriaAhmadi"


# همه‌ی فایل‌های tif رو لیست کن
files = [f for f in os.listdir(folder) if f.endswith('.tif')]

for f in files:
    local_path = os.path.join(folder, f)
    asset_id = f"{gee_asset_folder}/{os.path.splitext(f)[0]}"
    cmd = [
        "earthengine", "upload", "image",
        "--asset_id=" + asset_id,
        local_path
    ]
    print(f"Uploading: {f} → {asset_id}")
    subprocess.run(cmd)
