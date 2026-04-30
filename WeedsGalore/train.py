"""
DRONEQUBE Gorev 4 — YOLOv8l-seg Egitim Pipeline
WeedsGalore veri setini YOLO formatina donusturur ve model egitir.
"""

import os
import zipfile
import shutil
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

# Kullanici girisleri

DATASET_ZIP = input("Veri seti ZIP yolu: ").strip()
SAVE_DIR    = input("Cikti kayit yolu: ").strip()

os.makedirs(SAVE_DIR, exist_ok=True)
YOLO_DIR  = os.path.join(SAVE_DIR, "yolo_dataset")
TRAIN_DIR = os.path.join(SAVE_DIR, "egitim")

# Veri setini cikart 

EXTRACT_DIR = "/content/_weedsgalore_tmp"
if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)

with zipfile.ZipFile(DATASET_ZIP, "r") as z:
    z.extractall(EXTRACT_DIR)

DS = EXTRACT_DIR
for root, dirs, _ in os.walk(EXTRACT_DIR):
    if "splits" in dirs or "2023-05-25" in dirs:
        DS = root
        break

print(f"Veri seti: {DS}")

# Split dosyalarini oku 

splits_dir = os.path.join(DS, "splits")
split_ids  = {}

for sp in ["train", "val", "test"]:
    path = os.path.join(splits_dir, f"{sp}.txt")
    with open(path) as f:
        split_ids[sp] = [line.strip() for line in f if line.strip()]
    print(f"  {sp}: {len(split_ids[sp])} goruntu")

# Donusum fonksiyonlari 

# Sinif eslestirme: maize=crop(0), tum otlar=weed(1)
CLASS_MAP = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}
MIN_CONTOUR_AREA = 50

def combine_rgb(ds_root, sample_id):
    """R, G, B bantlarini birlestirip RGB goruntu olusturur."""
    date = sample_id[:10]
    base = os.path.join(ds_root, date, "images", sample_id)

    r = plt.imread(base + "_R.png")
    g = plt.imread(base + "_G.png")
    b = plt.imread(base + "_B.png")

    rgb = np.stack([r, g, b], axis=-1)
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
    return rgb

def instance_mask_to_polygons(sem_path, inst_path, img_h, img_w):
    """Instance maskesinden YOLO polygon formatina donusturur."""
    sem  = np.array(Image.open(sem_path))
    inst = np.array(Image.open(inst_path))
    annotations = []

    for inst_id in np.unique(inst):
        if inst_id == 0:
            continue

        binary = (inst == inst_id).astype(np.uint8)
        pixels = sem[binary == 1]
        if len(pixels) == 0:
            continue

        sem_class = int(np.bincount(pixels).argmax())
        if sem_class not in CLASS_MAP:
            continue

        yolo_class = CLASS_MAP[sem_class]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA or len(cnt) < 3:
                continue

            cnt = cnt.squeeze()
            if cnt.ndim != 2:
                continue

            epsilon = 0.005 * cv2.arcLength(cnt.reshape(-1, 1, 2), True)
            approx  = cv2.approxPolyDP(cnt.reshape(-1, 1, 2), epsilon, True).squeeze()

            if approx.ndim != 2 or len(approx) < 3:
                continue

            points = []
            for pt in approx:
                points.extend([
                    np.clip(pt[0] / img_w, 0, 1),
                    np.clip(pt[1] / img_h, 0, 1)
                ])
            annotations.append((yolo_class, points))

    return annotations

# Veri donusumu 

stats = defaultdict(lambda: {"img": 0, "crop": 0, "weed": 0})

for split, ids in split_ids.items():
    img_dir = os.path.join(YOLO_DIR, "images", split)
    lbl_dir = os.path.join(YOLO_DIR, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i, sid in enumerate(ids):
        date    = sid[:10]
        sem_p   = os.path.join(DS, date, "semantics", sid + ".png")
        inst_p  = os.path.join(DS, date, "instances", sid + ".png")

        if not os.path.exists(sem_p) or not os.path.exists(inst_p):
            continue

        rgb = combine_rgb(DS, sid)
        h, w = rgb.shape[:2]

        cv2.imwrite(os.path.join(img_dir, sid + ".png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        anns = instance_mask_to_polygons(sem_p, inst_p, h, w)
        with open(os.path.join(lbl_dir, sid + ".txt"), "w") as f:
            for cls_id, pts in anns:
                coords = " ".join(f"{x:.6f}" for x in pts)
                f.write(f"{cls_id} {coords}\n")

        n_crop = sum(1 for c, _ in anns if c == 0)
        n_weed = sum(1 for c, _ in anns if c == 1)
        stats[split]["img"]  += 1
        stats[split]["crop"] += n_crop
        stats[split]["weed"] += n_weed

        if (i + 1) % 25 == 0:
            print(f"  {split} [{i+1}/{len(ids)}]")

    print(f"  {split}: {stats[split]}")

yaml_path = os.path.join(YOLO_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(
        f"path: {YOLO_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n"
        f"  0: crop\n"
        f"  1: weed\n"
    )
print(f"data.yaml -> {yaml_path}")

# Model egitimi — YOLOv8l-seg 

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK'}")

model = YOLO("yolov8l-seg.pt")

model.train(
    data=yaml_path,
    epochs=200,
    imgsz=1024,
    batch=2,
    patience=50,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=90.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.5,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.3,
    close_mosaic=20,
    project=TRAIN_DIR,
    name="yolov8l_seg",
    save=True,
    save_period=50,
    plots=True,
    workers=2,
    seed=42,
    val=True,
    verbose=True,
)

# Test seti degerlendirme 

best_path  = os.path.join(TRAIN_DIR, "yolov8l_seg", "weights", "best.pt")
best_model = YOLO(best_path)

metrics = best_model.val(data=yaml_path, split="test", imgsz=1024, batch=2, plots=True, save_json=True)

print(f"\nBox  mAP50:    {metrics.box.map50:.4f}")
print(f"Box  mAP50-95: {metrics.box.map:.4f}")
print(f"Mask mAP50:    {metrics.seg.map50:.4f}")
print(f"Mask mAP50-95: {metrics.seg.map:.4f}")

for i, name in enumerate(["crop", "weed"]):
    print(f"  [{name}] P={metrics.box.p[i]:.3f}  R={metrics.box.r[i]:.3f}  AP50={metrics.box.ap50[i]:.3f}")

print(f"\nAgirliklar: {best_path}")
