"""
DRONEQUBE Gorev 4 - WeedyRice semantic test raporu.
DeepLabV3+ checkpoint'i ile proje icindeki secili test goruntulerini degerlendirir,
PNG/CSV/JSON/PDF ve HTML harita uretir.
"""

import csv
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mpl-cache"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fpdf import FPDF

from weedyrice_utils import (
    build_heading_lookup,
    contour_centroid,
    ensure_dir,
    get_segmentation_model,
    image_footprint_polygon,
    logits_to_mask,
    mask_to_polygons,
    pixel_to_world,
    polygon_area_px,
    polygon_to_world,
    summarize_binary_metrics,
    write_json,
    write_leaflet_map,
    WeedyRiceSemanticDataset,
)


MODEL_PATH = BASE_DIR / "best.pt"
INFO_PATH = BASE_DIR / "dataset_info.json"
META_PATH = BASE_DIR / "metadata_index.json"
ALT_INFO_PATH = BASE_DIR / "yolo_dataset" / "dataset_info.json"
ALT_META_PATH = BASE_DIR / "yolo_dataset" / "metadata_index.json"
OUTPUT_DIR = BASE_DIR / "sonuclar"
IMAGE_DIR = OUTPUT_DIR / "gorseller"
TEST_IMAGE_DIR = BASE_DIR / "yolo_dataset" / "images" / "test"
TEST_MASK_DIR = BASE_DIR / "yolo_dataset" / "masks" / "test"
MAP_DIR = OUTPUT_DIR / "haritalar"
CLASS_NAME = "weedy_rice"


def overlay(image_rgb, mask, color=(220, 53, 69), alpha=0.45):
    out = image_rgb.copy()
    overlay_color = np.array(color, dtype=np.uint8)
    out[mask == 1] = (out[mask == 1] * (1 - alpha) + overlay_color * alpha).astype(np.uint8)
    return out


def load_runtime():
    assert MODEL_PATH.exists(), f"Model bulunamadi: {MODEL_PATH}"
    info_path = INFO_PATH if INFO_PATH.exists() else ALT_INFO_PATH
    meta_path = META_PATH if META_PATH.exists() else ALT_META_PATH
    assert info_path.exists(), f"dataset_info.json bulunamadi: {INFO_PATH} veya {ALT_INFO_PATH}"
    assert meta_path.exists(), f"metadata_index.json bulunamadi: {META_PATH} veya {ALT_META_PATH}"

    info = json.loads(info_path.read_text(encoding="utf-8"))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return info, metadata


def image_area_metrics(record, image_shape, ordered_records):
    location = pixel_to_world((image_shape[1] / 2.0, image_shape[0] / 2.0), image_shape, record, ordered_records)
    if not location:
        return {"pixel_area_m2": 0.0, "image_area_m2": 0.0}
    image_area_m2 = location["footprint_width_m"] * location["footprint_height_m"]
    return {"pixel_area_m2": image_area_m2 / (image_shape[0] * image_shape[1]), "image_area_m2": image_area_m2}


info, metadata_index = load_runtime()
ordered_records = build_heading_lookup(metadata_index)
dataset_root = Path(info["source_root"])
imgsz = int(info.get("imgsz", 1024))

ensure_dir(OUTPUT_DIR)
ensure_dir(IMAGE_DIR)
ensure_dir(MAP_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)
arch = checkpoint.get("arch", info.get("model", "deeplabv3plus"))
model = get_segmentation_model(arch, out_channels=1, pretrained_backbone=False).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

assert TEST_IMAGE_DIR.exists(), f"Test RGB klasoru bulunamadi: {TEST_IMAGE_DIR}"
assert TEST_MASK_DIR.exists(), f"Test maske klasoru bulunamadi: {TEST_MASK_DIR}"

test_images = sorted(TEST_IMAGE_DIR.glob("*.JPG"))
assert test_images, f"Test RGB klasorunde gorsel yok: {TEST_IMAGE_DIR}"

print(f"Model: {MODEL_PATH}")
print(f"Test goruntu sayisi: {len(test_images)}")
print(f"Cikti: {OUTPUT_DIR}\n")

results = []
all_detections = []
map_groups = {"Merkezler": [], "Footprint": [], "GT Segment": [], "Tahmin Segment": []}

for idx, image_path in enumerate(test_images, start=1):
    image_name = image_path.name
    stem = image_path.stem
    gt_path = TEST_MASK_DIR / f"{stem}.png"

    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    assert gt_path.exists(), f"Maske bulunamadi: {gt_path}"
    gt_mask = (cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

    resized = cv2.resize(image_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    resized = (resized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized = np.transpose(resized, (2, 0, 1))
    input_tensor = torch.from_numpy(resized).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
    pred_mask = logits_to_mask(logits, image_rgb.shape)

    gt_polygons = mask_to_polygons(gt_mask)
    pred_polygons = mask_to_polygons(pred_mask)
    metrics = summarize_binary_metrics(gt_mask, pred_mask)

    record = metadata_index.get(stem)
    area_info = image_area_metrics(record, image_rgb.shape, ordered_records)
    image_area_m2 = area_info["image_area_m2"]
    gt_area_m2 = metrics["gt_pixels"] * area_info["pixel_area_m2"]
    pred_area_m2 = metrics["pred_pixels"] * area_info["pixel_area_m2"]

    detections = []
    for det_idx, polygon in enumerate(pred_polygons, start=1):
        cx, cy = contour_centroid(polygon)
        world = pixel_to_world((cx, cy), image_rgb.shape, record, ordered_records)
        polygon_world = polygon_to_world(polygon, image_rgb.shape, record, ordered_records) if world else []
        area_px = polygon_area_px(polygon)
        detection_id = f"{stem}_weed_{det_idx:03d}"
        detection = {
            "id": detection_id,
            "image": image_name,
            "class": CLASS_NAME,
            "cx_px": round(float(cx), 2),
            "cy_px": round(float(cy), 2),
            "area_px": round(area_px, 2),
            "area_m2": round(area_px * area_info["pixel_area_m2"], 4),
            "latitude": round(world["latitude"], 7) if world else None,
            "longitude": round(world["longitude"], 7) if world else None,
            "east_offset_m": round(world["east_offset_m"], 3) if world else None,
            "north_offset_m": round(world["north_offset_m"], 3) if world else None,
        }
        detections.append(detection)
        all_detections.append(detection)
        if polygon_world:
            map_groups["Tahmin Segment"].append(
                {
                    "kind": "polygon",
                    "coords": polygon_world,
                    "color": "#c0392b",
                    "fillColor": "#e74c3c",
                    "fillOpacity": 0.22,
                    "popup": f"<b>{image_name}</b><br>Tahmin alan: {detection['area_m2']:.3f} m2",
                }
            )

    for polygon in gt_polygons:
        polygon_world = polygon_to_world(polygon, image_rgb.shape, record, ordered_records) if record else []
        if polygon_world:
            map_groups["GT Segment"].append(
                {
                    "kind": "polygon",
                    "coords": polygon_world,
                    "color": "#1f77b4",
                    "fillColor": "#5dade2",
                    "fillOpacity": 0.16,
                    "popup": f"<b>{image_name}</b><br>Ground truth segment",
                }
            )

    if record:
        popup = (
            f"<b>{image_name}</b><br>"
            f"Lat/Lon: {record['latitude']:.7f}, {record['longitude']:.7f}<br>"
            f"Altitude: {record['altitude_m']:.2f} m<br>"
            f"GT alan: {gt_area_m2:.3f} m2<br>"
            f"Tahmin alan: {pred_area_m2:.3f} m2<br>"
            f"IoU: {metrics['iou']:.3f}"
        )
        map_groups["Merkezler"].append(
            {"kind": "marker", "coords": [record["latitude"], record["longitude"]], "color": "#111827", "fillColor": "#f59e0b", "radius": 5, "popup": popup}
        )
        footprint = image_footprint_polygon(record, image_rgb.shape, ordered_records)
        if footprint:
            map_groups["Footprint"].append(
                {"kind": "polygon", "coords": footprint, "color": "#f39c12", "fillColor": "#f8c471", "fillOpacity": 0.08, "popup": popup + "<br>Approximate image footprint"}
            )

        single_groups = {
            "Merkez": [
                {"kind": "marker", "coords": [record["latitude"], record["longitude"]], "color": "#111827", "fillColor": "#f59e0b", "radius": 6, "popup": popup}
            ],
            "Footprint": map_groups["Footprint"][-1:] if footprint else [],
            "GT Segment": [],
            "Tahmin Segment": [],
        }
        for polygon in gt_polygons:
            polygon_world = polygon_to_world(polygon, image_rgb.shape, record, ordered_records)
            if polygon_world:
                single_groups["GT Segment"].append(
                    {"kind": "polygon", "coords": polygon_world, "color": "#1f77b4", "fillColor": "#5dade2", "fillOpacity": 0.16, "popup": f"<b>{image_name}</b><br>Ground truth segment"}
                )
        for detection, polygon in zip(detections, pred_polygons):
            polygon_world = polygon_to_world(polygon, image_rgb.shape, record, ordered_records)
            if polygon_world:
                single_groups["Tahmin Segment"].append(
                    {
                        "kind": "polygon",
                        "coords": polygon_world,
                        "color": "#c0392b",
                        "fillColor": "#e74c3c",
                        "fillOpacity": 0.22,
                        "popup": f"<b>{image_name}</b><br>ID: {detection['id']}<br>Tahmin alan: {detection['area_m2']:.3f} m2",
                    }
                )
        write_leaflet_map(MAP_DIR / f"{stem}.html", [record["latitude"], record["longitude"]], f"WeedyRice Harita - {image_name}", single_groups)

    density = pred_area_m2 / image_area_m2 * 100.0 if image_area_m2 else 0.0
    results.append(
        {
            "sample_id": stem,
            "image": image_name,
            "acquisition_date": record.get("acquisition_date") if record else None,
            "latitude": round(record["latitude"], 7) if record else None,
            "longitude": round(record["longitude"], 7) if record else None,
            "altitude_m": round(record["altitude_m"], 3) if record else None,
            "image_area_m2": round(image_area_m2, 4),
            "gt_instances": len(gt_polygons),
            "pred_instances": len(pred_polygons),
            "gt_weed_area_m2": round(gt_area_m2, 4),
            "pred_weed_area_m2": round(pred_area_m2, 4),
            "weed_density_pct": round(density, 2),
            "iou": round(metrics["iou"], 4),
            "dice": round(metrics["dice"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
        }
    )

    diff = np.zeros_like(image_rgb)
    diff[np.logical_and(gt_mask == 1, pred_mask == 1)] = [34, 197, 94]
    diff[np.logical_and(gt_mask == 1, pred_mask == 0)] = [239, 68, 68]
    diff[np.logical_and(gt_mask == 0, pred_mask == 1)] = [245, 158, 11]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(image_rgb); axes[0].set_title("Orijinal"); axes[0].axis("off")
    axes[1].imshow(overlay(image_rgb, gt_mask, color=(59, 130, 246))); axes[1].set_title(f"Ground Truth\nAlan: {gt_area_m2:.3f} m2"); axes[1].axis("off")
    axes[2].imshow(overlay(image_rgb, pred_mask, color=(239, 68, 68))); axes[2].set_title(f"Tahmin\nAlan: {pred_area_m2:.3f} m2"); axes[2].axis("off")
    axes[3].imshow(diff); axes[3].set_title(f"Hata Haritasi\nIoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f}"); axes[3].axis("off")
    plt.suptitle(f"{image_name} | alan ~ {image_area_m2:.2f} m2 | yogunluk %{density:.2f}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"analiz_{idx:03d}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[{idx:03d}/{len(test_images)}] {image_name} | GT_m2={gt_area_m2:.3f} Pred_m2={pred_area_m2:.3f} IoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f}")


mean_iou = float(np.mean([row["iou"] for row in results]))
mean_dice = float(np.mean([row["dice"] for row in results]))
mean_precision = float(np.mean([row["precision"] for row in results]))
mean_recall = float(np.mean([row["recall"] for row in results]))
total_image_area = float(sum(row["image_area_m2"] for row in results))
total_gt_area = float(sum(row["gt_weed_area_m2"] for row in results))
total_pred_area = float(sum(row["pred_weed_area_m2"] for row in results))
mean_density = float(np.mean([row["weed_density_pct"] for row in results]))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
indices = np.arange(len(results))
axes[0][0].bar(indices, [row["gt_weed_area_m2"] / row["image_area_m2"] * 100 if row["image_area_m2"] else 0 for row in results], label="GT Yogunluk %", color="#60a5fa", alpha=0.85)
axes[0][0].bar(indices, [row["weed_density_pct"] for row in results], label="Tahmin Yogunluk %", color="#ef4444", alpha=0.55)
axes[0][0].set_title("Yabanci Ot Yogunlugu (%)"); axes[0][0].legend()
axes[0][1].plot(indices, [row["iou"] for row in results], marker="o", color="#1d4ed8", label="IoU")
axes[0][1].plot(indices, [row["dice"] for row in results], marker="s", color="#059669", label="Dice")
axes[0][1].set_title("Segmentasyon metrikleri"); axes[0][1].set_ylim(0, 1); axes[0][1].legend()
axes[1][0].bar(indices, [row["pred_weed_area_m2"] for row in results], color="#f59e0b"); axes[1][0].set_title("Tahmin Edilen Alan (m2)")
axes[1][1].axis("off")
axes[1][1].text(0.08, 0.95, f"Test goruntu: {len(results)}\n\nOrtalama IoU: {mean_iou:.4f}\nOrtalama Dice: {mean_dice:.4f}\nOrtalama Precision: {mean_precision:.4f}\nOrtalama Recall: {mean_recall:.4f}\n\nToplam goruntu alani: {total_image_area:.2f} m2\nGT alani: {total_gt_area:.2f} m2\nTahmin alani: {total_pred_area:.2f} m2\nOrtalama yogunluk: %{mean_density:.2f}\n\nKonum notu:\nMerkez GPS gercektir.\nFootprint ve segment harita cizimleri yaklasiktir.", transform=axes[1][1].transAxes, va="top", fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="#f3f4f6"))
plt.suptitle("WeedyRice Semantic Test Ozeti", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "genel_ozet.png", dpi=180, bbox_inches="tight")
plt.close()

with open(OUTPUT_DIR / "tespit_detay.csv", "w", newline="", encoding="utf-8") as handle:
    if all_detections:
        writer = csv.DictWriter(handle, fieldnames=all_detections[0].keys()); writer.writeheader(); writer.writerows(all_detections)

with open(OUTPUT_DIR / "goruntu_ozet.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)

summary = {
    "model": arch,
    "dataset": "WeedyRice-RGBMS-DB",
    "class_name": CLASS_NAME,
    "test_image_count": len(results),
    "mean_iou": round(mean_iou, 4),
    "mean_dice": round(mean_dice, 4),
    "mean_precision": round(mean_precision, 4),
    "mean_recall": round(mean_recall, 4),
    "total_image_area_m2": round(total_image_area, 4),
    "total_gt_weed_area_m2": round(total_gt_area, 4),
    "total_pred_weed_area_m2": round(total_pred_area, 4),
    "mean_density_pct": round(mean_density, 4),
}
write_json(OUTPUT_DIR / "sonuclar.json", {"summary": summary, "per_image": results, "detections": all_detections})

map_center = [float(np.mean([row["latitude"] for row in results if row["latitude"] is not None])), float(np.mean([row["longitude"] for row in results if row["longitude"] is not None]))]
write_leaflet_map(OUTPUT_DIR / "weedy_rice_harita.html", map_center, "WeedyRice Semantic Test Haritasi", map_groups)

with open(OUTPUT_DIR / "parsel_yogunluk.csv", "w", newline="", encoding="utf-8") as handle:
    fieldnames = ["acquisition_date", "image_count", "total_image_area_m2", "total_pred_weed_area_m2", "mean_density_pct", "mean_iou", "note"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    by_date = {}
    for row in results:
        key = row["acquisition_date"] or "unknown_date"
        by_date.setdefault(key, []).append(row)
    for key, rows in sorted(by_date.items()):
        total_area = sum(item["image_area_m2"] for item in rows)
        total_pred = sum(item["pred_weed_area_m2"] for item in rows)
        writer.writerow(
            {
                "acquisition_date": key,
                "image_count": len(rows),
                "total_image_area_m2": round(total_area, 4),
                "total_pred_weed_area_m2": round(total_pred, 4),
                "mean_density_pct": round(float(np.mean([item["weed_density_pct"] for item in rows])), 4),
                "mean_iou": round(float(np.mean([item["iou"] for item in rows])), 4),
                "note": "Date-grouped parcel-style summary from georeferenced image centers and approximate footprints.",
            }
        )


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 10, "DRONEQUBE - WeedyRice Semantic Test Raporu", new_x="LMARGIN", new_y="NEXT", align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Sayfa {self.page_no()}", new_x="RIGHT", new_y="TOP", align="C")


pdf = Report()
pdf.add_page()
pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5, f"Model: {arch}\nTest goruntu: {len(results)}\nOrtalama IoU: {mean_iou:.4f}\nOrtalama Dice: {mean_dice:.4f}\nToplam tahmin alani: {total_pred_area:.2f} m2\n\nHarita notu: Merkez koordinatlar metadata'dan gelir. Segment ve footprint cizimleri yaklasiktir.\nHer test goruntusu icin ayri HTML harita uretilmistir: sonuclar/haritalar/")
if (IMAGE_DIR / "genel_ozet.png").exists():
    pdf.image(str(IMAGE_DIR / "genel_ozet.png"), x=8, w=194)

for idx, row in enumerate(results[: min(20, len(results))], start=1):
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"{idx}. {row['image']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, f"Lat/Lon: {row['latitude']}, {row['longitude']}\nAltitude: {row['altitude_m']} m\nIoU: {row['iou']:.4f} | Dice: {row['dice']:.4f}\nGT alan: {row['gt_weed_area_m2']:.3f} m2 | Tahmin alan: {row['pred_weed_area_m2']:.3f} m2\nYogunluk: %{row['weed_density_pct']:.2f}")
    panel = IMAGE_DIR / f"analiz_{idx:03d}.png"
    if panel.exists():
        pdf.image(str(panel), x=5, w=200)

pdf.output(str(OUTPUT_DIR / "WeedyRice_Test_Raporu.pdf"))

print("\nOzet")
print(f"Mean IoU:  {mean_iou:.4f}")
print(f"Mean Dice: {mean_dice:.4f}")
print(f"HTML map:  {OUTPUT_DIR / 'weedy_rice_harita.html'}")
print(f"PDF:       {OUTPUT_DIR / 'WeedyRice_Test_Raporu.pdf'}")
