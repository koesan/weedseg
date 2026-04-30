"""
DRONEQUBE Gorev 4 - Tekil WeedyRice semantic analiz.
DeepLabV3+ checkpoint'i ile tek bir RGB goruntuyu analiz eder.
"""

import argparse
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
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_heading_lookup,
    contour_centroid,
    ensure_dir,
    get_segmentation_model,
    image_footprint_polygon,
    logits_to_mask,
    mask_to_polygons,
    parse_filename_metadata,
    pixel_to_world,
    polygon_area_px,
    polygon_to_world,
    write_json,
    write_leaflet_map,
)


MODEL_PATH = BASE_DIR / "best.pt"
META_PATH = BASE_DIR / "metadata_index.json"
INFO_PATH = BASE_DIR / "dataset_info.json"
CLASS_NAME = "weedy_rice"


def overlay(image_rgb, mask, alpha=0.45):
    out = image_rgb.copy()
    color = np.array([220, 53, 69], dtype=np.uint8)
    out[mask == 1] = (out[mask == 1] * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Tekil WeedyRice semantic analiz")
    parser.add_argument("image", help="Analiz edilecek RGB goruntu")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Model yolu")
    parser.add_argument("--latitude", type=float, help="Merkez latitude override")
    parser.add_argument("--longitude", type=float, help="Merkez longitude override")
    parser.add_argument("--altitude", type=float, help="Altitude metre override")
    return parser.parse_args()


def preprocess(image_rgb, imgsz):
    resized = cv2.resize(image_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    resized = (resized - IMAGENET_MEAN) / IMAGENET_STD
    resized = np.transpose(resized, (2, 0, 1))
    return torch.from_numpy(resized).float().unsqueeze(0)


def main():
    args = parse_args()
    image_path = Path(args.image).resolve()
    model_path = Path(args.model).resolve()

    assert image_path.exists(), f"Gorsel bulunamadi: {image_path}"
    assert model_path.exists(), f"Model bulunamadi: {model_path}"

    info = json.loads(INFO_PATH.read_text(encoding="utf-8")) if INFO_PATH.exists() else {"imgsz": 1024}
    metadata_index = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}

    checkpoint = torch.load(model_path, map_location="cpu")
    imgsz = int(checkpoint.get("imgsz", info.get("imgsz", 1024)))
    arch = checkpoint.get("arch", info.get("model", "deeplabv3plus"))
    model = get_segmentation_model(arch, out_channels=1, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    out_dir = BASE_DIR / "sonuclar" / f"tekil_{image_path.stem}"
    ensure_dir(out_dir)

    record = metadata_index.get(image_path.stem) or parse_filename_metadata(image_path.name)
    if record is not None:
        if args.latitude is not None:
            record["latitude"] = args.latitude
        if args.longitude is not None:
            record["longitude"] = args.longitude
        if args.altitude is not None:
            record["altitude_m"] = args.altitude

    ordered_records = build_heading_lookup(metadata_index)

    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    with torch.no_grad():
        logits = model(preprocess(image_rgb, imgsz))
    pred_mask = logits_to_mask(logits, image_rgb.shape)
    polygons = mask_to_polygons(pred_mask)

    detections = []
    map_groups = {"Merkez": [], "Footprint": [], "Tahmin Segment": []}
    area_m2_total = 0.0

    area_scale = None
    if record and "latitude" in record and "longitude" in record:
        center_info = pixel_to_world((width / 2.0, height / 2.0), image_rgb.shape, record, ordered_records)
        image_area_m2 = center_info["footprint_width_m"] * center_info["footprint_height_m"]
        area_scale = image_area_m2 / (width * height)
    else:
        image_area_m2 = None

    for det_idx, polygon in enumerate(polygons, start=1):
        cx, cy = contour_centroid(polygon)
        world = pixel_to_world((cx, cy), image_rgb.shape, record, ordered_records)
        poly_world = polygon_to_world(polygon, image_rgb.shape, record, ordered_records) if world else []
        area_px = polygon_area_px(polygon)
        area_m2 = area_px * area_scale if area_scale is not None else None
        if area_m2 is not None:
            area_m2_total += area_m2
        detection_id = f"{image_path.stem}_weed_{det_idx:03d}"
        detections.append(
            {
                "id": detection_id,
                "class": CLASS_NAME,
                "cx_px": round(float(cx), 2),
                "cy_px": round(float(cy), 2),
                "area_px": round(area_px, 2),
                "area_m2": round(area_m2, 4) if area_m2 is not None else None,
                "latitude": round(world["latitude"], 7) if world else None,
                "longitude": round(world["longitude"], 7) if world else None,
                "east_offset_m": round(world["east_offset_m"], 3) if world else None,
                "north_offset_m": round(world["north_offset_m"], 3) if world else None,
            }
        )
        if poly_world:
            map_groups["Tahmin Segment"].append({"kind": "polygon", "coords": poly_world, "color": "#c0392b", "fillColor": "#e74c3c", "fillOpacity": 0.24, "popup": f"<b>{image_path.name}</b><br>ID: {detection_id}<br>Tahmin alan: {detections[-1]['area_m2']} m2"})

    if record and "latitude" in record and "longitude" in record:
        popup = f"<b>{image_path.name}</b><br>Lat/Lon: {record['latitude']:.7f}, {record['longitude']:.7f}<br>Altitude: {record.get('altitude_m', 20.0):.2f} m"
        map_groups["Merkez"].append({"kind": "marker", "coords": [record["latitude"], record["longitude"]], "color": "#111827", "fillColor": "#f59e0b", "radius": 6, "popup": popup})
        footprint = image_footprint_polygon(record, image_rgb.shape, ordered_records)
        if footprint:
            map_groups["Footprint"].append({"kind": "polygon", "coords": footprint, "color": "#f39c12", "fillColor": "#f8c471", "fillOpacity": 0.08, "popup": popup + "<br>Approximate image footprint"})

    density_pct = area_m2_total / image_area_m2 * 100.0 if image_area_m2 else None

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].imshow(image_rgb); axes[0].set_title("Orijinal"); axes[0].axis("off")
    axes[1].imshow(overlay(image_rgb, pred_mask)); axes[1].set_title(f"Segmentasyon\nAlan: {area_m2_total:.3f} m2" if image_area_m2 else "Segmentasyon"); axes[1].axis("off")
    mask_view = np.zeros_like(image_rgb); mask_view[pred_mask == 1] = [220, 53, 69]
    axes[2].imshow(mask_view); axes[2].set_title("Maske"); axes[2].axis("off")
    plt.suptitle(image_path.name, fontsize=14, fontweight="bold"); plt.tight_layout()
    seg_path = out_dir / "segmentasyon.png"; plt.savefig(seg_path, dpi=150, bbox_inches="tight"); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].imshow(image_rgb, alpha=0.75)
    for det in detections:
        axes[0].plot(det["cx_px"], det["cy_px"], "o", color="#e74c3c", markersize=5, markeredgecolor="white")
    axes[0].set_title("Lokal konumlar"); axes[0].axis("off")
    if image_area_m2: axes[1].bar(["Alan (m2)"], [area_m2_total], color=["#ef4444"]); axes[1].set_title("Tahmin Alani")
    else: axes[1].axis("off")
    axes[2].axis("off")
    info_text = f"Goruntu: {image_path.name}\nBoyut: {width}x{height} px\nMask px: {int(pred_mask.sum())}\n"
    if image_area_m2 is not None:
        info_text += f"Alan m2: {area_m2_total:.4f}\nYogunluk: %{density_pct:.2f}\n"
    if record and "latitude" in record:
        info_text += f"Merkez lat/lon: {record['latitude']:.7f}, {record['longitude']:.7f}\nAltitude: {record.get('altitude_m',20.0):.2f} m\n"
    info_text += "\nHarita notu:\nMerkez GPS gercektir.\nSegment poligonlari approximate georeference'dir."
    axes[2].text(0.04, 0.96, info_text, transform=axes[2].transAxes, va="top", fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="#f3f4f6"))
    plt.suptitle("Analiz detaylari", fontsize=14, fontweight="bold"); plt.tight_layout()
    detail_path = out_dir / "detay.png"; plt.savefig(detail_path, dpi=150, bbox_inches="tight"); plt.close()

    if record and "latitude" in record and "longitude" in record:
        write_leaflet_map(out_dir / "weedy_rice_harita.html", [record["latitude"], record["longitude"]], f"WeedyRice Tekil Harita - {image_path.name}", map_groups)

    payload = {
        "image": image_path.name,
        "model": arch,
        "image_area_m2": round(image_area_m2, 4) if image_area_m2 is not None else None,
        "predicted_area_m2": round(area_m2_total, 4) if image_area_m2 is not None else None,
        "density_pct": round(density_pct, 4) if density_pct is not None else None,
        "center_latitude": round(record["latitude"], 7) if record and "latitude" in record else None,
        "center_longitude": round(record["longitude"], 7) if record and "longitude" in record else None,
        "altitude_m": round(record["altitude_m"], 3) if record and "altitude_m" in record else None,
        "detections": detections,
        "world_map_note": "Center GPS is real metadata. Footprint and polygon geolocation are approximate.",
    }
    write_json(out_dir / "sonuc.json", payload)

    class Report(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 13)
            self.cell(0, 10, "DRONEQUBE - WeedyRice Tekil Semantic Analiz", new_x="LMARGIN", new_y="NEXT", align="C")

        def footer(self):
            self.set_y(-15); self.set_font("Helvetica", "I", 8); self.cell(0, 10, f"Sayfa {self.page_no()}", new_x="RIGHT", new_y="TOP", align="C")

    pdf = Report()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, f"Goruntu: {image_path.name}\nMerkez lat/lon: {payload['center_latitude']}, {payload['center_longitude']}\nAltitude: {payload['altitude_m']} m\nTahmin alan: {payload['predicted_area_m2']} m2\nYogunluk: {payload['density_pct']}\n\nNot: Merkez koordinat gercektir. Segment poligonlarinin dunya haritasina yerlestirilmesi yaklasiktir.")
    if seg_path.exists(): pdf.image(str(seg_path), x=5, w=200)
    pdf.add_page()
    if detail_path.exists(): pdf.image(str(detail_path), x=5, w=200)
    pdf.output(str(out_dir / f"Analiz_{image_path.stem}.pdf"))

    print(f"Gorsel: {image_path}")
    print(f"Model:  {model_path}")
    print(f"Alan:   {payload['predicted_area_m2']} m2")
    print(f"JSON:   {out_dir / 'sonuc.json'}")
    if (out_dir / "weedy_rice_harita.html").exists():
        print(f"Harita: {out_dir / 'weedy_rice_harita.html'}")


if __name__ == "__main__":
    main()
