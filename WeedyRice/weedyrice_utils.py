import csv
import json
import math
import os
import re
import shutil
import zipfile
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision.models import resnet50


FILENAME_RE = re.compile(
    r"DJI_DateTime_(?P<date>\d{4}_\d{2}_\d{2})_"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_"
    r"(?P<seq>\d{4})_lat_(?P<lat>-?\d+\.\d+)_"
    r"lon_(?P<lon>-?\d+\.\d+)_alt_(?P<alt>\d+\.\d+)m",
    re.IGNORECASE,
)

RGB_SENSOR_WIDTH_MM = 17.3
RGB_SENSOR_HEIGHT_MM = 13.0
RGB_FOCAL_LENGTH_MM = 12.29
MIN_CONTOUR_AREA = 80
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_dataset_root(path):
    root = Path(path)
    if (root / "RGB").is_dir() and (root / "Masks").is_dir() and (root / "Metadata").is_dir():
        return root
    for sub in root.rglob("readme.md"):
        candidate = sub.parent
        if (candidate / "RGB").is_dir() and (candidate / "Masks").is_dir():
            return candidate
    raise FileNotFoundError(f"WeedyRice veri seti bulunamadi: {path}")


def extract_dataset(zip_path, extract_dir):
    extract_root = Path(extract_dir)
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    return find_dataset_root(extract_root)


def load_split_lists(dataset_root):
    dataset_root = Path(dataset_root)
    split_names = {}
    for split in ("train", "val", "test"):
        split_file = dataset_root / f"{split}_list.txt"
        with open(split_file, encoding="utf-8") as handle:
            split_names[split] = [line.strip() for line in handle if line.strip()]
    return split_names


def stem_from_name(filename):
    name = Path(filename).name
    stem = Path(name).stem
    for suffix in ("_G", "_R", "_RE", "_NIR"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def parse_filename_metadata(filename):
    stem = stem_from_name(filename)
    match = FILENAME_RE.search(stem)
    if not match:
        return None
    date_text = match.group("date").replace("_", "-")
    return {
        "stem": stem,
        "filename": Path(filename).name,
        "acquisition_date": date_text,
        "time": f"{match.group('hour')}:{match.group('minute')}:00",
        "sequence_id": int(match.group("seq")),
        "latitude": float(match.group("lat")),
        "longitude": float(match.group("lon")),
        "altitude_m": float(match.group("alt")),
    }


def load_metadata_index(dataset_root, split_names=None):
    dataset_root = Path(dataset_root)
    mapping_path = dataset_root / "Metadata" / "filename_mapping.csv"
    meta_path = dataset_root / "Metadata" / "image_metadata.csv"

    standardized = {}
    if mapping_path.exists():
        with open(mapping_path, encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                original = row.get("original_filename", "").strip()
                standard = row.get("standardized_filename", "").strip()
                if original and standard:
                    standardized[original] = standard

    requested = None
    if split_names is not None:
        requested = {stem_from_name(name) for names in split_names.values() for name in names}

    index = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("sensor_type") != "RGB":
                    continue
                original = row.get("original_filename", "").strip()
                filename = standardized.get(original, original)
                record = parse_filename_metadata(filename)
                if record is None:
                    continue
                record["camera_model"] = row.get("camera_model", "").strip()
                record["sensor_type"] = row.get("sensor_type", "").strip()
                if requested is None or record["stem"] in requested:
                    index[record["stem"]] = record

    if requested is not None:
        for stem in requested:
            index.setdefault(stem, parse_filename_metadata(stem) or {"stem": stem})

    return index


def clean_binary_mask(mask, min_area=MIN_CONTOUR_AREA, open_kernel=3, close_kernel=5):
    binary = (mask > 0).astype(np.uint8)

    if open_kernel and open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if close_kernel and close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for comp_id in range(1, component_count):
        if stats[comp_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == comp_id] = 1
    return cleaned


def binary_mask_to_yolo_annotations(
    mask,
    min_area=MIN_CONTOUR_AREA,
    epsilon_ratio=0.0035,
    open_kernel=3,
    close_kernel=5,
):
    height, width = mask.shape[:2]
    binary = clean_binary_mask(
        mask,
        min_area=min_area,
        open_kernel=open_kernel,
        close_kernel=close_kernel,
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotations = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area or len(contour) < 3:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        points = []
        for x, y in approx:
            points.extend([
                float(np.clip(x / width, 0.0, 1.0)),
                float(np.clip(y / height, 0.0, 1.0)),
            ])
        annotations.append((0, points))
    return annotations


def load_yolo_polygons(label_path, image_shape):
    height, width = image_shape[:2]
    polygons = []
    if not os.path.exists(label_path):
        return polygons
    with open(label_path, encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.strip().split()
            if len(parts) < 7:
                continue
            coords = list(map(float, parts[1:]))
            points = []
            for idx in range(0, len(coords), 2):
                points.append([coords[idx] * width, coords[idx + 1] * height])
            polygons.append(np.array(points, dtype=np.float32))
    return polygons


def polygons_to_mask(polygons, image_shape):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        pts = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask


def contour_centroid(points):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return 0.0, 0.0
    moments = cv2.moments(pts.reshape(-1, 1, 2))
    if abs(moments["m00"]) > 1e-6:
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def polygon_area_px(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return abs(float(cv2.contourArea(pts)))


def summarize_binary_metrics(gt_mask, pred_mask):
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    union = int(np.logical_or(gt, pred).sum())
    gt_sum = int(gt.sum())
    pred_sum = int(pred.sum())
    iou = tp / union if union else 0.0
    dice = (2 * tp) / (gt_sum + pred_sum) if (gt_sum + pred_sum) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "gt_pixels": gt_sum,
        "pred_pixels": pred_sum,
    }


def footprint_dimensions_m(altitude_m):
    width_m = altitude_m * (RGB_SENSOR_WIDTH_MM / RGB_FOCAL_LENGTH_MM)
    height_m = altitude_m * (RGB_SENSOR_HEIGHT_MM / RGB_FOCAL_LENGTH_MM)
    return width_m, height_m


def _meters_to_latlon(base_lat, base_lon, east_m, north_m):
    lat_scale = 111_320.0
    lon_scale = max(1e-6, 111_320.0 * math.cos(math.radians(base_lat)))
    return base_lat + (north_m / lat_scale), base_lon + (east_m / lon_scale)


def build_heading_lookup(metadata_index):
    rgb_records = [value for value in metadata_index.values() if value and "latitude" in value and "longitude" in value]
    rgb_records.sort(key=lambda item: (item.get("acquisition_date", ""), item.get("time", ""), item.get("sequence_id", 0)))
    return rgb_records


def estimate_heading_deg(record, ordered_records):
    if not record or "latitude" not in record or "longitude" not in record:
        return 0.0
    same_day = [item for item in ordered_records if item.get("acquisition_date") == record.get("acquisition_date")]
    if len(same_day) < 2:
        return 0.0
    stems = [item["stem"] for item in same_day]
    try:
        idx = stems.index(record["stem"])
    except ValueError:
        return 0.0
    previous_record = same_day[max(0, idx - 1)]
    next_record = same_day[min(len(same_day) - 1, idx + 1)]
    if previous_record["stem"] == next_record["stem"]:
        return 0.0
    lat0 = math.radians(record["latitude"])
    east = (next_record["longitude"] - previous_record["longitude"]) * 111_320.0 * math.cos(lat0)
    north = (next_record["latitude"] - previous_record["latitude"]) * 111_320.0
    if abs(east) < 1e-6 and abs(north) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(east, north))


def pixel_to_world(point_xy, image_shape, record, ordered_records=None):
    if record is None or "latitude" not in record or "longitude" not in record:
        return None
    height, width = image_shape[:2]
    altitude_m = float(record.get("altitude_m", 20.0) or 20.0)
    footprint_w, footprint_h = footprint_dimensions_m(altitude_m)
    meters_per_px_x = footprint_w / width
    meters_per_px_y = footprint_h / height

    x, y = point_xy
    east = (x - (width / 2.0)) * meters_per_px_x
    north = -1.0 * (y - (height / 2.0)) * meters_per_px_y

    heading = estimate_heading_deg(record, ordered_records or [])
    angle = math.radians(heading)
    rot_east = east * math.cos(angle) + north * math.sin(angle)
    rot_north = -east * math.sin(angle) + north * math.cos(angle)
    lat, lon = _meters_to_latlon(record["latitude"], record["longitude"], rot_east, rot_north)
    return {
        "latitude": lat,
        "longitude": lon,
        "east_offset_m": rot_east,
        "north_offset_m": rot_north,
        "heading_deg": heading,
        "footprint_width_m": footprint_w,
        "footprint_height_m": footprint_h,
        "meters_per_px_x": meters_per_px_x,
        "meters_per_px_y": meters_per_px_y,
    }


def polygon_to_world(points, image_shape, record, ordered_records=None):
    world_points = []
    for x, y in np.asarray(points, dtype=np.float32):
        location = pixel_to_world((float(x), float(y)), image_shape, record, ordered_records)
        if location is None:
            return []
        world_points.append([location["latitude"], location["longitude"]])
    return world_points


def image_footprint_polygon(record, image_shape, ordered_records=None):
    height, width = image_shape[:2]
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return polygon_to_world(corners, image_shape, record, ordered_records)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_leaflet_map(output_path, center, title, groups):
    output_path = Path(output_path)
    payload = {"center": center, "title": title, "groups": groups}
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    html, body {{ height: 100%; margin: 0; font-family: 'Inter', sans-serif; background-color: #111827; color: #f9fafb; }}
    #map {{ height: calc(100% - 70px); width: 100%; z-index: 1; }}
    .header {{ padding: 16px 24px; background: #1f2937; border-bottom: 1px solid #374151; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); display: flex; justify-content: space-between; align-items: center; z-index: 10; position: relative; }}
    .header h1 {{ margin: 0; font-size: 20px; font-weight: 600; color: #f9fafb; }}
    .header p {{ margin: 4px 0 0 0; color: #9ca3af; font-size: 13px; font-weight: 400; }}
    .legend {{ background: #1f2937; padding: 12px 16px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); border: 1px solid #374151; color: #f9fafb; }}
    .legend h4 {{ margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #d1d5db; }}
    .legend-item {{ display: flex; align-items: center; margin-bottom: 6px; font-size: 12px; }}
    .legend-color {{ width: 16px; height: 16px; border-radius: 4px; margin-right: 8px; border: 1px solid rgba(255,255,255,0.2); }}
    
    /* Leaflet popup dark mode overrides */
    .leaflet-popup-content-wrapper, .leaflet-popup-tip {{ background: #1f2937; color: #f9fafb; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); border: 1px solid #374151; }}
    .leaflet-popup-content b {{ color: #60a5fa; }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>{title}</h1>
      <p>Gerçek GPS Koordinatları İle Semantik İzdüşüm (Georeferenced Semantic Footprint)</p>
    </div>
  </div>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const payload = {json.dumps(payload, ensure_ascii=False)};
    
    // Satelite view default
    const map = L.map('map').setView(payload.center, 20);
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 22,
      attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    }}).addTo(map);

    const overlayMaps = {{}};
    const bounds = [];

    function extendBounds(coords) {{ coords.forEach(pt => bounds.push(pt)); }}

    Object.entries(payload.groups).forEach(([groupName, features]) => {{
      const layer = L.layerGroup();
      features.forEach(feature => {{
        if (feature.kind === 'marker') {{
          const marker = L.circleMarker(feature.coords, {{
            radius: feature.radius || 6,
            color: feature.color || '#f59e0b',
            weight: 2,
            fillColor: feature.fillColor || '#f59e0b',
            fillOpacity: feature.fillOpacity ?? 0.85
          }});
          if (feature.popup) marker.bindPopup(feature.popup);
          marker.addTo(layer);
          bounds.push(feature.coords);
        }} else if (feature.kind === 'polygon') {{
          const polygon = L.polygon(feature.coords, {{
            color: feature.color || '#ef4444',
            weight: feature.weight || 2,
            fillColor: feature.fillColor || '#ef4444',
            fillOpacity: feature.fillOpacity ?? 0.3
          }});
          if (feature.popup) polygon.bindPopup(feature.popup);
          polygon.addTo(layer);
          extendBounds(feature.coords);
        }}
      }});
      layer.addTo(map);
      overlayMaps[groupName] = layer;
    }});

    L.control.layers(null, overlayMaps, {{ collapsed: false }}).addTo(map);
    
    // Custom Legend
    const legend = L.control({{ position: 'bottomright' }});
    legend.onAdd = function (map) {{
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML = `
        <h4>Harita Lejantı</h4>
        <div class="legend-item"><div class="legend-color" style="background: #f59e0b;"></div> GPS Merkezi</div>
        <div class="legend-item"><div class="legend-color" style="background: #f8c471; opacity: 0.5;"></div> Görüntü İzdüşümü (Footprint)</div>
        <div class="legend-item"><div class="legend-color" style="background: #5dade2; opacity: 0.5;"></div> Ground Truth (Gerçek)</div>
        <div class="legend-item"><div class="legend-color" style="background: #e74c3c; opacity: 0.5;"></div> Model Tahmini (Prediction)</div>
      `;
      return div;
    }};
    legend.addTo(map);

    if (bounds.length) {{ map.fitBounds(bounds, {{ padding: [30, 30] }}); }}
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


class WeedyRiceSemanticDataset(Dataset):
    def __init__(self, dataset_root, split, imgsz=1024, augment=False):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.imgsz = int(imgsz)
        self.augment = augment
        self.items = load_split_lists(self.dataset_root)[split]

    def __len__(self):
        return len(self.items)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))
        if random.random() < 0.2:
            image = np.ascontiguousarray(np.flipud(image))
            mask = np.ascontiguousarray(np.flipud(mask))
        if random.random() < 0.25:
            k = random.choice([1, 2, 3])
            image = np.ascontiguousarray(np.rot90(image, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
        if random.random() < 0.3:
            alpha = 1.0 + random.uniform(-0.12, 0.12)
            beta = random.uniform(-12.0, 12.0)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return image, mask

    def __getitem__(self, index):
        image_name = self.items[index]
        stem = stem_from_name(image_name)
        image_path = self.dataset_root / "RGB" / image_name
        mask_path = self.dataset_root / "Masks" / f"{stem}.png"

        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = clean_binary_mask(mask, min_area=MIN_CONTOUR_AREA, open_kernel=0, close_kernel=0).astype(np.float32)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float(),
            "name": image_name,
            "stem": stem,
        }


class IntermediateLayerGetter(torch.nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        original = dict(return_layers)
        pending = dict(return_layers)
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in pending:
                del pending[name]
            if not pending:
                break

        super().__init__(layers)
        self.return_layers = original

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out


class _SimpleSegmentationModel(torch.nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = torch.nn.functional.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class ASPPConv(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )


class ASPPPooling(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        x = self.relu(x)
        return torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(torch.nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super().__init__()
        out_channels = 256
        modules = [
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
            )
        ]
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = torch.nn.ModuleList(modules)
        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
        )

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return self.project(torch.cat(outputs, dim=1))


class DeepLabHeadV3Plus(torch.nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=(12, 24, 36)):
        super().__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(low_level_channels, 48, 1, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(in_channels, aspp_dilate)
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(304, 256, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, features):
        low_level = self.project(features["low_level"])
        out = self.aspp(features["out"])
        out = torch.nn.functional.interpolate(out, size=low_level.shape[2:], mode="bilinear", align_corners=False)
        return self.classifier(torch.cat([low_level, out], dim=1))


class DeepLabV3(_SimpleSegmentationModel):
    pass


class DeepLabV3Plus(DeepLabV3):
    def __init__(self, num_classes=1, backbone_weights=None, output_stride=8):
        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilate = [6, 12, 18]

        backbone = resnet50(weights=backbone_weights, replace_stride_with_dilation=replace_stride_with_dilation)
        backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "out", "layer1": "low_level"})
        classifier = DeepLabHeadV3Plus(2048, 256, num_classes, aspp_dilate)
        super().__init__(backbone, classifier)


def get_segmentation_model(name, out_channels=1, pretrained_backbone=True):
    key = str(name).strip().lower()
    if key in {"deeplabv3plus", "deeplabv3+"}:
        weights = "DEFAULT" if pretrained_backbone else None
        from torchvision.models import ResNet50_Weights
        backbone_weights = ResNet50_Weights.DEFAULT if weights == "DEFAULT" else None
        return DeepLabV3Plus(num_classes=out_channels, backbone_weights=backbone_weights)
    raise ValueError(f"Bilinmeyen model: {name}")


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def segmentation_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds) * targets).sum(dim=(1, 2, 3))
    union = tp + fp + fn
    iou = torch.where(union > 0, tp / union, torch.zeros_like(union))
    dice = torch.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), torch.zeros_like(tp))
    precision = torch.where((tp + fp) > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where((tp + fn) > 0, tp / (tp + fn), torch.zeros_like(tp))
    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


def logits_to_mask(logits, original_shape, threshold=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0, 0]
    mask = (probs >= threshold).astype(np.uint8)
    return cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)


def mask_to_polygons(mask, min_area=MIN_CONTOUR_AREA, epsilon_ratio=0.0035):
    binary = clean_binary_mask(mask, min_area=min_area, open_kernel=0, close_kernel=0)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area or len(contour) < 3:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) >= 3:
            polygons.append(approx.astype(np.float32))
    return polygons
