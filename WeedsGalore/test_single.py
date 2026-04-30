"""
DRONEQUBE Gorev 4 — Tekil Goruntu Analizi
Kullanicinin verdigi tek bir gorseli YOLOv8l-seg ile analiz eder.
Etiket gerektirmez, sadece gorsel yeterlidir.
"""

import os, sys, argparse, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from fpdf import FPDF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

parser = argparse.ArgumentParser(description="Tekil goruntu analizi")
parser.add_argument("image", help="Analiz edilecek gorsel yolu")
parser.add_argument("--conf", type=float, default=0.15, help="Confidence esigi (varsayilan: 0.15)")
parser.add_argument("--gsd", type=float, default=2.5, help="GSD mm/px (varsayilan: 2.5)")
parser.add_argument("--model", default=MODEL_PATH, help="Model yolu (varsayilan: best.pt)")
args = parser.parse_args()

IMG_PATH = os.path.abspath(args.image)
GSD = args.gsd / 1000  # mm -> metre
CONF = args.conf
IMGSZ = 1024
CLS = {0: "crop", 1: "weed"}

assert os.path.exists(IMG_PATH), f"Gorsel bulunamadi: {IMG_PATH}"
assert os.path.exists(args.model), f"Model bulunamadi: {args.model}"


def mask_centroid(binary_mask, fallback_xy):
    moments = cv2.moments(binary_mask, binaryImage=True)
    if moments["m00"] > 0:
        cx = int(round(moments["m10"] / moments["m00"]))
        cy = int(round(moments["m01"] / moments["m00"]))
        return cx, cy
    return fallback_xy

img_base = os.path.splitext(os.path.basename(IMG_PATH))[0]
OUT_DIR = os.path.join(BASE_DIR, "sonuclar", f"tekil_{img_base}")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Gorsel: {IMG_PATH}")
print(f"Model:  {args.model}")
print(f"GSD:    {args.gsd} mm/px | Conf: {CONF}\n")

model = YOLO(args.model)
img_bgr = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]
area_m2 = h * w * GSD * GSD

preds = model.predict(IMG_PATH, imgsz=IMGSZ, conf=CONF, verbose=False)[0]

pred_crop, pred_weed = 0, 0
pred_mask = np.zeros((h, w), dtype=np.uint8)
detections = []

if preds.masks is not None:
    for det_idx, (box, md) in enumerate(zip(preds.boxes, preds.masks), start=1):
        cls = int(box.cls[0])
        cf = float(box.conf[0])
        m = cv2.resize(md.data[0].cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        pred_mask[m == 1] = cls + 1
        apx = int(m.sum())
        box_cx, box_cy = int(box.xywh[0][0]), int(box.xywh[0][1])
        cx, cy = mask_centroid(m, (box_cx, box_cy))
        if cls == 0: pred_crop += 1
        else: pred_weed += 1
        det_id = f"{img_base}_{CLS[cls]}_{det_idx:03d}"
        detections.append({
            "id": det_id,
            "sinif": CLS[cls], "guven": round(cf, 3),
            "cx_px": cx, "cy_px": cy,
            "x_m": round(cx * GSD, 4), "y_m": round(cy * GSD, 4),
            "alan_px": apx, "alan_m2": round(apx * GSD * GSD, 6),
        })

crop_area = np.sum(pred_mask == 1) * GSD * GSD
weed_area = np.sum(pred_mask == 2) * GSD * GSD
crop_pct = round(crop_area / area_m2 * 100, 2) if area_m2 > 0 else 0
weed_pct = round(weed_area / area_m2 * 100, 2) if area_m2 > 0 else 0
soil_pct = round(100 - crop_pct - weed_pct, 2)

print(f"Sonuc: Crop={pred_crop} Weed={pred_weed} | Yogunluk: %{weed_pct}")

def overlay(img, mask, alpha=0.5):
    out = img.copy()
    out[mask == 1] = (out[mask == 1] * (1 - alpha) + np.array([0, 200, 0]) * alpha).astype(np.uint8)
    out[mask == 2] = (out[mask == 2] * (1 - alpha) + np.array([220, 50, 50]) * alpha).astype(np.uint8)
    return out

# 1) Segmentasyon gorseli
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
axes[0].imshow(img_rgb)
axes[0].set_title("Orijinal Goruntu", fontsize=12)
axes[0].axis("off")

axes[1].imshow(overlay(img_rgb, pred_mask))
axes[1].set_title(f"Segmentasyon Sonucu\nCrop: {pred_crop} | Weed: {pred_weed}", fontsize=12)
axes[1].axis("off")

# Sadece maske
mask_vis = np.zeros_like(img_rgb)
mask_vis[pred_mask == 1] = [0, 200, 0]
mask_vis[pred_mask == 2] = [220, 50, 50]
axes[2].imshow(mask_vis)
axes[2].set_title(f"Maske\nYesil=Crop Kirmizi=Weed", fontsize=12)
axes[2].axis("off")

plt.suptitle(os.path.basename(IMG_PATH), fontsize=14, fontweight="bold")
plt.tight_layout()
seg_path = os.path.join(OUT_DIR, "segmentasyon.png")
plt.savefig(seg_path, dpi=150, bbox_inches="tight")
plt.close()

# 2) Detay gorseli
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

axes[0].pie([crop_area, weed_area, max(0, area_m2 - crop_area - weed_area)],
            labels=["Crop", "Weed", "Toprak"], autopct="%1.1f%%",
            colors=["#27ae60", "#e74c3c", "#bdc3c7"], startangle=90)
axes[0].set_title(f"Alan Dagilimi\nToplam: {area_m2:.2f} m2", fontsize=12)

# Snif dagilimi
axes[1].bar(["Crop", "Weed"], [pred_crop, pred_weed], color=["#27ae60", "#e74c3c"], edgecolor="black")
for i, v in enumerate([pred_crop, pred_weed]):
    axes[1].text(i, v + 0.5, str(v), ha="center", fontweight="bold", fontsize=14)
axes[1].set_title("Tespit Sayisi", fontsize=12)
axes[1].set_ylabel("Adet")

# Metrik ozet
axes[2].axis("off")
txt = (
    f"Goruntu: {os.path.basename(IMG_PATH)}\n"
    f"Boyut: {w}x{h} px\n"
    f"Alan: {area_m2:.2f} m2\n"
    f"GSD: {args.gsd:.1f} mm/px\n\n"
    f"Crop:  {pred_crop} adet | {crop_area:.4f} m2 | %{crop_pct}\n"
    f"Weed:  {pred_weed} adet | {weed_area:.4f} m2 | %{weed_pct}\n"
    f"Toprak: %{soil_pct}\n\n"
    f"Toplam tespit: {pred_crop + pred_weed}\n"
    f"Confidence: {CONF}\n"
    f"Model: YOLOv8l-seg"
)
axes[2].text(0.05, 0.95, txt, transform=axes[2].transAxes, fontsize=11, va="top",
             fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="#ecf0f1"))

plt.suptitle("Analiz Detaylari", fontsize=14, fontweight="bold")
plt.tight_layout()
detay_path = os.path.join(OUT_DIR, "detay.png")
plt.savefig(detay_path, dpi=150, bbox_inches="tight")
plt.close()

# 3) Koordinat haritasi
if detections:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_rgb, alpha=0.6)
    for d in detections:
        color = "#27ae60" if d["sinif"] == "crop" else "#e74c3c"
        ax.plot(d["cx_px"], d["cy_px"], 'o', color=color, markersize=6, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_title(f"Tespit Koordinatlari ({len(detections)} nesne)", fontsize=13)
    ax.legend(handles=[
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#27ae60', markersize=10, label=f'Crop ({pred_crop})'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label=f'Weed ({pred_weed})')
    ], loc="upper right", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    koord_path = os.path.join(OUT_DIR, "koordinatlar.png")
    plt.savefig(koord_path, dpi=150, bbox_inches="tight")
    plt.close()

# JSON kaydet
with open(os.path.join(OUT_DIR, "sonuc.json"), "w") as f:
    json.dump({
        "gorsel": os.path.basename(IMG_PATH),
        "boyut": f"{w}x{h}", "alan_m2": round(area_m2, 4),
        "gsd_mm": args.gsd, "conf": CONF,
        "crop_sayi": pred_crop, "weed_sayi": pred_weed,
        "crop_alan_m2": round(crop_area, 4), "weed_alan_m2": round(weed_area, 4),
        "crop_yuzde": crop_pct, "weed_yuzde": weed_pct,
        "tespitler": detections
    }, f, indent=2, ensure_ascii=False)

# PDF Rapor
class Rapor(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "DRONEQUBE Gorev 4 - Tekil Goruntu Analiz Raporu", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(2)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Sayfa {self.page_no()}", new_x="RIGHT", new_y="TOP", align="C")
    def baslik(self, t):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
    def yaz(self, t):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, t)
        self.ln(2)

pdf = Rapor()

# Sayfa 1: Segmentasyon
pdf.add_page()
pdf.baslik("1. Segmentasyon Sonucu")
pdf.yaz(f"Gorsel: {os.path.basename(IMG_PATH)}\nBoyut: {w}x{h} piksel | Alan: {area_m2:.2f} m2\nGSD: {args.gsd:.1f} mm/px | Model: YOLOv8l-seg | Conf: {CONF}")
if os.path.exists(seg_path):
    pdf.image(seg_path, x=5, w=200)

# Sayfa 2: Detaylar
pdf.add_page()
pdf.baslik("2. Analiz Detaylari")
if os.path.exists(detay_path):
    pdf.image(detay_path, x=5, w=200)
pdf.ln(3)

pdf.set_font("Courier", "B", 9)
pdf.cell(0, 6, "  Tespit Ozet Tablosu", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Courier", "", 8)
pdf.cell(0, 5, "-" * 60, new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"  {'Metrik':<25} {'Deger':>20}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, "-" * 60, new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"  {'Goruntu Alani':<25} {area_m2:>19.2f} m2", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"  {'Crop Sayisi':<25} {pred_crop:>20d}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"  {'Weed Sayisi':<25} {pred_weed:>20d}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"  {'Toplam Tespit':<25} {pred_crop + pred_weed:>20d}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"  {'Crop Alani':<25} {crop_area:>17.4f} m2", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"  {'Weed Alani':<25} {weed_area:>17.4f} m2", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"  {'Crop Orani':<25} {'%' + str(crop_pct):>20}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"  {'Weed Orani (Yogunluk)':<25} {'%' + str(weed_pct):>20}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"  {'Toprak Orani':<25} {'%' + str(soil_pct):>20}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, "-" * 60, new_x="LMARGIN", new_y="NEXT")

# Sayfa 3: Koordinatlar
if detections:
    pdf.add_page()
    pdf.baslik("3. Koordinat Haritasi ve Listesi")
    kp = os.path.join(OUT_DIR, "koordinatlar.png")
    if os.path.exists(kp):
        pdf.image(kp, x=15, w=180)
    pdf.ln(3)

    pdf.set_font("Courier", "B", 8)
    pdf.cell(0, 5, f"  Tespit Koordinatlari ({len(detections)} nesne)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 6.5)
    pdf.cell(0, 4, f"  {'ID':<24} {'Sinif':<6} {'X(px)':>6} {'Y(px)':>6} {'X(m)':>7} {'Y(m)':>7} {'Alan_m2':>10}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "  " + "-" * 74, new_x="LMARGIN", new_y="NEXT")
    for d in detections:
        pdf.cell(0, 4, f"  {d['id']:<24} {d['sinif']:<6} {d['cx_px']:>6} {d['cy_px']:>6} {d['x_m']:>7.3f} {d['y_m']:>7.3f} {d['alan_m2']:>10.5f}", new_x="LMARGIN", new_y="NEXT")
        if pdf.get_y() > 270:
            pdf.add_page()
            pdf.set_font("Courier", "", 6.5)

pdf_path = os.path.join(OUT_DIR, f"Analiz_{img_base}.pdf")
pdf.output(pdf_path)

print(f"\n{'='*50}")
print(f"PDF:    {pdf_path}")
print(f"JSON:   {OUT_DIR}/sonuc.json")
print(f"Gorsel: {OUT_DIR}/")
print(f"{'='*50}")
