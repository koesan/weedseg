"""
DRONEQUBE Gorev 4 — Yabanci Ot Tespiti Test Raporu
YOLOv8l-seg modelini test setinde degerlendirir, PDF rapor uretir.
"""

import os, csv, json
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from fpdf import FPDF

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
YOLO_DIR   = os.path.join(BASE_DIR, "yolo_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "sonuclar")
IMG_DIR    = os.path.join(OUTPUT_DIR, "gorseller")
os.makedirs(IMG_DIR, exist_ok=True)

TEST_IMG_DIR = os.path.join(YOLO_DIR, "images", "test")
TEST_LBL_DIR = os.path.join(YOLO_DIR, "labels", "test")

# Sabitler
GSD      = 0.0025   # metre/piksel (5m ucus, 2.5mm/px)
CONF     = 0.15
IMG_SIZE = 1024
CLASS_NAMES = {0: "crop", 1: "weed"}

assert os.path.exists(MODEL_PATH), f"Model bulunamadi: {MODEL_PATH}"
assert os.path.isdir(TEST_IMG_DIR), f"Test dizini bulunamadi: {TEST_IMG_DIR}"

model = YOLO(MODEL_PATH)
test_images = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".png")])
print(f"Model: {MODEL_PATH}\nTest: {len(test_images)} goruntu\nCikti: {OUTPUT_DIR}\n")

def overlay(img, mask, alpha=0.5):
    out = img.copy()
    out[mask == 1] = (out[mask == 1] * (1-alpha) + np.array([0, 200, 0]) * alpha).astype(np.uint8)
    out[mask == 2] = (out[mask == 2] * (1-alpha) + np.array([220, 50, 50]) * alpha).astype(np.uint8)
    return out


def mask_centroid(binary_mask, fallback_xy):
    moments = cv2.moments(binary_mask, binaryImage=True)
    if moments["m00"] > 0:
        cx = int(round(moments["m10"] / moments["m00"]))
        cy = int(round(moments["m01"] / moments["m00"]))
        return cx, cy
    return fallback_xy

# Goruntu bazli analiz
results = []

for idx, img_name in enumerate(test_images):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    lbl_path = os.path.join(TEST_LBL_DIR, img_name.replace(".png", ".txt"))
    img_rgb  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w     = img_rgb.shape[:2]
    area_m2  = h * w * GSD * GSD

    # Ground truth
    gt_crop, gt_weed = 0, 0
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))
                pts = np.array([[int(coords[j]*w), int(coords[j+1]*h)] for j in range(0, len(coords), 2)], np.int32)
                cv2.fillPoly(gt_mask, [pts], cls + 1)
                if cls == 0: gt_crop += 1
                else: gt_weed += 1

    # Model tahmini
    preds = model.predict(img_path, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
    pred_crop, pred_weed = 0, 0
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    detections = []

    if preds.masks is not None:
        for det_idx, (box, md) in enumerate(zip(preds.boxes, preds.masks), start=1):
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            m = cv2.resize(md.data[0].cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            pred_mask[m == 1] = cls + 1
            area_px = int(m.sum())
            box_cx, box_cy = int(box.xywh[0][0]), int(box.xywh[0][1])
            cx, cy = mask_centroid(m, (box_cx, box_cy))
            if cls == 0: pred_crop += 1
            else: pred_weed += 1
            det_id = f"{os.path.splitext(img_name)[0]}_{CLASS_NAMES[cls]}_{det_idx:03d}"
            detections.append({
                "id": det_id,
                "image": img_name, "class": CLASS_NAMES[cls], "confidence": round(conf, 3),
                "cx_px": cx, "cy_px": cy,
                "local_x_m": round(cx * GSD, 4), "local_y_m": round(cy * GSD, 4),
                "area_px": area_px, "area_m2": round(area_px * GSD * GSD, 6),
            })

    iou = {}
    for v, n in [(1, "crop"), (2, "weed")]:
        inter = np.logical_and(gt_mask == v, pred_mask == v).sum()
        union = np.logical_or(gt_mask == v, pred_mask == v).sum()
        iou[n] = round(inter / union, 4) if union > 0 else 0.0

    gt_crop_area  = np.sum(gt_mask == 1) * GSD * GSD
    gt_weed_area  = np.sum(gt_mask == 2) * GSD * GSD
    pred_crop_area = np.sum(pred_mask == 1) * GSD * GSD
    pred_weed_area = np.sum(pred_mask == 2) * GSD * GSD
    density = round(pred_weed_area / area_m2 * 100, 2) if area_m2 > 0 else 0

    results.append({
        "image": img_name,
        "gt_crop": gt_crop, "gt_weed": gt_weed,
        "pred_crop": pred_crop, "pred_weed": pred_weed,
        "img_area_m2": round(area_m2, 4),
        "gt_crop_area_m2": round(gt_crop_area, 4), "gt_weed_area_m2": round(gt_weed_area, 4),
        "pred_crop_area_m2": round(pred_crop_area, 4), "pred_weed_area_m2": round(pred_weed_area, 4),
        "crop_iou": iou["crop"], "weed_iou": iou["weed"],
        "weed_density_pct": density, "detections": detections,
    })

    # 4'lu gorsel
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(img_rgb); axes[0].set_title("Orijinal", fontsize=10); axes[0].axis("off")
    axes[1].imshow(overlay(img_rgb, gt_mask)); axes[1].set_title(f"Ground Truth\nCrop:{gt_crop} Weed:{gt_weed}", fontsize=10); axes[1].axis("off")
    axes[2].imshow(overlay(img_rgb, pred_mask)); axes[2].set_title(f"Tahmin\nCrop:{pred_crop} Weed:{pred_weed}", fontsize=10); axes[2].axis("off")

    diff = np.zeros_like(img_rgb)
    diff[np.logical_and(gt_mask > 0, gt_mask == pred_mask)] = [0, 200, 0]
    diff[np.logical_and(gt_mask > 0, gt_mask != pred_mask)] = [220, 50, 50]
    diff[np.logical_and(gt_mask == 0, pred_mask > 0)]       = [255, 165, 0]
    miou = np.mean(list(iou.values()))
    axes[3].imshow(diff); axes[3].set_title(f"Hata Haritasi\nYesil=Dogru Kirmizi=Miss Turuncu=FP\nmIoU: {miou:.1%}", fontsize=9); axes[3].axis("off")
    plt.suptitle(f"{img_name} | {area_m2:.2f} m2 | Yogunluk: {density:.1f}%", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, f"analiz_{idx:02d}.png"), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [{idx+1}/{len(test_images)}] {img_name}: GT(c={gt_crop},w={gt_weed}) Pred(c={pred_crop},w={pred_weed}) mIoU={miou:.2f}")

# Toplu istatistikler
S = lambda key: sum(r[key] for r in results)
total_area = S("img_area_m2")
mean_crop_iou = np.mean([r["crop_iou"] for r in results])
mean_weed_iou = np.mean([r["weed_iou"] for r in results])
mean_miou     = (mean_crop_iou + mean_weed_iou) / 2
crop_pct = S("pred_crop_area_m2") / total_area * 100
weed_pct = S("pred_weed_area_m2") / total_area * 100

# Ozet grafik
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
x = np.arange(len(results))

axes[0][0].bar(x-0.15, [r["gt_weed"] for r in results], 0.3, label="GT", color="#c0392b", alpha=0.7)
axes[0][0].bar(x+0.15, [r["pred_weed"] for r in results], 0.3, label="Pred", color="#e74c3c", alpha=0.7)
axes[0][0].set_title("Weed Sayisi: GT vs Tahmin"); axes[0][0].legend()

axes[0][1].bar(x-0.15, [r["crop_iou"] for r in results], 0.3, label="Crop IoU", color="#27ae60")
axes[0][1].bar(x+0.15, [r["weed_iou"] for r in results], 0.3, label="Weed IoU", color="#e74c3c")
axes[0][1].axhline(y=mean_miou, color="navy", ls="--", label=f"Ort: {mean_miou:.2f}")
axes[0][1].set_title("IoU Dagilimi"); axes[0][1].legend()

dens = [r["weed_density_pct"] for r in results]
max_d = max(max(dens), 1)
axes[0][2].bar(x, dens, color=plt.cm.RdYlGn_r(np.array(dens)/max_d))
axes[0][2].set_title("Yabanci Ot Yogunlugu (%)")

bg = total_area - S("pred_crop_area_m2") - S("pred_weed_area_m2")
axes[1][0].pie([S("pred_crop_area_m2"), S("pred_weed_area_m2"), max(0, bg)],
               labels=["Crop","Weed","Toprak"], autopct="%1.1f%%", colors=["#27ae60","#e74c3c","#bdc3c7"])
axes[1][0].set_title(f"Alan Dagilimi ({total_area:.1f} m2)")

cats = ["Crop\nSayi", "Weed\nSayi", "Crop\nAlan", "Weed\nAlan"]
gv = [S("gt_crop"), S("gt_weed"), S("gt_crop_area_m2"), S("gt_weed_area_m2")]
pv = [S("pred_crop"), S("pred_weed"), S("pred_crop_area_m2"), S("pred_weed_area_m2")]
bx = np.arange(4)
axes[1][1].bar(bx-0.15, gv, 0.3, label="GT", color="#2c3e50")
axes[1][1].bar(bx+0.15, pv, 0.3, label="Pred", color="#3498db")
axes[1][1].set_xticks(bx); axes[1][1].set_xticklabels(cats); axes[1][1].legend()
axes[1][1].set_title("GT vs Tahmin Toplam")

axes[1][2].axis("off")
txt = (f"Goruntu: {len(results)} | Alan: {total_area:.2f} m2\n\n"
       f"GT  Crop: {S('gt_crop')} adet | {S('gt_crop_area_m2'):.3f} m2\n"
       f"GT  Weed: {S('gt_weed')} adet | {S('gt_weed_area_m2'):.3f} m2\n\n"
       f"Pred Crop: {S('pred_crop')} adet | {S('pred_crop_area_m2'):.3f} m2\n"
       f"Pred Weed: {S('pred_weed')} adet | {S('pred_weed_area_m2'):.3f} m2\n\n"
       f"Crop: %{crop_pct:.1f} | Weed: %{weed_pct:.1f}\n\n"
       f"Crop IoU: {mean_crop_iou:.4f}\nWeed IoU: {mean_weed_iou:.4f}\nmIoU: {mean_miou:.4f}\n\n"
       f"GSD: {GSD*1000:.1f} mm/px")
axes[1][2].text(0.1, 0.95, txt, transform=axes[1][2].transAxes, fontsize=12, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#ecf0f1"))
plt.suptitle("DRONEQUBE Gorev 4 - Genel Ozet", fontsize=15, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, "genel_ozet.png"), dpi=200, bbox_inches="tight"); plt.close()

# CSV
all_dets = [d for r in results for d in r["detections"]]
csv_det = os.path.join(OUTPUT_DIR, "tespit_detay.csv")
if all_dets:
    with open(csv_det, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_dets[0].keys()); w.writeheader(); w.writerows(all_dets)

csv_ozet = os.path.join(OUTPUT_DIR, "goruntu_ozet.csv")
fields = ["image","gt_crop","gt_weed","pred_crop","pred_weed","img_area_m2",
          "gt_crop_area_m2","gt_weed_area_m2","pred_crop_area_m2","pred_weed_area_m2",
          "crop_iou","weed_iou","weed_density_pct"]
with open(csv_ozet, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in results: w.writerow({k: r[k] for k in fields})

# Parsel yogunluk ozeti
parsel_csv = os.path.join(OUTPUT_DIR, "parsel_yogunluk.csv")
with open(parsel_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tarih","goruntu_sayisi","toplam_alan_m2","crop_alan_m2","weed_alan_m2","weed_yogunluk_pct","ort_weed_sayisi","not"])
    dates = sorted(set(r["image"][:10] for r in results))
    for d in dates:
        dr = [r for r in results if r["image"].startswith(d)]
        w.writerow([d, len(dr),
                     round(sum(r["img_area_m2"] for r in dr), 3),
                     round(sum(r["pred_crop_area_m2"] for r in dr), 4),
                     round(sum(r["pred_weed_area_m2"] for r in dr), 4),
                     round(sum(r["pred_weed_area_m2"] for r in dr) / sum(r["img_area_m2"] for r in dr) * 100, 2),
                     round(np.mean([r["pred_weed"] for r in dr]), 1),
                     "Tarih bazli goruntu ozetidir, gercek dunya parsel georeference ozeti degildir."])

# JSON
with open(os.path.join(OUTPUT_DIR, "sonuclar.json"), "w") as f:
    json.dump({"summary": {
        "model": "YOLOv8l-seg", "test_count": len(results),
        "total_area_m2": round(total_area, 2), "gsd_mm": GSD*1000,
        "mean_crop_iou": round(mean_crop_iou, 4), "mean_weed_iou": round(mean_weed_iou, 4),
        "mean_miou": round(mean_miou, 4), "crop_pct": round(crop_pct, 1), "weed_pct": round(weed_pct, 1),
        "gt_crop": S("gt_crop"), "gt_weed": S("gt_weed"),
        "pred_crop": S("pred_crop"), "pred_weed": S("pred_weed"),
    }, "per_image": [{k: v for k, v in r.items() if k != "detections"} for r in results]}, f, indent=2)

# PDF
class Rapor(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 10, "DRONEQUBE Gorev 4 - Yabanci Ot Tespiti Raporu", new_x="LMARGIN", new_y="NEXT", align="C"); self.ln(2)
    def footer(self):
        self.set_y(-15); self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Sayfa {self.page_no()}", new_x="RIGHT", new_y="TOP", align="C")
    def baslik(self, t):
        self.set_font("Helvetica", "B", 11); self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT"); self.ln(2)
    def yaz(self, t):
        self.set_font("Helvetica", "", 9); self.multi_cell(0, 5, t.replace("\u2014","-").replace("\u2013","-")); self.ln(2)

pdf = Rapor()
pdf.add_page()

pdf.baslik("1. Proje ve Yontem")
pdf.yaz(f"Model: YOLOv8l-seg (45.9M parametre)\nVeri Seti: WeedsGalore (156 goruntu, crop/weed)\nTest Seti: {len(results)} goruntu\nGSD: {GSD*1000:.1f} mm/px | Ucus: 5m | imgsz: {IMG_SIZE}px | conf: {CONF}\nKoordinat Donusumu: piksel x {GSD} = metre")

pdf.baslik("2. Genel Sonuclar")
pdf.yaz(f"Toplam Alan: {total_area:.2f} m2\n\nGround Truth:\n  Crop: {S('gt_crop')} adet | {S('gt_crop_area_m2'):.3f} m2\n  Weed: {S('gt_weed')} adet | {S('gt_weed_area_m2'):.3f} m2\n\nTahmin:\n  Crop: {S('pred_crop')} adet | {S('pred_crop_area_m2'):.3f} m2\n  Weed: {S('pred_weed')} adet | {S('pred_weed_area_m2'):.3f} m2\n\nCrop: %{crop_pct:.1f} | Weed: %{weed_pct:.1f} | Diger: %{100-crop_pct-weed_pct:.1f}")

pdf.baslik("3. Basari Metrikleri")
pdf.yaz(f"Crop IoU: {mean_crop_iou:.4f}\nWeed IoU: {mean_weed_iou:.4f}\nmIoU: {mean_miou:.4f}\n\nSayim Hatasi:\n  Crop: {S('gt_crop')} vs {S('pred_crop')} ({S('pred_crop')-S('gt_crop'):+d})\n  Weed: {S('gt_weed')} vs {S('pred_weed')} ({S('pred_weed')-S('gt_weed'):+d})")

pdf.baslik("4. Parsel Yogunluk Ozeti")
pdf.set_font("Courier", "", 7)
pdf.cell(0, 4, f"{'Tarih':<12} {'Img':>3} {'Alan_m2':>8} {'Crop_m2':>8} {'Weed_m2':>8} {'Yog%':>6} {'OrtWeed':>7}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 4, "-"*60, new_x="LMARGIN", new_y="NEXT")
dates = sorted(set(r["image"][:10] for r in results))
date_data = []
for d in dates:
    dr = [r for r in results if r["image"].startswith(d)]
    ta = sum(r["img_area_m2"] for r in dr)
    wa = sum(r["pred_weed_area_m2"] for r in dr)
    ca = sum(r["pred_crop_area_m2"] for r in dr)
    date_data.append({"date":d,"n":len(dr),"gt_w":sum(r["gt_weed"] for r in dr),"pr_w":sum(r["pred_weed"] for r in dr),
        "gt_c":sum(r["gt_crop"] for r in dr),"pr_c":sum(r["pred_crop"] for r in dr),
        "ca":round(ca,4),"wa":round(wa,4),"ta":round(ta,2),"wp":round(wa/ta*100,2),"mi":round(np.mean([np.mean([r["crop_iou"],r["weed_iou"]]) for r in dr]),4)})
    pdf.cell(0, 4, f"{d:<12} {len(dr):>3} {ta:>8.2f} {ca:>8.3f} {wa:>8.3f} {wa/ta*100:>6.1f} {np.mean([r['pred_weed'] for r in dr]):>7.1f}", new_x="LMARGIN", new_y="NEXT")
pdf.ln(4)

fig, axes = plt.subplots(2, 3, figsize=(22, 13))
dx = np.arange(len(date_data)); dlbl = [d["date"] for d in date_data]
axes[0][0].bar(dx-0.15, [d["gt_w"] for d in date_data], 0.3, label="GT", color="#c0392b")
axes[0][0].bar(dx+0.15, [d["pr_w"] for d in date_data], 0.3, label="Pred", color="#e74c3c", alpha=0.7)
axes[0][0].plot(dx, [d["gt_w"] for d in date_data], 'o--', color="#8b0000", lw=2)
axes[0][0].set_xticks(dx); axes[0][0].set_xticklabels(dlbl, rotation=20, fontsize=8)
axes[0][0].set_title("Yabanci Ot Sayisi Buyumesi", fontweight="bold"); axes[0][0].legend(); axes[0][0].set_ylabel("Adet")
axes[0][1].bar(dx-0.15, [d["gt_c"] for d in date_data], 0.3, label="GT", color="#1e8449")
axes[0][1].bar(dx+0.15, [d["pr_c"] for d in date_data], 0.3, label="Pred", color="#27ae60", alpha=0.7)
axes[0][1].set_xticks(dx); axes[0][1].set_xticklabels(dlbl, rotation=20, fontsize=8)
axes[0][1].set_title("Kultur Bitkisi Sayisi", fontweight="bold"); axes[0][1].legend(); axes[0][1].set_ylabel("Adet")
axes[0][2].fill_between(dx, [d["ca"] for d in date_data], alpha=0.4, color="#27ae60", label="Crop")
axes[0][2].fill_between(dx, [d["wa"] for d in date_data], alpha=0.4, color="#e74c3c", label="Weed")
axes[0][2].plot(dx, [d["ca"] for d in date_data], 'o-', color="#1e8449", lw=2)
axes[0][2].plot(dx, [d["wa"] for d in date_data], 'o-', color="#c0392b", lw=2)
axes[0][2].set_xticks(dx); axes[0][2].set_xticklabels(dlbl, rotation=20, fontsize=8)
axes[0][2].set_title("Bitki Alani (m2)", fontweight="bold"); axes[0][2].legend(); axes[0][2].set_ylabel("m2")
wps = [d["wp"] for d in date_data]; mxw = max(max(wps),1)
axes[1][0].bar(dx, wps, color=plt.cm.RdYlGn_r(np.array(wps)/mxw), edgecolor="black")
for i,d in enumerate(date_data): axes[1][0].text(i, d["wp"]+0.3, f"%{d['wp']:.1f}", ha="center", fontsize=9, fontweight="bold")
axes[1][0].set_xticks(dx); axes[1][0].set_xticklabels(dlbl, rotation=20, fontsize=8)
axes[1][0].set_title("Yabanci Ot Yogunlugu (%)", fontweight="bold"); axes[1][0].set_ylabel("%")
axes[1][1].plot(dx, [d["mi"] for d in date_data], 's-', color="#2980b9", lw=2.5, ms=10)
for i,d in enumerate(date_data): axes[1][1].text(i, d["mi"]+0.01, f"{d['mi']:.3f}", ha="center", fontsize=9)
axes[1][1].set_xticks(dx); axes[1][1].set_xticklabels(dlbl, rotation=20, fontsize=8)
axes[1][1].set_title("Model Basarisi (mIoU)", fontweight="bold"); axes[1][1].set_ylabel("mIoU"); axes[1][1].set_ylim(0,1); axes[1][1].grid(alpha=0.3)
axes[1][2].axis("off")
tlines = [f"{'Tarih':<12} {'GT_W':>5} {'Pr_W':>5} {'W_m2':>6} {'W%':>5} {'mIoU':>6}", "-"*45]
for d in date_data: tlines.append(f"{d['date']:<12} {d['gt_w']:>5} {d['pr_w']:>5} {d['wa']:>6.3f} {d['wp']:>5.1f} {d['mi']:>6.3f}")
axes[1][2].text(0.05, 0.95, "\n".join(tlines), transform=axes[1][2].transAxes, fontsize=10, va="top", fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="#f9e79f"))
axes[1][2].set_title("Tarihsel Ozet", fontweight="bold")
plt.suptitle("DRONEQUBE Gorev 4 - Tarihsel Bitki Buyumesi", fontsize=15, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, "tarihsel_analiz.png"), dpi=200, bbox_inches="tight"); plt.close()

pdf.baslik("5. Genel Ozet Grafigi")
g = os.path.join(IMG_DIR, "genel_ozet.png")
if os.path.exists(g): pdf.image(g, x=5, w=200)

pdf.add_page()
pdf.baslik("5b. Tarihsel Bitki Buyumesi ve Yabanci Ot Analizi")
pdf.yaz("Asagidaki grafik farkli tarihlerde bitki ve yabanci ot sayisi, alani ve yogunlugunun nasil degistigini gostermektedir.")
tg = os.path.join(IMG_DIR, "tarihsel_analiz.png")
if os.path.exists(tg): pdf.image(tg, x=5, w=200)

# Her goruntu icin detayli sayfa
for idx, r in enumerate(results):
    pdf.add_page()
    pdf.baslik(f"6.{idx+1}. {r['image']}")
    p = os.path.join(IMG_DIR, f"analiz_{idx:02d}.png")
    if os.path.exists(p): pdf.image(p, x=5, w=200)
    pdf.ln(3)
    pdf.set_font("Courier", "B", 8)
    pdf.cell(0, 5, f"  Goruntu Detay Tablosu: {r['image']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 7)
    miou_r = round((r["crop_iou"]+r["weed_iou"])/2, 4)
    gt_total = r["gt_crop"] + r["gt_weed"]
    pr_total = r["pred_crop"] + r["pred_weed"]
    crop_f = r["pred_crop"] - r["gt_crop"]
    weed_f = r["pred_weed"] - r["gt_weed"]
    crop_ah = abs(r["pred_crop_area_m2"]-r["gt_crop_area_m2"])/r["gt_crop_area_m2"]*100 if r["gt_crop_area_m2"]>0 else 0
    weed_ah = abs(r["pred_weed_area_m2"]-r["gt_weed_area_m2"])/r["gt_weed_area_m2"]*100 if r["gt_weed_area_m2"]>0 else 0
    pdf.cell(0, 4, "-"*70, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, f"  {'Metrik':<20} {'Ground Truth':>14} {'Tahmin':>14} {'Fark/Hata':>14}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "-"*70, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, f"  {'Crop Sayisi':<20} {r['gt_crop']:>14d} {r['pred_crop']:>14d} {crop_f:>+14d}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, f"  {'Weed Sayisi':<20} {r['gt_weed']:>14d} {r['pred_weed']:>14d} {weed_f:>+14d}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, f"  {'Toplam':<20} {gt_total:>14d} {pr_total:>14d} {pr_total-gt_total:>+14d}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, f"  {'Crop Alan (m2)':<20} {r['gt_crop_area_m2']:>14.4f} {r['pred_crop_area_m2']:>14.4f} {'%'+str(round(crop_ah,1)):>14}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, f"  {'Weed Alan (m2)':<20} {r['gt_weed_area_m2']:>14.4f} {r['pred_weed_area_m2']:>14.4f} {'%'+str(round(weed_ah,1)):>14}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "-"*70, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, f"  Crop IoU: {r['crop_iou']:.4f}  |  Weed IoU: {r['weed_iou']:.4f}  |  mIoU: {miou_r:.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, f"  Goruntu Alani: {r['img_area_m2']:.2f} m2  |  Yogunluk: %{r['weed_density_pct']:.1f}  |  GSD: 2.5mm/px", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "-"*70, new_x="LMARGIN", new_y="NEXT")
    img_dets = r["detections"]
    if img_dets:
        pdf.ln(2)
        pdf.set_font("Courier", "B", 7)
        pdf.cell(0, 4, f"  Koordinat Listesi (ilk {min(15, len(img_dets))} tespit):", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 6)
        pdf.cell(0, 3.5, f"  {'ID':<18} {'Sinif':<6} {'X(px)':>6} {'Y(px)':>6} {'X(m)':>7} {'Y(m)':>7} {'Alan':>8}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 3.5, "  "+"-"*74, new_x="LMARGIN", new_y="NEXT")
        for det in img_dets[:15]:
            pdf.cell(0, 3.5, f"  {det['id']:<18} {det['class']:<6} {det['cx_px']:>6} {det['cy_px']:>6} {det['local_x_m']:>7.3f} {det['local_y_m']:>7.3f} {det['area_m2']:>8.5f}", new_x="LMARGIN", new_y="NEXT")
        if len(img_dets) > 15:
            pdf.cell(0, 3.5, f"  ... ve {len(img_dets)-15} tespit daha", new_x="LMARGIN", new_y="NEXT")

pdf.add_page()
pdf.baslik("7. Goruntu Ozet Tablosu")
pdf.set_font("Courier", "", 5.5)
hdr = f"{'Goruntu':<22} {'GT_C':>4} {'GT_W':>4} {'Pr_C':>4} {'Pr_W':>4} {'C_IoU':>6} {'W_IoU':>6} {'Yog%':>5} {'m2':>6}"
pdf.cell(0, 3.5, hdr, new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 3.5, "-"*len(hdr), new_x="LMARGIN", new_y="NEXT")
for r in results:
    pdf.cell(0, 3.5, f"{r['image']:<22} {r['gt_crop']:>4} {r['gt_weed']:>4} {r['pred_crop']:>4} {r['pred_weed']:>4} {r['crop_iou']:>6.3f} {r['weed_iou']:>6.3f} {r['weed_density_pct']:>5.1f} {r['img_area_m2']:>6.2f}", new_x="LMARGIN", new_y="NEXT")

pdf.add_page()
pdf.baslik("8. Tum Koordinat ve ID Tablosu (Ilk 80 Tespit)")
pdf.set_font("Courier", "", 5.5)
ch = f"{'ID':<24} {'Sinif':>5} {'X_px':>5} {'Y_px':>5} {'X_m':>6} {'Y_m':>6} {'Alan':>8} {'Conf':>5}"
pdf.cell(0, 3.5, ch, new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 3.5, "-"*len(ch), new_x="LMARGIN", new_y="NEXT")
for d in all_dets[:80]:
    pdf.cell(0, 3.5, f"{d['id']:<24} {d['class']:>5} {d['cx_px']:>5} {d['cy_px']:>5} {d['local_x_m']:>6.3f} {d['local_y_m']:>6.3f} {d['area_m2']:>8.5f} {d['confidence']:>5.2f}", new_x="LMARGIN", new_y="NEXT")

pdf.output(os.path.join(OUTPUT_DIR, "Gorev4_Test_Raporu.pdf"))

print(f"\n{'='*50}")
print(f"PDF:    {OUTPUT_DIR}/Gorev4_Test_Raporu.pdf")
print(f"CSV:    {csv_det}")
print(f"Parsel: {parsel_csv}")
print(f"JSON:   {OUTPUT_DIR}/sonuclar.json")
print(f"Gorsel: {IMG_DIR}/ ({len(test_images)+2} dosya)")
print(f"{'='*50}")
