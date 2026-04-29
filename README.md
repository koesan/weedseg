# DRONEQUBE Görev 4 — Yabancı Ot Tespiti ve Konumlandırma

Mısır tarlalarındaki yabancı otların **YOLOv8l-seg** modeli ile instance segmentation yöntemiyle tespiti, sayımı, alan ölçümü ve konumlandırılması.

> **Veri Seti:** [WeedsGalore](https://zenodo.org/records/13628857) (Celikkan et al., WACV 2025)  
> **Model:** YOLOv8l-seg (45.9M parametre)  
> **GSD:** 2.5 mm/px — 5 metre uçuş yüksekliği

---

## Özellikler

- Kültür bitkisi (mısır) ve yabancı ot instance segmentation
- GSD tabanlı piksel → metre koordinat dönüşümü
- Alan ölçümü (m²) ve yoğunluk analizi (%)
- Tarihsel bitki büyümesi ve yabancı ot yayılım grafiği
- Her görüntü için GT vs tahmin karşılaştırma raporu
- Otomatik PDF rapor üretimi
- Tekil görüntü analizi (etiket gerektirmez)

---

## Dosya Yapısı

```
DRONEQUBE_Gorev4/
│
├── train.py            # Model eğitimi (WeedsGalore → YOLO dönüşüm + eğitim)
├── test.py             # Toplu test — 26 görüntü, PDF rapor, CSV, JSON
├── test_single.py      # Tekil görüntü analizi (etiketsiz, herhangi bir görsel)
├── best.pt             # Eğitilmiş YOLOv8l-seg ağırlıkları
│
├── yolo_dataset/       # Test veri seti
│   ├── data.yaml
│   ├── images/test/    # 26 test görseli (600×600 px)
│   └── labels/test/    # YOLO formatında etiketler
│
├── sonuclar/           # Çıktı dizini (otomatik oluşur)
│   ├── Gorev4_Test_Raporu.pdf
│   ├── gorseller/
│   ├── goruntu_ozet.csv
│   ├── tespit_detay.csv
│   ├── tarihsel_ozet.csv
│   ├── parsel_yogunluk.csv
│   ├── sonuclar.json
│   └── tekil_<isim>/   # test_single.py çıktıları
│
└── README.md
```

---

## Kurulum

```bash
# Repoyu klonla
git clone https://github.com/<kullanici>/DRONEQUBE_Gorev4.git
cd DRONEQUBE_Gorev4

# Gerekli kütüphaneleri kur
pip install ultralytics fpdf2 opencv-python matplotlib numpy
```

> **Not:** `best.pt` dosyası büyük olduğundan Git LFS kullanmanız önerilir.

---

## Kullanım

### 1. Toplu Test (26 görüntü)

Test setindeki tüm görüntüleri analiz eder, PDF rapor ve CSV çıktıları üretir:

```bash
python test.py
```

**Çıktılar** (`sonuclar/` klasörüne kaydedilir):

| Dosya | Açıklama |
|-------|----------|
| `Gorev4_Test_Raporu.pdf` | Kapsamlı PDF rapor (genel özet + tarihsel analiz + per-image detay) |
| `gorseller/` | Her görüntü için 4'lü karşılaştırma görseli |
| `goruntu_ozet.csv` | Görüntü bazlı metrikler (IoU, sayım, alan) |
| `tespit_detay.csv` | Her tespite ait koordinat ve alan bilgisi |
| `tarihsel_ozet.csv` | Tarih bazlı büyüme istatistikleri |
| `parsel_yogunluk.csv` | Parsel yoğunluk özeti |
| `sonuclar.json` | Tüm verilerin JSON formatı |

### 2. Tekil Görüntü Analizi

Herhangi bir görseli model ile analiz eder. Etiket gerektirmez:

```bash
# Basit kullanım
python test_single.py gorsel.png

# Opsiyonel parametreler
python test_single.py gorsel.jpg --conf 0.20 --gsd 3.0
```

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `image` | — | Analiz edilecek görsel yolu (zorunlu) |
| `--conf` | 0.15 | Confidence eşiği |
| `--gsd` | 2.5 | GSD değeri (mm/px) |
| `--model` | best.pt | Model ağırlık dosyası |

**Çıktılar** (`sonuclar/tekil_<isim>/` klasörüne kaydedilir):

| Dosya | Açıklama |
|-------|----------|
| `Analiz_<isim>.pdf` | 3 sayfalık detaylı PDF rapor |
| `segmentasyon.png` | Orijinal / Overlay / Maske görseli |
| `detay.png` | Alan dağılımı pasta grafiği + sayım |
| `koordinatlar.png` | Tespit noktaları harita üzerinde |
| `sonuc.json` | Tüm tespitler ve metrikler |

### 3. Model Eğitimi

WeedsGalore veri setini indirip sıfırdan model eğitmek için:

```bash
python train.py
```

> Veri seti ZIP dosyasının yolunu ve çıktı dizinini script soracaktır.  
> Eğitim süresi GPU'ya bağlı olarak ~30-60 dakika sürer.

---

## Sınıflar

| ID | Sınıf | Açıklama |
|----|-------|----------|
| 0 | `crop` | Mısır (*Zea mays*) |
| 1 | `weed` | Yabancı otlar (tilki kuyruğu, darıcan, horozibiği vb.) |

---

## Veri Seti

| Özellik | Değer |
|---------|-------|
| Kaynak | WeedsGalore (Potsdam, Almanya — 2023) |
| Çözünürlük | 600×600 px RGB |
| GSD | 2.5 mm/piksel |
| Split | Train: 104 / Val: 26 / Test: 26 |
| Toplam instance | ~12,200 (crop: 2,169 + weed: 10,031) |

---

## Teknik Detaylar

- **Optimizer:** AdamW
- **Görüntü boyutu:** 1024 px
- **Veri artırma:** Mosaic, CopyPaste, MixUp
- **Confidence eşiği:** 0.15
- **Koordinat dönüşümü:** `piksel × GSD (0.0025) = metre`
- **Alan hesabı:** `piksel_sayısı × GSD² = m²`

---

## Sonuçlar

| Metrik | Değer |
|--------|-------|
| Crop IoU | 0.5857 |
| Weed IoU | 0.4189 |
| mIoU | 0.5023 |
| Ortalama eşleşme | %77.6 |

---

## Referanslar

```bibtex
@InProceedings{Celikkan_2025_WACV,
    author    = {Celikkan, Ekin and Kunzmann, Timo and others},
    title     = {WeedsGalore: A Multispectral and Multitemporal UAV-Based Dataset for Crop-Weed Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year      = {2025},
    pages     = {4767--4777}
}
```

---

## Lisans

Bu proje DRONEQUBE yarışması kapsamında geliştirilmiştir.
