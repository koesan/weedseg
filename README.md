<<<<<<< HEAD
# Yabancı Ot Tespiti ve Konumlandırma
=======
# DRONEQUBE - Yabancı Ot Tespiti ve Konumlandırma (Görev 4)
>>>>>>> cd5d68b (update crop and weed detection and segmentation project)

Bu depo, DRONEQUBE İnsansız Hava Aracı (İHA) sistemleri için tarım alanlarındaki yabancı otların (weed) tespit edilmesi ve gerçek dünya koordinatlarında (georeference) konumlandırılması amacıyla geliştirilmiş iki farklı projeyi içermektedir.

<<<<<<< HEAD
> **Veri Seti:** [WeedsGalore](https://github.com/GFZ/weedsgalore) (Celikkan et al., WACV 2025)  
> **Model:** YOLOv8l-seg (45.9M parametre)  
> **GSD:** 2.5 mm/px — 5 metre uçuş yüksekliği
=======
Aşağıdaki Google Drive bağlantısından gerekli model ağırlıklarını (`best.pt`) indirip, her modelin kendi klasörüne eklemeniz gerekmektedir:
📥 **[Model Ağırlıklarını İndir (Google Drive)](https://drive.google.com/drive/folders/1nB4qQeIFdyFNywe3V92VN5Hj5yg3jVsz?usp=drive_link)**
>>>>>>> cd5d68b (update crop and weed detection and segmentation project)

## Proje Yapısı ve Farkları

Depo içerisinde iki farklı yaklaşım ve model barındıran iki klasör bulunmaktadır:

### 1. WeedsGalore (YOLOv8l-seg)
Bu proje, mısır tarlalarındaki yabancı otları **Nesne Bölütleme (Instance Segmentation)** yöntemiyle tespit eder. 
- **Veri Seti:** WeedsGalore (Farklı büyüme evrelerini içerir).
- **Model:** YOLOv8l-seg.
- **Mantık:** Yabancı otları nesne olarak algılar, poligon çıkarır ve GSD (Yer Örnekleme Mesafesi) kullanarak lokal metre cinsinden (X, Y) alan/yoğunluk hesabı yapar.
- **Kullanım Yeri:** Erken büyüme evrelerindeki ayrık otların tespiti için idealdir.
- **Kaynaklar:** 
  - **Makale:** [WeedsGalore: A Multispectral and Multitemporal UAV-based Dataset... (ArXiv)](https://arxiv.org/abs/2502.13103)
  - **Veri Seti:** [WeedsGalore GitHub Deposu](https://github.com/GFZ/weedsgalore)

<<<<<<< HEAD
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
│
└── README.md
```

---

## Kurulum

```bash
# Gerekli kütüphaneleri kur
pip install ultralytics fpdf2 opencv-python matplotlib numpy
```

---

## Kullanım

### 1. Toplu Test (26 görüntü)

Test setindeki tüm görüntüleri analiz eder, PDF rapor ve CSV çıktıları üretir:
=======
### 2. WeedyRice (DeepLabV3+)
Bu ek çalışma, yoğun bitki örtüsünde nesne sayımının yarattığı hataları (over-count) önlemek amacıyla **Semantik Segmentasyon (Alan Bazlı)** olarak geliştirilmiştir.
- **Veri Seti:** WeedyRice-RGBMS-DB.
- **Model:** DeepLabV3+
- **Mantık:** Bitkileri tek tek saymaz, yalnızca "yabancı ot alanını" ölçer. Metadata dosyasındaki gerçek GPS verilerini kullanarak tespitleri harita üzerinde küresel koordinatlarla (WGS84 Enlem/Boylam) konumlandırır.
- **Kullanım Yeri:** Bitkilerin birbirine geçtiği yoğun tarım arazilerinde, gerçek dünya koordinatlarıyla interaktif haritalama (Leaflet) yapmak için idealdir.
- **Kaynaklar:** 
  - **Makale:** [A dataset of aligned RGB and multispectral UAV imagery for semantic segmentation of weedy rice (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2352340925009588)
  - **Veri Seti:** [WeedyRice Mendeley Data](https://data.mendeley.com/datasets/vt4s83pxx6/1)

## Kurulum

Projeyi çalıştırmak için Python 3.8+ ortamında aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
>>>>>>> cd5d68b (update crop and weed detection and segmentation project)

```bash
pip install torch torchvision ultralytics opencv-python numpy matplotlib fpdf
```

## Nasıl Kullanılır?

### WeedsGalore (YOLO) Kullanımı
1. İndirdiğiniz YOLOv8 `best.pt` dosyasını `WeedsGalore` klasörünün içine atın.
2. Terminalden klasöre girin: `cd WeedsGalore`
3. **Eğitim (Train):** `python train.py` (Zipli veri setini soracaktır, yolunu belirtin).
4. **Test (Toplu):** `python test.py` (Test klasöründeki görüntüleri analiz eder, PDF ve CSV üretir).
5. **Tekil Analiz:** `python test_single.py yol/resim.JPG` (Tek bir resmi analiz eder).

<<<<<<< HEAD
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
> Eğitim süresi GPU'ya bağlı olarak ~20-60 dakika sürer.

---

## Sınıflar

| ID | Sınıf | Açıklama |
|----|-------|----------|
| 0 | `crop` | Mısır |
| 1 | `weed` | Yabancı otlar |

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

## Sonuçlar

| Metrik | Değer |
|--------|-------|
| Crop IoU | 0.5857 |
| Weed IoU | 0.4189 |
| mIoU | 0.5023 |
| Ortalama eşleşme | %77.6 |

---
=======
### WeedyRice (DeepLabV3+) Kullanımı
1. İndirdiğiniz DeepLabV3+ `best.pt` dosyasını `WeedyRice` klasörünün içine atın.
2. Terminalden klasöre girin: `cd WeedyRice`
3. **Test ve Haritalama:** `python test.py` (Yabancı ot alanlarını ölçer, interaktif HTML haritası ve PDF rapor üretir).
4. **Tekil Haritalama:** `python test_single.py yol/resim.JPG` (Tek görüntü üzerinden semantik alan çıkarır ve harita üretir).
>>>>>>> cd5d68b (update crop and weed detection and segmentation project)
