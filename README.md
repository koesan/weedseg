# weedseg

Bu depo, İnsansız Hava Aracı (İHA) sistemleri için tarım alanlarındaki yabancı otların (weed) tespit edilmesi ve gerçek dünya koordinatlarında (georeference) konumlandırılması amacıyla geliştirilmiş iki farklı projeyi içermektedir.

Aşağıdaki Google Drive bağlantısından gerekli model ağırlıklarını (`best.pt`) indirip, her modelin kendi klasörüne eklemeniz gerekmektedir:
📥 **[Model Ağırlıklarını İndir (Google Drive)](https://drive.google.com/drive/folders/1nB4qQeIFdyFNywe3V92VN5Hj5yg3jVsz?usp=drive_link)**

## Proje Yapısı ve Farkları

Depo içerisinde iki farklı yaklaşım ve model barındıran iki klasör bulunmaktadır:

### 1. WeedsGalore (YOLOv8l-seg)
Bu proje, mısır tarlalarındaki yabancı otları **Nesne Bölütleme** yöntemiyle tespit eder. 
- **Veri Seti:** WeedsGalore (Farklı büyüme evrelerini içerir).
- **Model:** YOLOv8l-seg.
- **Mantık:** Yabancı otları nesne olarak algılar, poligon çıkarır ve GSD (Yer Örnekleme Mesafesi) kullanarak lokal metre cinsinden (X, Y) alan/yoğunluk hesabı yapar.
- **Kullanım Yeri:** Erken büyüme evrelerindeki ayrık otların tespiti için idealdir.
- **WeedsGalore Makalesi (ArXiv):** https://arxiv.org/abs/2502.13103
- **WeedsGalore Veri Seti ve Kod Deposu:** https://github.com/GFZ/weedsgalore


### 2. WeedyRice (DeepLabV3+)
Bu ek çalışma, yoğun bitki örtüsünde nesne sayımının yarattığı hataları (over-count) önlemek amacıyla **Semantik Segmentasyon (Alan Bazlı)** olarak geliştirilmiştir.
- **Veri Seti:** WeedyRice-RGBMS-DB.
- **Model:** DeepLabV3+
- **Mantık:** Bitkileri tek tek saymaz, yalnızca "yabancı ot alanını" ölçer. Metadata dosyasındaki gerçek GPS verilerini kullanarak tespitleri harita üzerinde küresel koordinatlarla (WGS84 Enlem/Boylam) konumlandırır.
- **Kullanım Yeri:** Bitkilerin birbirine geçtiği yoğun tarım arazilerinde, gerçek dünya koordinatlarıyla interaktif haritalama (Leaflet) yapmak için idealdir.
- **Weedy Rice Makalesi:** https://www.sciencedirect.com/science/article/pii/S2352340925009588
- **Weedsrice Veri Seti:** https://data.mendeley.com/datasets/vt4s83pxx6/1

## Kurulum

Projeyi çalıştırmak için Python 3.8+ ortamında aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

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

### WeedyRice (DeepLabV3+) Kullanımı
1. İndirdiğiniz DeepLabV3+ `best.pt` dosyasını `WeedyRice` klasörünün içine atın.
2. Terminalden klasöre girin: `cd WeedyRice`
3. **Test ve Haritalama:** `python test.py` (Yabancı ot alanlarını ölçer, interaktif HTML haritası ve PDF rapor üretir).
4. **Tekil Haritalama:** `python test_single.py yol/resim.JPG` (Tek görüntü üzerinden semantik alan çıkarır ve harita üretir).
