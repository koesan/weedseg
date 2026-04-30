# DRONEQUBE - LiDAR Obstacle Detection & RTH Simülasyonu

ArduPilot SITL + Gazebo + LiDAR + Derinlik Kamera ile engel algılama ve Return-to-Home simülasyonu.

## Senaryo

1. Drone Home konumunda kalkış yapar (5m irtifa)
2. Baktığı yönde 100m ilerideki B noktasına yönelir
3. LiDAR + Derinlik kamerasıyla engel tespit edildiğinde fren yapar
4. Güvenli şekilde RTL moduna geçip Home'a döner

## Mimari

```
Gazebo Simülasyon
├── Iris Drone + LiDAR + 5 Kamera + Derinlik Kamera
├── Engel Modeli (Kutu)
│
├── ArduPilot SITL (UDP:14550)
│
└── main.py (Companion Computer)
    ├── DroneKit → MAVLink bağlantı
    ├── LiDAR Callback → 360° mesafe ölçümü
    ├── Derinlik Callback → Ön mesafe haritası
    ├── Füzyon → LiDAR + Derinlik engel onayı
    ├── Durum Makinesi → HAREKET / FREN / RTL
    └── OpenCV Panel → 5 kamera + Derinlik + LiDAR radar
```

## Durum Makinesi

```
HAREKET ──(LiDAR veya Derinlik < 10m)──► FREN ──(4sn sonra)──► RTL
   │                                       │                      │
   └── B'ye git (2m/s)                    └── Havada dur         └── Home'a dön
```

## Teknoloji

| Bileşen        | Teknoloji                               |
| -------------- | --------------------------------------- |
| Uçuş Kontrolcü | ArduPilot SITL                          |
| Simülasyon     | Gazebo Classic 11                       |
| Companion      | DroneKit + pymavlink                    |
| Sensör         | 360° LiDAR (`/spur/laser/scan`)         |
| Derinlik       | Ön Depth Camera (`/camera/depth/depth`) |
| Kamera         | 5x ROS Image (`/camera/*/image_raw`)    |
| Görselleştirme | OpenCV Kontrol Paneli                   |
| Ortam          | Docker + VS Code Dev Container          |

## Kurulum

### Docker Ortamı

Daha önce geliştirdiğim [ArduGazeboSim-Docker](https://github.com/koesan/ArduGazeboSim-Docker) repom ile Docker + VS Code ortamında tüm bağımlılıklar hazır şekilde çalışabilirsiniz. Her işletim sisteminde docker içinde ROS + Gazebo geliştirme yapılabilir.

```bash
# Repoyu klonla
git clone https://github.com/koesan/ArduGazeboSim-Docker.git
cd ArduGazeboSim-Docker
code .

# VS Code'da "Reopen in Container" seç
# Container içine girince:
chmod +x setup_simulation.sh
./setup_simulation.sh
```

Detaylı kurulum: [ArduGazeboSim-Docker README](https://github.com/koesan/ArduGazeboSim-Docker)

### Çalıştırma

Depo içindeki model ve world dosyalarını kullanın. 

Terminal 1 - Gazebo:

```bash
source ~/.bashrc
roslaunch iq_sim multi_drone.launch
```

Terminal 2 - ArduPilot SITL:

```bash
cd ~/ardupilot
sim_vehicle.py -v ArduCopter -f gazebo-iris -I0
```

Terminal 3 - Görev:

```bash
python3 main.py
```

## Engel Algılama Mantığı

**LiDAR + Derinlik Füzyonu**: İki sensör birlikte engel onaylar.

| Sensör          | Veri               | Topic                 |
| --------------- | ------------------ | --------------------- |
| LiDAR           | 360° mesafe        | `/spur/laser/scan`    |
| Derinlik Kamera | Ön mesafe haritası | `/camera/depth/depth` |

- LiDAR: 360° tarama, ön alana bakar (Gazeboda 90° sola monte, kodda düzeltiliyor)
- Derinlik: Ön kameranın üst bölgesini tarar (ufuk çizgisi altını filtreliyor)
- İkisi de <10m → **ENGEL (LiDAR+Derinlik)** kesin engel
- Sadece LiDAR <10m → **ENGEL (LiDAR)** fren
- Sadece Derinlik <10m → **ENGEL (Derinlik)** fren
- İkisi de temiz → B'ye devam
- FREN: 4sn hover → RTL
- B noktasına 3m içinde → FREN → RTL

## Kontrol Paneli

OpenCV ile 2x3 grid gösterim:

- Sol / Ön / Sağ Kamera
- Derinlik (JET colormap) / Alt Kamera / LiDAR Radar

ESC ile çıkış.

## Kaynaklar ve Önceki Çalışmalar

- [ArduGazeboSim-Docker](https://github.com/koesan/ArduGazeboSim-Docker) - Docker simülasyon ortamı
- [Python_Dronekit](https://github.com/koesan/Python_Dronekit) - Yangın tespiti ve hassas iniş projeleri
- [TepeGöz](https://github.com/koesan/TepeGoz) - Çoklu drone otonom gözetim sistemi

## Gereksinimler

```
dronekit
pymavlink
opencv-python
numpy
ros-noetic-sensor-msgs
cv-bridge
```
