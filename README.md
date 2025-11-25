````markdown
# VideoSystem

**VideoSystem**, ROS 2 (**Jazzy Jalisco**) tabanlı, modüler bir görüntü işleme ve olay algılama sistemidir.  
Sistem, kamera akışından aldığı görüntüler üzerinde:

- **YOLOv8 (ONNX Runtime)** ile nesne tespiti,
- **ByteTrack** ile çoklu nesne takibi (ID bazlı takip),
- (İsteğe bağlı) **homografi** ile görüntü koordinatlarından dünya düzlemine izdüşüm

yapar ve sonuçları ROS 2 mesajları olarak yayımlar.

---

## 1. Örnek Çıktı

<p align="center">
  <img src="VideoSystem/docs/img/resim.png" alt="VideoSystem YOLOv8 + ByteTrack demo" width="400">
</p>

---

## 2. Özellikler

- **YOLOv8n (ONNX)**  
  - ONNX Runtime ile çalışır, OpenCV-DNN yerine daha stabil ve hızlı inference.
  - COCO sınıf etiketleri (örn. `person`, `bed`, `chair`) desteklenir.

- **ByteTrack tabanlı takip**  
  - Her tespit için mümkün olduğunda **kararlı track ID** üretir.
  - `/events` mesajında (mesaj tipine göre) `track_id` alanı doldurulabilir.

- **Homografi desteği (opsiyonel)**  
  - Kalibre edilmiş ortamlar için bounding-box merkezini dünya düzlemine (ör. zemin planı) izdüşüm.
  - Homografi matrisi `*.npy` dosyasından okunur.

- **ROS 2 ile tam entegrasyon**
  - `video_system_interfaces` → `Event.msg`
  - `video_system_perception` → YOLOv8 + ByteTrack node’u
  - `video_system_core` → olayları tüketen örnek node’lar
  - `video_system_bringup` → launch / param dosyaları

---

## 3. Depo Yapısı

```text
VideoSystem/
├─ build/                      # colcon build çıktıları
├─ install/                    # colcon install çıktıları
├─ log/                        # colcon logları
├─ docs/
│  └─ img/
│     └─ resim.png             # örnek ekran görüntüsü
├─ scripts/                    # yardımcı betikler (ileride)
├─ src/
│  ├─ video_system_interfaces/
│  │  └─ msg/Event.msg         # olay mesajı (label, score, bbox, vb.)
│  ├─ video_system_core/
│  ├─ video_system_bringup/
│  └─ video_system_perception/
│     ├─ package.xml
│     ├─ CMakeLists.txt
│     ├─ models/
│     │  └─ yolov8n.onnx       # YOLOv8n ONNX modeli
│     ├─ config/
│     │  └─ homography_room1.npy (opsiyonel)
│     └─ scripts/
│        └─ detector_node_py.py
├─ README.md
└─ yolov8n.pt                  # Orijinal PyTorch ağırlıkları (referans)
````

> Not: ROS 2 çalışma alanı **doğrudan `VideoSystem` kök dizinidir** (`src/` altındaki paketler).

---

## 4. Gereksinimler

* **İşletim Sistemi**

  * Ubuntu 22.04 / 24.04 (64-bit)

* **ROS 2**

  * **ROS 2 Jazzy Jalisco** (desktop kurulumu)

* **Sistem paketleri**

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-image-tools \
  python3-colcon-common-extensions \
  python3-opencv \
  python3-yaml
```

* **Python paketleri**

> Uyum için **NumPy 1.x** kullanılması tavsiye edilir (NumPy 2, bazı ROS paketlerinde henüz desteklenmiyor).

```bash
python3 -m pip install "numpy<2" onnxruntime cjm_byte_track
```

Gerektiğinde bunu bir **virtualenv** içinde de yapabilirsiniz.

---

## 5. Derleme

```bash
cd ~/VideoSystem

# ROS 2 ortamını yükle
source /opt/ros/jazzy/setup.bash

# Workspace kökü VideoSystem olduğu için doğrudan buradan build
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Build tamamlandıktan sonra her yeni terminalde:

```bash
cd ~/VideoSystem
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

---

## 6. Çalıştırma Akışı

Üç terminalde çalıştırmak önerilir.

### Terminal A — Kamera yayıncısı

```bash
cd ~/VideoSystem
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run image_tools cam2image \
  --ros-args -p video_device:=/dev/video0 \
  -r /image:=/camera/image_raw
```

> Gerekirse `video_device` parametresini farklı bir kamera için (`/dev/video2` vb.) güncelleyin.

---

### Terminal B — Dedektör (YOLOv8 + ONNX Runtime + ByteTrack)

```bash
cd ~/VideoSystem
source /opt/ros/jazzy/setup.bash
source install/setup.bash

./install/video_system_perception/lib/video_system_perception/detector_node_py.py \
  --ros-args \
  -p model_path:="$HOME/VideoSystem/src/video_system_perception/models/yolov8n.onnx" \
  -p visualize:=true \
  -p conf_threshold:=0.25 \
  -p frame_rate:=15.0 \
  -p use_homography:=false
```

Eğer yürütme izni hatası alırsanız:

```bash
chmod +x ./install/video_system_perception/lib/video_system_perception/detector_node_py.py
```

**Önemli parametreler:**

* `model_path` – ONNX model dosyasının tam yolu.
* `visualize` – `true` ise OpenCV penceresinde önizleme açar.
* `conf_threshold` – sınıflandırma skor eşiği.
* `nms_threshold` – NMS için IoU eşiği (kod içinde varsayılan `0.5`).
* `frame_rate` – ByteTrack için FPS tahmini (ID kararlılığı için önemli).
* `use_homography` – `true` ise `homography_path`’ten 3×3 matris okunur.

---

### Terminal C — Olayları izleme

```bash
cd ~/VideoSystem
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 topic echo /events
```

Örnek `/events` çıktısı (mesaj tanımına bağlı olarak alanlar değişebilir):

```text
stamp:
  sec: 1716183986
  nanosec: 695550316
label: person
score: 0.87
x: 34
y: 0
w: 231
h: 205
track_id: 29          # (Event.msg bu alanı içeriyorsa)
world_x: 1.23         # (use_homography=true ve Event.msg destekliyorsa)
world_y: 2.34
```

---

## 7. `detector_node_py.py` Özeti

* YOLOv8n ONNX modeli ile inference yapar (640×640 input).
* Çıktıyı post-process ederek COCO sınıflarını ve bounding-box’ları üretir.
* OpenCV `NMSBoxes` ile NMS uygular.
* NMS sonrası bbox’ları **ByteTrack**’e verir:

  * Giriş: `[x1, y1, x2, y2, score]`
  * Çıkış: aktif track’ler ve ID’ler
* Her tespit için bir `Event` mesajı yayınlar:

  * `label`, `score`, `x, y, w, h`
  * `track_id` (varsa)
  * Homografi etkinse `world_x`, `world_y` (varsa)

---

## 8. Yol Haritası

* `/detections` ve `/tracks` için ayrı mesaj tipleri (örn. `DetectionArray`, `TrackArray`).
* Olay özetleme ve SQLite tabanlı event logging.
* Behaviour Tree tabanlı karar katmanı (ROS 2 ile entegrasyon).
* ARM cihazlara (Raspberry Pi 5 vb.) yönelik hafif profil ve Docker imajları.
* CI/CD (GitHub Actions ile `colcon build`, `ament_lint`, testler).

---

## 9. Katkı ve İletişim

Pull request ve issue’lar memnuniyetle karşılanır.

**Geliştiriciler:**

* **Ulaş S.** – GitHub: [ukis666](https://github.com/ukis666)
* **Göktürk C.** – GitHub: [GokturkCan](https://github.com/GokturkCan)

```
::contentReference[oaicite:0]{index=0}
```
