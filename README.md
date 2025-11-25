
````markdown
---

# VideoSystem

**VideoSystem**, ROS 2 (**Jazzy Jalisco**) tabanlı modüler bir görüntü işleme altyapısıdır.  
Sistem, bilgisayar kamerasından alınan görüntüleri işler ve tespit/olay bilgilerini ROS 2 mesajları olarak yayınlar.

Mevcut mimari:

- **YOLOv8 + ONNX Runtime dedektörü**
- **ByteTrack tabanlı multi-object tracking (MOT)**
- İsteğe bağlı **homografi ile dünya düzlemine izdüşüm** (örn. oda planı)
- Gelecekte özetleme ve olay analizi modülleri ile genişlemeye uygun yapı

---

## 1) Öne Çıkanlar

* **YOLOv8 dedektörü (Python + ONNX Runtime)**  
  OpenCV-DNN yerine resmi ORT kullanır; daha stabil ve taşınabilir.

* **ByteTrack ile takip**  
  Her tespit için kalıcı `track_id` üretir; kişi/nesne takibi ve “ne kadar süredir kadrajda?” gibi sorulara temel oluşturur.

* **Opsiyonel homografi**  
  Bounding box merkezini 3×3 homografi ile dünya düzlemine map eder  
  (`world_x`, `world_y` alanları Event mesajında kullanılabilir).

* **COCO sınıf etiketleri**  
  `/events` üzerinde anlamlı `label` alanı (örn. `person`, `bed`, `teddy bear`).

* **Kolay entegrasyon**  
  Paketler modüler: `interfaces / core / perception / bringup`.

* **Esnek koşum**  
  Dedektör **ros2 run kullanılmadan**, direkt binary üzerinden çalıştırılır.

---

## 2) Klasör Yapısı

Güncel, aktif workspace **deponun kökü**dür:

```text
VideoSystem/
├─ src/
│  ├─ video_system_interfaces/     # Event.msg vb. arayüzler
│  ├─ video_system_core/           # Örnek çekirdek node'lar
│  ├─ video_system_bringup/        # launch + param dosyaları
│  └─ video_system_perception/     # YOLOv8 + ByteTrack dedektörü
│     ├─ scripts/
│     │   └─ detector_node_py.py   # Python + ORT + ByteTrack + homografi
│     ├─ models/
│     │   └─ yolov8n.onnx          # ONNX model
│     └─ config/
│         └─ homography_room1.npy  # (opsiyonel) 3×3 homografi matrisi
├─ docs/
│  └─ img/                         # README görselleri
├─ scripts/                        # Yardımcı betikler
├─ build/                          # colcon build çıktısı (git ile takip edilmez)
├─ install/                        # colcon install (git ile takip edilmez)
├─ log/                            # colcon log (git ile takip edilmez)
├─ ros2_ws/                        # Eski workspace (legacy, artık kullanılmıyor)
├─ README.md
└─ yolov8n.pt                      # Orijinal PyTorch model (isteğe bağlı)
````

> Not: `ros2_ws/` klasörü **sadece geriye dönük referans** için tutulmaktadır.
> Güncel akışta **`src/` altındaki paketler** kullanılmalı, build ve overlay kök dizine göre yapılmalıdır.

---

## 3) Gereksinimler

* Ubuntu 22.04 / 24.04
* ROS 2 **Jazzy Jalisco**
* `colcon` ve ROS demo paketleri
* Python 3.12 (Ubuntu 24.04 varsayılanı)

Temel bağımlılıklar:

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-image-tools \
  python3-colcon-common-extensions \
  python3-opencv \
  python3-yaml
```

**ONNX Runtime + ByteTrack (pip üzerinden):**

> Debian tabanlı sistemlerde PEP 668 nedeniyle `--break-system-packages` gerekebilir.

```bash
python3 -m pip install --break-system-packages \
  'numpy<2' onnxruntime cjm_byte_track
```

---

## 4) Klonlama ve Build

```bash
cd ~
git clone https://github.com/ukis666/VideoSystem.git
cd VideoSystem

source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

Build tamamlandığında `build/`, `install/` ve `log/` dizinleri oluşur.

### Önerilen overlay (tek satır)

Bu depo için tercih edilen overlay sırası:

```bash
source /opt/ros/jazzy/setup.bash && \
VS_PREFIX="$HOME/VideoSystem/install" && \
export AMENT_PREFIX_PATH="$VS_PREFIX:${AMENT_PREFIX_PATH:-}" && \
export AMENT_INDEX_RESOURCE_PATH="$VS_PREFIX/share/ament_index:${AMENT_INDEX_RESOURCE_PATH:-}" && \
[ -d "$VS_PREFIX/lib/python3.12/site-packages" ] && \
export PYTHONPATH="$VS_PREFIX/lib/python3.12/site-packages:${PYTHONPATH:-}"
```

> İpucu: Bu satırı `ros2on` gibi bir alias olarak `.bashrc` içine koymak kullanımı ciddi rahatlatır.

---

## 5) Çalıştırma

Aşağıdaki üç terminal akışı önerilir.
Her terminalde önce **overlay**’i aktifleştirdiğinden emin ol.

### Terminal A — Kamera yayıncısı

```bash
source /opt/ros/jazzy/setup.bash
source ~/VideoSystem/install/setup.bash

ros2 run image_tools cam2image \
  --ros-args -p video_device:=/dev/video0 \
  -r /image:=/camera/image_raw
```

Bu node `/camera/image_raw` topic'ine görüntü yayınlar.

---

### Terminal B — Dedektör (YOLOv8 + ByteTrack + opsiyonel homografi)

Dedektör binary’si bilerek **ros2 run kullanılmadan** doğrudan çalıştırılır.

```bash
source /opt/ros/jazzy/setup.bash
source ~/VideoSystem/install/setup.bash

/home/uki/VideoSystem/install/video_system_perception/lib/video_system_perception/detector_node_py.py \
  --ros-args \
  -p model_path:="/home/uki/VideoSystem/src/video_system_perception/models/yolov8n.onnx" \
  -p visualize:=true \
  -p conf_threshold:=0.25 \
  -p nms_threshold:=0.50 \
  -p frame_rate:=15.0 \
  -p use_homography:=false \
  -p homography_path:="/home/uki/VideoSystem/src/video_system_perception/config/homography_room1.npy"
```

> İzin hatası görürsen:
>
> ```bash
> chmod +x ~/VideoSystem/install/video_system_perception/lib/video_system_perception/detector_node_py.py
> ```

Dedektör:

* `/camera/image_raw`’ı dinler,
* YOLOv8 + ONNX Runtime ile tespit yapar,
* ByteTrack ile takip (`track_id` üretimi),
* İsteğe bağlı homografi ile dünya koordinatı hesaplar,
* Sonuçları `/events` topic'inde **`video_system_interfaces/msg/Event`** mesajı olarak yayınlar,
* İsteğe bağlı olarak OpenCV penceresinde `ID <track_id> | label:score` overlay’i gösterir.

---

### Terminal C — Olayları izle

```bash
source /opt/ros/jazzy/setup.bash
source ~/VideoSystem/install/setup.bash

ros2 topic echo /events
```

Örnek `/events` mesajı (homografi kapalıyken):

```yaml
stamp:
  sec: 1716183986
  nanosec: 695550316
label: person
score: 0.83
x: 34
y: 0
w: 231
h: 205
track_id: 7           # (mesaj tanımına göre opsiyonel alan)
world_x: 0.0          # homografi kapalıysa 0 veya doldurulmamış olabilir
world_y: 0.0
```

---

## 6) Dedektör Parametreleri

`detector_node_py` tarafından kullanılan başlıca parametreler:

| Adı               | Tip    | Varsayılan | Açıklama                                                 |
| ----------------- | ------ | ---------: | -------------------------------------------------------- |
| `model_path`      | string |    zorunlu | YOLOv8 ONNX modelinin tam yolu                           |
| `visualize`       | bool   |     `true` | OpenCV önizleme penceresi açılıp açılmayacağı            |
| `conf_threshold`  | float  |     `0.45` | Skor eşiği (README örneklerinde 0.25 kullanılıyor)       |
| `nms_threshold`   | float  |     `0.50` | NMS (Non-Max Suppression) eşiği                          |
| `frame_rate`      | float  |     `15.0` | ByteTrack için tahmini FPS (takip kararlılığını etkiler) |
| `use_homography`  | bool   |    `false` | `true` ise homografi matrisi kullanılır                  |
| `homography_path` | string |       `""` | 3×3 homografi matrisi içeren `.npy` dosyasının yolu      |

> Not: Homografi açıkken bounding box merkezleri `(cx, cy)` homografi ile `(world_x, world_y)` düzlemine map edilir.
> Bu koordinatlar Event mesajında tanımlıysa doldurulur, ayrıca log’da `"Track X mapped to world coords: (...)"` satırları görülebilir.

---

## 7) Örnek Çalışma (YOLOv8 ORT + ByteTrack + `/events`)

<p align="center">
  <img src="docs/img/yolov8_ort_demo.png" alt="YOLOv8 ORT demo" width="800">
</p>

Üç terminal akışı:

1. `cam2image` → `/camera/image_raw`
2. `detector_node_py.py` → `/events`
3. `ros2 topic echo /events` → Event akışını izle

Görsel pencerede her kutunun üzerinde:

```text
ID <track_id> | <label>:<score>
```

formatında overlay görünür. Aynı ID, aynı nesnenin zaman içinde takip edildiğini ifade eder.

---

## 8) Sık Karşılaşılan Sorunlar

* **`Permission denied` (dedektör):**
  Binary’ye yürütme izni verin:

  ```bash
  chmod +x ~/VideoSystem/install/video_system_perception/lib/video_system_perception/detector_node_py.py
  ```

* **`Package 'video_system_perception' not found` veya benzeri:**
  Overlay’in doğru sırayla yüklendiğinden emin olun (yukarıdaki tek satırlık overlay önerilir).

* **NumPy / `cv_bridge` uyumsuzluğu:**
  `numpy<2` kurulu olmalı:

  ```bash
  python3 -m pip install --break-system-packages 'numpy<2'
  ```

* **Model bulunamadı:**
  `model_path` değerinin tam ve doğru ABSOLUTE path olduğundan emin olun.

* **OpenCV GStreamer uyarıları (`/dev/video0` açılamıyor):**

  * Kamera gerçekten `/dev/video0` mı?
  * Başka bir uygulama kamerayı kilitlemiş olabilir.

---

## 9) Yol Haritası (Roadmap)

* **Üst-seviye olay çıkarımı**

  * Person + baby etkileşimi, belirli süre ve alan koşullarına göre “olay” üretimi.
  * `/events` → `/high_level_events` dönüştürücü düğümler.

* **Gelişmiş takip**

  * ByteTrack yanında isteğe bağlı **OC-SORT** profili.
  * Çoklu ROI/zone takibi ve zone-based alarm (örn. “crib zone”, “door zone”).

* **Kayıt & Analiz**

  * **SQLite** tabanlı `events.db` (zaman damgası, sınıf, skor, bbox, track_id, world_x/y).
  * `rosbag2` ile ham görüntü + `/events` kaydı ve offline oynatma.

* **Çoklu Kamera / Namespace**

  * Birden fazla kamerayı `namespace` ile paralel çalıştırma.
  * Homografi + kamera iç parametreleriyle 2D düzlem üzerinde multi-camera fuse.

* **Yapılandırma**

  * Tüm parametrelerin YAML üzerinden yönetimi (eşik, sınıf filtreleri, yayın oranları).
  * Çalışırken değiştirilebilir parametreler (dynamic parameters).

* **Dağıtım**

  * **Dockerfile** (ros:jazzy tabanlı) ve VS Code devcontainer.
  * Raspberry Pi 5 için optimize edilmiş build profili.

---

## 10) Yazarlar

**ULAS S.**
GitHub: [ukis666](https://github.com/ukis666)

**GOKTURK C.**
GitHub: [GokturkCan](https://github.com/GokturkCan)

---

```
