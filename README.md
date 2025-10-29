

# VideoSystem

**VideoSystem**, ROS 2 (Jazzy Jalisco) tabanlı bir görüntü işleme altyapısıdır.  
Sistem, bilgisayar kamerasından alınan görüntüleri ROS 2 mesajı olarak işler ve belirli periyotlarda olay mesajları (`/events`) yayınlar.  
Bu yapı, ilerleyen aşamalarda YOLO tabanlı nesne tespiti, özetleme ve olay analizi modülleri ile genişletilmeye uygundur.

---

## 1. Genel Mimari

```

ros2_ws/
└── src/
├── video_system_interfaces/   → ROS 2 mesaj tanımı (Event.msg)
├── video_system_core/         → Ana C++ düğümü (ImageEventNode)
└── video_system_bringup/      → Launch ve parametre dosyaları

````

---

## 2. Paketlerin Tanımı

| Paket Adı | Açıklama |
|------------|-----------|
| **video_system_interfaces** | `Event.msg` adlı özel mesaj tanımı içerir. <br> Alanlar: `stamp`, `label`, `score`, `x`, `y`, `w`, `h`. |
| **video_system_core** | `image_event_node_main` adlı C++ ROS 2 düğümünü içerir. <br> `/camera/image_raw` konusuna abone olur ve her belirli sayıda karede bir `/events` konusuna mesaj yayınlar. <br> Parametre: `heartbeat_stride` (varsayılan değer: 15). |
| **video_system_bringup** | `bringup.launch.py` adlı başlatma dosyasını ve `params.yaml` adlı parametre dosyasını içerir. <br> Bu paket, sistemi tek komutla başlatmak için kullanılır. |

---

## 3. Kullanılan Harici Araç

Sistemin test edilmesi sırasında **`image_tools`** adlı ROS 2 demo paketi, gerçek kamera görüntüsünü yayınlamak amacıyla kullanılmıştır.  
Bu paket projeye doğrudan dahil edilmemiştir; yalnızca harici bir görüntü kaynağı olarak kullanılmaktadır.

---

## 4. Kurulum Adımları

### 4.1. Gereksinimler

- Ubuntu 24.04 (veya 22.04)
- ROS 2 Jazzy Jalisco
- `colcon` ve `rosdep` araçları

Kurulum örneği:

```bash
sudo apt install ros-jazzy-desktop ros-jazzy-image-tools python3-colcon-common-extensions
````

### 4.2. Depoyu Klonlama

```bash
git clone git@github.com:ukis666/VideoSystem.git
cd VideoSystem/ros2_ws
```

### 4.3. Derleme

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --merge-install
```

### 4.4. Ortam Değişkenlerinin Ayarlanması

Proje, özel bir overlay sıralamasıyla çalışır:

```bash
source /opt/ros/jazzy/setup.bash
VS_PREFIX="$HOME/VideoSystem/ros2_ws/install"
export AMENT_PREFIX_PATH="$VS_PREFIX:${AMENT_PREFIX_PATH:-}"
export AMENT_INDEX_RESOURCE_PATH="$VS_PREFIX/share/ament_index:${AMENT_INDEX_RESOURCE_PATH:-}"
[ -d "$VS_PREFIX/lib/python3.12/site-packages" ] && \
export PYTHONPATH="$VS_PREFIX/lib/python3.12/site-packages:${PYTHONPATH:-}"
```

---

## 5. Çalıştırma

### 5.1. Kamera Yayını Başlatma

```bash
ros2 run image_tools cam2image \
  --ros-args -p video_device:=/dev/video0 \
  -r /image:=/camera/image_raw
```

### 5.2. Düğümü (Node) Başlatma

```bash
ros2 launch video_system_bringup bringup.launch.py
```

### 5.3. Yayınlanan Olayları İzleme

```bash
ros2 topic echo /events
```

Örnek çıktı:

```
stamp:
  sec: 1698787453
  nanosec: 123456789
label: "heartbeat"
score: 1.0
x: 0
y: 0
w: 0
h: 0
```

---

## 6. Klasör Yapısı

```
ros2_ws/src/
├── video_system_interfaces/
│   ├── CMakeLists.txt
│   ├── package.xml
│   └── msg/Event.msg
│
├── video_system_core/
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── include/video_system_core/image_event_node.hpp
│   └── src/
│       ├── image_event_node.cpp
│       └── image_event_main.cpp
│
└── video_system_bringup/
    ├── CMakeLists.txt
    ├── package.xml
    ├── config/params.yaml
    └── launch/bringup.launch.py
```

---

## 7. Gelecek Çalışmalar

* YOLOv11 veya ByteTrack tabanlı `detector_node` eklentisi
* `/detections` ve `/summary` konularının tanımlanması
* SQLite tabanlı olay kaydı
* Çoklu düğüm (multi-node) başlatma desteği





## 9. Yazarlar

**ULAS S.**
GitHub: [ukis666](https://github.com/ukis666)
**GOKTURK C.**
GitHub: [GokturkCan](https://github.com/GokturkCan)



