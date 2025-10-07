# Crowd-Monitoring-
# Crowd Monitoring System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00ADD8.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> AI-powered crowd monitoring system with YOLOv11 - ROI segmentation and ByteTrack integration for real-time person detection and tracking

An intelligent crowd monitoring and analysis platform featuring two powerful pipelines for real-time person detection, counting, and tracking using state-of-the-art YOLOv11 models and advanced computer vision techniques.

![Crowd Monitoring Demo](docs/screenshots/demo.gif)
*Real-time crowd detection and tracking in action*

---

## 🌟 Key Features

### 🎯 Dual Pipeline Architecture

**Pipeline 1: ROI-Based Segmentation**
- Interactive custom Region of Interest (ROI) definition with polygon drawing
- High-precision instance segmentation using YOLOv11m-seg
- Real-time crowd counting within specific areas
- Automated visual alerts when thresholds exceeded
- Color-coded overlay visualization for detected persons
- Perfect for entrance monitoring, restricted zones, and specific area surveillance

**Pipeline 2: ByteTrack Integration**
- Advanced multi-object tracking with stable ID assignment
- Persistent person tracking across video frames
- Motion trail visualization showing movement patterns
- Conservative detection filtering for maximum accuracy
- Comprehensive data export (JSON + summary reports)
- Automated video recording with annotations
- Built with YOLOv11n for optimal real-time performance

### ⚡ Performance & Reliability
- Real-time processing at 30+ FPS (with GPU)
- Robust validation filters minimize false positives
- Configurable confidence thresholds

# 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (optional, recommended for performance)
Webcam or video source
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/crowd-monitoring.git
cd crowd-monitoring
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download models** (automatic on first run)
```bash
# Models download automatically when you run the pipelines
# yolo11m-seg.pt (~48 MB) and yolov11n.pt (~6 MB)
```

---

## 📖 Usage

### Pipeline 1: ROI-Based Crowd Monitoring

Perfect for monitoring **specific areas** like building entrances, walkways, or restricted zones.

```bash
python pipelines/roi_segmentation.py
```

**Interactive Setup:**
1. **Draw ROI**: Click 3+ points on the video to create your monitoring zone
2. **Finalize**: Press `ENTER` to confirm the polygon
3. **Monitor**: System automatically detects people within ROI
4. **Alerts**: Red warning appears when count exceeds threshold (default: 3 people)
5. **Exit**: Press `Q` to quit

**Example Output:**
```
✅ ROI polygon set with 5 points.
People in ROI: 2
People in ROI: 4  ⚠️ ALERT: Too Many People!
```

**Customization:**
```python
# In roi_segmentation.py

# Adjust alert threshold
if people_in_roi > 3:  # Change to your desired limit
    cv2.putText(frame, "ALERT: Too Many People!", ...)

# Adjust detection confidence
results = model(frame, conf=0.25)  # Lower = more detections

# Change input resolution
results = model(frame, imgsz=1280)  # Higher = better accuracy, slower
```

---

### Pipeline 2: ByteTrack Person Tracking

Ideal for **comprehensive tracking** across entire scenes with data logging.

```bash
python pipelines/tracking_detection.py
```

**Keyboard Controls:**
- `Q` - Quit and export results
- `R` - Reset people count
- `C` - Clear all active tracks

**Real-time Display:**
```
People in frame: 5
Total counted: 23
Active tracks: 5
Frame: 1847
```

**Output Files** (saved in `Revised/` folder):
```
Revised/
├── videos/
│   └── fixed_tracking_20251007_143022.mp4
└── data/
    ├── clean_tracking_20251007_143022.json
    └── summary_20251007_143022.txt
```

**JSON Export Sample:**
```json
{
  "frame": 150,
  "timestamp": "2025-10-07T14:30:22.123456",
  "people_in_frame": 3,
  "total_counted": 15,
  "tracks": [
    {
      "id": 5,
      "confidence": 0.87,
      "bbox": [320.5, 180.2, 450.8, 520.6],
      "age": 45
    }
  ]
}
```

**Configuration Options:**
```python
# In tracking_detection.py

# Detection thresholds
conf_threshold = 0.6        # Higher = more precise (0.3-0.9)
iou_threshold = 0.5         # NMS threshold

# Size validation
min_bbox_area = 1000        # Minimum detection size (pixels²)
min_height = 40             # Minimum person height (pixels)

# Tracking stability
min_track_frames = 5        # Frames before counting (prevents false positives)
track_buffer = 25           # Keep lost tracks for N frames
```

---

## 🎬 Video Sources

Both pipelines support multiple input sources:

```python
# Webcam (default camera)
cap = cv2.VideoCapture(0)

# External USB camera
cap = cv2.VideoCapture(1)

# Video file
cap = cv2.VideoCapture("path/to/video.mp4")

# IP Camera / RTSP stream
cap = cv2.VideoCapture("rtsp://username:password@192.168.1.100:554/stream")

# HTTP stream
cap = cv2.VideoCapture("http://camera-ip:port/video")
```

---

## 📁 Project Structure

```
crowd-monitoring/
│
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
│
├── pipelines/                        # Main code directory
│   ├── __init__.py
│   ├── roi_segmentation.py           # Pipeline 1: ROI monitoring
│   └── tracking_detection.py         # Pipeline 2: ByteTrack integration
│
├── models/                           # YOLO model storage
│   ├── README.md                     # Model documentation
│   ├── yolo11m-seg.pt               # Segmentation model (auto-download)
│   └── yolov11n.pt                  # Detection model (auto-download)
│
├── Revised/                          # Auto-generated outputs
│   ├── videos/                       # Recorded videos with annotations
│   └── data/                         # JSON logs and summaries
│
├── docs/                             # Documentation
│   ├── installation.md               # Detailed setup guide
│   ├── api_reference.md             # API documentation
│   └── screenshots/                  # Demo images
│
├── examples/                         # Usage examples
│   ├── basic_usage.py
│   ├── advanced_config.py
│   └── multi_camera.py
│
└── utils/                            # Utility functions
    ├── __init__.py
    ├── config.py                     # Configuration management
    └── visualization.py              # Visualization helpers
```

---

## ⚙️ Advanced Configuration

### ROI Pipeline Settings

```python
# Model configuration
model = YOLO("yolo11m-seg.pt")

# Detection parameters
conf=0.25              # Confidence threshold (0.1-0.9)
iou=0.4                # IoU for NMS (0.3-0.7)
imgsz=1280             # Input size (640, 1280, 1920)

# Alert configuration
alert_threshold = 3    # People count trigger
```

### Tracking Pipeline Settings

```python
# Detection thresholds
conf_threshold = 0.6          # Detection confidence
iou_threshold = 0.5           # NMS IoU threshold

# Bounding box validation
min_bbox_area = 1000          # Minimum size (pixels²)
max_bbox_area = 100000        # Maximum size (pixels²)
min_height = 40               # Minimum height (pixels)
min_aspect_ratio = 0.3        # Min width/height ratio
max_aspect_ratio = 3.0        # Max width/height ratio

# Tracking parameters
min_track_frames = 5          # Stability before counting
track_buffer = 25             # Frames to keep lost tracks
```


**❌ Model download fails**
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolo11m-seg.pt
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov11n.pt

# Or use Python
python -c "from ultralytics import YOLO; YOLO('yolov11n.pt')"
```

**❌ Low FPS / Performance issues**
```python
# Use smaller model
model = YOLO('yolov11n.pt')  # Instead of yolo11m-seg.pt

# Reduce resolution
results = model(frame, imgsz=640)  # Instead of 1280

# Lower frame rate
cv2.waitKey(50)  # Instead of waitKey(1)
```

**❌ Too many false detections**
```python
# Increase confidence threshold
conf_threshold = 0.7  # Instead of 0.6

# Increase minimum track frames
min_track_frames = 10  # Instead of 5

# Increase minimum detection size
min_bbox_area = 1500  # Instead of 1000
```

**❌ Missing detections**
```python
# Lower confidence
conf=0.2  # Instead of 0.25

# Use larger model
model = YOLO('yolo11l-seg.pt')  # Instead of yolo11m-seg.pt

# Increase image size
imgsz=1920  # Instead of 1280
```

**❌ Webcam not detected**
```bash
# Linux: Check permissions
sudo usermod -a -G video $USER
ls -l /dev/video*

# Try different camera indices
cap = cv2.VideoCapture(1)  # Or 2, 3, etc.
```

**❌ CUDA out of memory**
```python
# Force CPU mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Or in model call
results = model(frame, device='cpu')
```

---

## 📈 Performance Benchmarks

**Tested on:**
- CPU: Intel Core i7-10700K
- GPU: NVIDIA RTX 3070 (8GB VRAM)
- Resolution: 1280x720

| Pipeline | Device | FPS | Latency | Accuracy |
|----------|--------|-----|---------|----------|
| ROI Segmentation | CPU | 8-12 | ~100ms | 95%+ |
| ROI Segmentation | GPU | 25-30 | ~35ms | 95%+ |
| ByteTrack | CPU | 15-20 | ~60ms | 92%+ |
| ByteTrack | GPU | 35-45 | ~25ms | 92%+ |

---


---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit** your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add docstrings to new functions
- Update documentation for new features
- Include tests where applicable
- Ensure code runs without errors

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** YOLOv11 models are licensed under AGPL-3.0 by Ultralytics. Commercial use requires an [Ultralytics Enterprise License](https://ultralytics.com/license).

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- [OpenCV](https://opencv.org/) - Computer vision library
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking algorithm
- [NumPy](https://numpy.org/) - Numerical computing


## 🔮 Roadmap & Future Enhancements

### Planned Features
- [ ] 🌡️ Heatmap generation for crowd density visualization
- [ ] 📊 Real-time web dashboard with live statistics
- [ ] 🤖 Anomaly detection (stampede, unusual behavior)





`
