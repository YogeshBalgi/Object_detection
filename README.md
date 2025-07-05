

📘 README.md
========================
# 🧐 Real-Time Object and Face Detection Web App

An intelligent web application for real-time **object detection** using **YOLOv3-Tiny** and **face recognition** using **OpenCV** — all served through a sleek **Flask web interface**.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?style=flat-square&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=flat-square&logo=opencv)

---

## 🚀 Features

- 🎥 Real-time webcam object detection
- 👤 Face detection and recognition with training
- 📹 Upload and analyze videos with object/person detection
- 💾 Automatically downloads YOLOv3-tiny weights and config
- 📆 Flask-powered clean web interface

---

## 🧰 Technologies Used

- **Flask** – For backend and routing
- **OpenCV** – Object detection, face detection & recognition
- **YOLOv3-tiny** – Lightweight object detection model
- **HTML/CSS/JS** – Frontend (optional customization)

---

## 📂 Project Structure

```
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # Main UI page
├── static/
│   ├── known_faces/       # Saved known face images
│   ├── uploads/           # Uploaded videos
│   └── models/            # YOLO weights/configs
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/YogeshBalgi/Object_detection.git
   cd Object_detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python app.py
   ```

4. **Visit the app in your browser**
   ```
   http://127.0.0.1:5000/
   ```

---

## 💡 Usage Instructions

- Click **Start Webcam** to begin live detection.
- Use **Add Person** to register a new face.
- Upload a video to detect objects frame by frame.
- Stop webcam or video processing anytime.

---

## 📅 Automatic YOLO Downloads

When the app starts, it checks and downloads the following if not present:
- `yolov3-tiny.weights`
- `yolov3-tiny.cfg`
- `coco.names`

---

## 🔐 Face Recognition Details

- Faces are saved in `static/known_faces/`
- The system trains using `LBPHFaceRecognizer` (OpenCV)
- Labels are saved in `static/label_map.txt`

---

## 🤖 Example Use Cases

- Smart surveillance
- Access control with known faces
- Real-time object tracking
- AI + Vision-based automation systems

---

## 💬 Credits

Created with ❤️ by [Yogesh Balgi](https://github.com/YogeshBalgi)  
Powered by OpenCV + Flask + YOLO

---

## 🪄 License

MIT License – Use freely and give credits!
