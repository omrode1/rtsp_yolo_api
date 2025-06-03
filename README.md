# RTSP YOLOv8 Model Testing Platform

A FastAPI-based web application for real-time object detection, tracking, and model evaluation using YOLOv8. Supports live camera/RTSP streams, batch image/video testing, model management, and is designed for extensibility and mobile-friendliness.

---

## 🚀 Features
- Real-time video streaming with YOLOv8 detection and overlays
- Object tracking with adjustable parameters
- Class filtering and confidence threshold controls
- Model selection, upload, and management
- GPU memory management
- Camera detection and status
- Responsive frontend (mobile-friendly)
- REST API for automation

---

## 📦 Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rtsp_yolo_api.git
   cd rtsp_yolo_api
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   uvicorn main:app --reload
   ```
4. **Access the UI:**
   Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📝 Usage
- Select a camera or RTSP stream to view detections in real time.
- Upload or select YOLOv8 models (.pt files).
- Adjust detection classes, confidence, and tracking parameters.
- View GPU status and manage memory.

---

## 🗂️ TODO

### Model Testing & Evaluation
- [ ] Batch image/video upload and testing
- [ ] Display detection results side-by-side with ground truth
- [ ] Compute and display metrics (mAP, precision, recall, FPS)
- [ ] Downloadable evaluation reports
- [ ] Model comparison (visual diff, overlay detections)
- [ ] Dataset management (upload, organize, annotate)
- [ ] Support for COCO, VOC, YOLO formats
- [ ] Interactive visualization (click on detections, zoom/pan)

### User Experience
- [ ] Session management (save settings, recent uploads)
- [ ] Model zoo integration (download/test public models)
- [ ] REST API documentation and expansion

### Mobile Version
- [ ] Responsive UI (PWA/mobile web)
- [ ] Native mobile app (React Native/Flutter)
- [ ] Real-time camera detection from phone
- [ ] On-device model testing (ONNX/TFLite)

### Cloud & Collaboration
- [ ] User authentication and multi-user support
- [ ] Cloud storage for datasets and results
- [ ] Share results/models with others

### Advanced Features
- [ ] AutoML: fine-tune models on user data
- [ ] Explainability: saliency maps, prediction explanations
- [ ] Edge device integration (Jetson, Raspberry Pi)

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
[MIT](LICENSE) 