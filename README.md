<div align="center">

# ♻️ AI-Based Plastic Waste Segregation

### An intelligent deep learning system that classifies plastic waste types from images and provides recycling guidance

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![ResNet18](https://img.shields.io/badge/Model-ResNet18-green?style=for-the-badge)](https://pytorch.org/vision/stable/models/resnet.html)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 🌍 About The Project

**EcoBot** is an AI-powered plastic waste classification system built with **ResNet18 deep learning**. Upload any image of a plastic item and EcoBot will instantly identify the plastic type, provide its recycling code, and give you actionable recycling guidance — helping reduce plastic pollution one scan at a time.

---

## ✨ Features

- 🤖 **AI Classification** — ResNet18 CNN model trained on real plastic waste images
- ♻️ **7 Plastic Types** — Identifies PET, HDPE, PVC, LDPE, PP, PS, and OTHER plastics
- 🌐 **Web Interface** — Beautiful EcoBot chatbot UI accessible from any browser
- 💻 **Terminal Mode** — Run predictions directly from the command line
- 📦 **Batch Processing** — Classify entire folders of images at once
- 📊 **Confidence Scores** — Shows prediction confidence and all class probabilities
- 🔄 **CPU & GPU Support** — Runs on any machine, no GPU required

---

## 🧠 Plastic Types Classified

| Code | Type | Full Name | Recyclability |
|------|------|-----------|---------------|
| #1 | **PET** | Polyethylene Terephthalate | ✅ Highly Recyclable |
| #2 | **HDPE** | High-Density Polyethylene | ✅ Highly Recyclable |
| #3 | **PVC** | Polyvinyl Chloride | ❌ Rarely Recyclable |
| #4 | **LDPE** | Low-Density Polyethylene | ⚠️ Sometimes Recyclable |
| #5 | **PP** | Polypropylene | ♻️ Recyclable |
| #6 | **PS** | Polystyrene | ❌ Rarely Recyclable |
| #7 | **OTHER** | Other Plastics | ❓ Check Guidelines |

---

## 🗂️ Project Structure

```
AI-Based-Plastic-Waste-Segregation/
│
├── 📄 app.py                        # Flask web server & REST API
├── 📄 run_classifier.py             # Terminal-based classifier
├── 📄 requirements.txt              # Python dependencies
│
├── 📁 templates/
│   └── index.html                   # EcoBot web UI
│
├── 📁 Waste segregation.v1i.multiclass/
│   ├── train/                       # Training images
│   ├── valid/                       # Validation images
│   └── test/                        # Test images
│
├── 📓 01_plastic_classification.ipynb
├── 📓 02_plastic_model_training.ipynb
├── 📓 03_resnet18_deployment.ipynb
├── 📓 04_project_integration.ipynb
├── 📓 05_algorithm_test_3epochs.ipynb
└── 📓 06_run_resnet18_terminal.ipynb
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/vimalvijay06/AI--Based-Plastic-Waste-Segregation.git
cd AI--Based-Plastic-Waste-Segregation
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

> If torch installation fails, install it manually:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 🌐 Running the Web App

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

You'll see the **EcoBot** interface — just upload a plastic image and click **Analyze**!

---

## 💻 Running from Terminal

**Classify a single image:**
```bash
python run_classifier.py --image "path/to/your/image.jpg"
```

**Classify a folder of images:**
```bash
python run_classifier.py --folder "path/to/folder"
```

**Limit batch size:**
```bash
python run_classifier.py --folder "path/to/folder" --max 10
```

**Use a custom model file:**
```bash
python run_classifier.py --image "image.jpg" --model "best_model_ResNet18.pth"
```

### Example Terminal Output

```
======================================================================
♻️  PLASTIC WASTE CLASSIFIER - TERMINAL RUNNER
======================================================================

🔄 Loading model from best_model_ResNet18.pth...
✅ Model loaded successfully!
🎯 Classes: ['HDPE', 'LDPE', 'OTHERS', 'PET', 'PP', 'PS', 'PVC']
🏆 Best Accuracy: 0.9950 (99.50%)
🖥️  Device: cpu

======================================================================
🔍 PLASTIC CLASSIFICATION RESULTS
======================================================================
📁 Image: bottle.jpg

🎯 Predicted Type: PET
📊 Confidence: 97.43%

📝 Full Name: Polyethylene Terephthalate
♻️  Recycling Code: #1 PET
🌍 Recyclability: Highly Recyclable
📦 Common Uses: Water bottles, soft drink bottles, food containers
💡 Recycling Tips: Rinse and remove caps before recycling
======================================================================
```

---

## 🔌 API Reference

### `POST /classify`

Classify a plastic image via REST API.

**Request:**
```bash
curl -X POST http://localhost:5000/classify \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "success": true,
  "plastic_type": "PET",
  "confidence": 0.9743,
  "info": {
    "full_name": "Polyethylene Terephthalate",
    "recycling_code": "#1 PET",
    "recyclability": "Highly Recyclable",
    "color": "green"
  }
}
```

### `GET /health`

Check if the server is running.

```json
{ "status": "healthy" }
```

---

## 🏗️ Model Architecture

```
Input Image (224×224×3)
        ↓
   ResNet18 Backbone
   (Pretrained on ImageNet)
        ↓
   Custom FC Layer
   (512 → 7 classes)
        ↓
   Softmax Output
        ↓
   Plastic Type + Confidence
```

- **Base Model:** ResNet18
- **Input Size:** 224 × 224 pixels
- **Normalization:** ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- **Output:** 7 plastic classes with confidence scores

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | ≥ 3.0.0 | Web server & REST API |
| flask-cors | 4.0.0 | Cross-origin requests |
| torch | ≥ 2.6.0 | Deep learning framework |
| torchvision | ≥ 0.15.0 | ResNet18 model & transforms |
| Pillow | ≥ 10.0.0 | Image loading & processing |
| numpy | ≥ 1.24.0 | Numerical operations |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author

**Vimal Vijay**

[![GitHub](https://img.shields.io/badge/GitHub-vimalvijay06-181717?style=for-the-badge&logo=github)](https://github.com/vimalvijay06)

---

<div align="center">

### 🌱 Together, let's make recycling smarter and our planet cleaner!

⭐ **Star this repo if you found it helpful!** ⭐

</div>
