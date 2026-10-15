# 🔫 Firearm Detection System

A real-time firearm and threat detection web application using an ensemble of **Faster R-CNN** (custom weapon model) and **EfficientDet** (COCO person model) with Weighted Box Fusion.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Asritha-sri-krishna/FIREARMS_DETECTION_SYSTEM.git
cd FIREARMS_DETECTION_SYSTEM
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

The model weights are not included in the repository due to file size limits.  
Download and place them inside the `models/weights/` folder:

| File | Description |
|------|-------------|
| `multi_weapon_model.pth` | Custom Faster R-CNN weapon model |
| `tf_efficientdet_d0.pth` | EfficientDet COCO model |

> Place both files at: `models/weights/`

### 5. Run the Application

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 🧠 Models Used

| Model | Purpose |
|-------|---------|
| Faster R-CNN (custom) | Detects **Pistol** and **Knife** |
| EfficientDet-D0 (COCO) | Detects **Persons** |
| Weighted Box Fusion | Ensemble merging of both models |

---

## 📁 Project Structure

```
├── app.py                  # Flask web server
├── config.py               # Model config, thresholds, class mappings
├── requirements.txt        # Python dependencies
├── models/
│   ├── faster_rcnn.py      # Faster R-CNN detector
│   ├── efficientdet.py     # EfficientDet detector
│   ├── ensemble.py         # WBF ensemble logic
│   └── weights/            # ← Place .pth model files here
├── templates/
│   └── index.html          # Frontend UI
└── static/
    ├── css/style.css
    └── js/dashboard.js
```

---

## ⚙️ Requirements

- Python 3.8+
- CUDA-compatible GPU *(optional but recommended — CPU works too)*
- PyTorch, TorchVision, EfficientDet, OpenCV

---

## 🖥️ Tech Stack

- **Backend:** Flask, PyTorch
- **Detection:** Faster R-CNN, EfficientDet, ensemble-boxes (WBF)
- **Frontend:** HTML, CSS, JavaScript
