# 🐟 Multiclass Fish Image Classification

A single-file Streamlit application that covers the entire project lifecycle:
EDA, CNN-from-scratch training, transfer learning with 5 pretrained backbones,
model comparison, and a live prediction UI.

## 1. Setup

```bash
pip install -r requirements.txt
```

Unzip your dataset so it looks like this (folder names become class labels):

```
images.cv_xxx/data/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg
└── test/<class_name>/*.jpg
```

## 2. Run

```bash
streamlit run app.py
```

In the sidebar, set **"Dataset root folder"** to the path above (e.g.
`images.cv_jzk6llhf18tm3k0kyttxz/data`).

## 3. Using the app

| Page | Purpose |
|---|---|
| 🏠 Home | Dataset summary + sample images |
| 📊 Data Exploration | Class balance, image-size/brightness analytics, augmentation preview |
| 🧠 Model Training | Pick any of: CNN (Scratch), VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0. Set epochs/batch size, click **Start Training** |
| 📈 Model Comparison | Leaderboard, radar chart, training curves, confusion matrices per model |
| 🔍 Predict | Upload an image, get the predicted species + confidence, with a session log |

Trained models, histories, and metrics are saved under `artifacts/` (`models/`,
`history/`, `reports/`) so results persist across app restarts.

## 4. Notes

- Input size is fixed at 224×224×3 (compatible with all 5 backbones).
- The **"animal fish bass"** class has very few images (~30 train samples) —
  the Data Exploration page will flag this imbalance; consider merging it
  with a related class or gathering more samples for production use.
- Training runs inside the Streamlit process — for a full run over many
  epochs, use a machine with a GPU.
- `.h5` model files can be large; if pushing to GitHub, use Git LFS or only
  commit the best-performing model.

## 5. Project structure

```
.
├── app.py              # single Streamlit application (this project)
├── requirements.txt
├── README.md
└── artifacts/           # created at runtime
    ├── models/*.h5
    ├── history/*.json
    └── reports/*.json
```
