"""
================================================================================
 MULTICLASS FISH IMAGE CLASSIFICATION — All-in-One Streamlit Application
================================================================================
A single, self-contained Streamlit program that covers the full project
lifecycle:

    1. Home                  -> project & dataset overview
    2. Data Exploration      -> analytics + visualizations (EDA)
    3. Model Training        -> CNN from scratch + 5 transfer-learning models
    4. Model Comparison      -> metrics, curves, confusion matrices, ranking
    5. Predict               -> upload an image and get a live prediction

Run with:
    streamlit run app.py

Expected dataset layout (ImageDataGenerator / flow_from_directory style):

    DATA_DIR/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg

Author: Generated for the "Multiclass Fish Image Classification" project.
================================================================================
"""

import os
import gc
# Disable oneDNN/MKL optimizations and limit thread pools to prevent "could not create a memory object" graph execution errors on CPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# Force pure-Python protobuf implementation to fix "EncodeError: Failed to serialize proto"
# caused by a version mismatch between TensorFlow and the C++ protobuf runtime.
# Must be set before any TF/Keras import.
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import io
import json
import time
import random
import glob
import zipfile
import shutil
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import streamlit as st

# TensorFlow / Keras — lazy-loaded to avoid crashing on low-memory systems -----
# These are only needed for Model Training, Model Comparison, and Predict pages.
tf = None
layers = models = optimizers = None
ImageDataGenerator = None
Callback = ModelCheckpoint = EarlyStopping = None
VGG16 = ResNet50 = MobileNet = InceptionV3 = EfficientNetB0 = None
confusion_matrix = classification_report = precision_recall_fscore_support = None
StreamlitProgressCallback = None


def _ensure_ml_imports():
    """Import TensorFlow and sklearn on first use. Called only by pages that need them."""
    global tf, layers, models, optimizers
    global ImageDataGenerator, Callback, ModelCheckpoint, EarlyStopping
    global VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
    global confusion_matrix, classification_report, precision_recall_fscore_support
    global StreamlitProgressCallback

    if tf is not None:
        return  # already loaded

    import tensorflow as _tf
    tf = _tf


    # Monkey-patch get_graph_debug_info to suppress protobuf DecodeError /
    # EncodeError that arise from a version mismatch between TensorFlow and the
    # installed protobuf runtime. The function only generates debug stack-trace
    # info for error messages — it has no effect on training correctness.
    try:
        import tensorflow.python.eager.context as _tf_eager_ctx
        _orig_get_graph_debug_info = _tf_eager_ctx.get_graph_debug_info

        def _safe_get_graph_debug_info(*args, **kwargs):
            try:
                return _orig_get_graph_debug_info(*args, **kwargs)
            except Exception:
                return None

        _tf_eager_ctx.get_graph_debug_info = _safe_get_graph_debug_info
    except Exception:
        pass  # If the internal API has changed, do nothing and let TF handle it.

    # Configure threading to prevent memory allocation errors (e.g. MKL memory objects) on CPU
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
    from tensorflow.keras import layers as _l, models as _m, optimizers as _o
    layers, models, optimizers = _l, _m, _o
    from tensorflow.keras.preprocessing.image import ImageDataGenerator as _IDG
    ImageDataGenerator = _IDG
    from tensorflow.keras.callbacks import (
        Callback as _Cb, ModelCheckpoint as _MC, EarlyStopping as _ES,
    )
    Callback, ModelCheckpoint, EarlyStopping = _Cb, _MC, _ES
    from tensorflow.keras.applications import (
        VGG16 as _v, ResNet50 as _r, MobileNet as _mn,
        InceptionV3 as _iv3, EfficientNetB0 as _eb0,
    )
    from tensorflow.keras.applications.vgg16 import preprocess_input as _vgg_pre
    from tensorflow.keras.applications.resnet50 import preprocess_input as _resnet_pre
    from tensorflow.keras.applications.mobilenet import preprocess_input as _mobilenet_pre
    from tensorflow.keras.applications.inception_v3 import preprocess_input as _inception_pre
    from tensorflow.keras.applications.efficientnet import preprocess_input as _effnet_pre
    VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0 = _v, _r, _mn, _iv3, _eb0
    from sklearn.metrics import (
        confusion_matrix as _cm, classification_report as _cr,
        precision_recall_fscore_support as _prfs,
    )
    confusion_matrix, classification_report, precision_recall_fscore_support = _cm, _cr, _prfs

    # Update the catalog since they were None at startup
    global MODEL_CATALOG
    MODEL_CATALOG["VGG16"]["base"]       = VGG16
    MODEL_CATALOG["VGG16"]["preprocess"] = _vgg_pre
    MODEL_CATALOG["ResNet50"]["base"]       = ResNet50
    MODEL_CATALOG["ResNet50"]["preprocess"] = _resnet_pre
    MODEL_CATALOG["MobileNet"]["base"]       = MobileNet
    MODEL_CATALOG["MobileNet"]["preprocess"] = _mobilenet_pre
    MODEL_CATALOG["InceptionV3"]["base"]       = InceptionV3
    MODEL_CATALOG["InceptionV3"]["preprocess"] = _inception_pre
    MODEL_CATALOG["EfficientNetB0"]["base"]       = EfficientNetB0
    MODEL_CATALOG["EfficientNetB0"]["preprocess"] = _effnet_pre

    class _StreamlitProgressCallback(Callback):
        """Pushes per-epoch metrics into a placeholder so the user sees live curves."""

        def __init__(self, total_epochs, chart_placeholder, status_placeholder, model_name):
            super().__init__()
            self.total_epochs = total_epochs
            self.chart_placeholder = chart_placeholder
            self.status_placeholder = status_placeholder
            self.model_name = model_name
            self.history = {"epoch": [], "accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.history["epoch"].append(epoch + 1)
            self.history["accuracy"].append(logs.get("accuracy"))
            self.history["val_accuracy"].append(logs.get("val_accuracy"))
            self.history["loss"].append(logs.get("loss"))
            self.history["val_loss"].append(logs.get("val_loss"))

            self.status_placeholder.progress(
                (epoch + 1) / self.total_epochs,
                text=f"Training {self.model_name}: epoch {epoch + 1}/{self.total_epochs} "
                     f"— val_accuracy={logs.get('val_accuracy', 0):.3f}"
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.history["epoch"], y=self.history["accuracy"],
                                      mode="lines+markers", name="train acc"))
            fig.add_trace(go.Scatter(x=self.history["epoch"], y=self.history["val_accuracy"],
                                      mode="lines+markers", name="val acc"))
            fig.update_layout(title=f"{self.model_name} — Accuracy per epoch",
                               xaxis_title="Epoch", yaxis_title="Accuracy",
                               height=320, margin=dict(l=10, r=10, t=40, b=10))
            self.chart_placeholder.plotly_chart(fig, use_container_width=True,
                                                 key=f"live_{self.model_name}_{epoch}")

    StreamlitProgressCallback = _StreamlitProgressCallback


# ==============================================================================
# 0. GLOBAL CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Fish Image Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = (224, 224)          # Standard input size compatible with all 5 pretrained nets
BATCH_SIZE_DEFAULT = 16
ARTIFACT_DIR = "artifacts"      # Where trained models / histories / reports are stored
MODELS_DIR = os.path.join(ARTIFACT_DIR, "models")
HISTORY_DIR = os.path.join(ARTIFACT_DIR, "history")
REPORTS_DIR = os.path.join(ARTIFACT_DIR, "reports")
for d in (MODELS_DIR, HISTORY_DIR, REPORTS_DIR):
    os.makedirs(d, exist_ok=True)

MODEL_CATALOG = {
    "CNN (Scratch)": {"builder": "cnn_scratch", "base": None,          "preprocess": None},
    "VGG16":         {"builder": "transfer",    "base": VGG16,         "preprocess": None},
    "ResNet50":      {"builder": "transfer",    "base": ResNet50,      "preprocess": None},
    "MobileNet":     {"builder": "transfer",    "base": MobileNet,     "preprocess": None},
    "InceptionV3":   {"builder": "transfer",    "base": InceptionV3,   "preprocess": None},
    "EfficientNetB0":{"builder": "transfer",    "base": EfficientNetB0,"preprocess": None},
}

PALETTE = px.colors.qualitative.Set2

# ==============================================================================
# 1. HELPER / UTILITY FUNCTIONS
# ==============================================================================

@st.cache_data(show_spinner=False)
def scan_dataset(data_dir: str):
    """Walk the train/val/test folders and build a tidy DataFrame of image counts."""
    rows = []
    for split in ("train", "val", "test"):
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            continue
        for cls in sorted(os.listdir(split_path)):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue
            n = len([f for f in os.listdir(cls_path)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            rows.append({"split": split, "class": cls, "count": n})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def sample_image_paths(data_dir: str, split: str, cls: str, n: int = 5):
    cls_path = os.path.join(data_dir, split, cls)
    if not os.path.isdir(cls_path):
        return []
    files = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.seed(42)
    random.shuffle(files)
    return [os.path.join(cls_path, f) for f in files[:n]]


@st.cache_data(show_spinner=False)
def analyze_image_properties(data_dir: str, split: str, samples_per_class: int = 15):
    """Sample a handful of images per class and compute size / brightness stats."""
    df = scan_dataset(data_dir)
    classes = sorted(df[df["split"] == split]["class"].unique()) if not df.empty else []
    records = []
    for cls in classes:
        paths = sample_image_paths(data_dir, split, cls, samples_per_class)
        for p in paths:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    arr = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)
                    brightness = arr.mean()
                records.append({"class": cls, "width": w, "height": h,
                                 "aspect_ratio": round(w / h, 2), "brightness": brightness})
            except Exception:
                continue
    return pd.DataFrame(records)


def get_generators(data_dir, img_size, batch_size, augment=True, preprocessing_fn=None):
    """Build train/val/test generators.

    Args:
        preprocessing_fn: If provided (e.g. vgg16.preprocess_input), it is used
            instead of the generic rescale=1/255.  Transfer-learning models each
            have their own expected input range; using the wrong one is the most
            common cause of near-random accuracy.
    """
    # Build the preprocessing config for ImageDataGenerator.
    # Using 'preprocessing_function' takes priority over 'rescale', so we only
    # set rescale when no model-specific preprocessor is available.
    pre_kwargs = (
        {"preprocessing_function": preprocessing_fn}
        if preprocessing_fn is not None
        else {"rescale": 1.0 / 255}
    )

    if augment:
        train_datagen = ImageDataGenerator(
            **pre_kwargs,
            rotation_range=30,
            zoom_range=0.2,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator(**pre_kwargs)

    eval_datagen = ImageDataGenerator(**pre_kwargs)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"), target_size=img_size,
        batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42,
    )
    val_gen = eval_datagen.flow_from_directory(
        os.path.join(data_dir, "val"), target_size=img_size,
        batch_size=batch_size, class_mode="categorical", shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        os.path.join(data_dir, "test"), target_size=img_size,
        batch_size=batch_size, class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen, test_gen


def build_cnn_scratch(input_shape, num_classes):
    """A compact CNN trained from scratch."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="CNN_Scratch")
    return model


def build_transfer_model(base_cls, input_shape, num_classes, fine_tune=False, weights="imagenet"):
    """Attach a classification head on top of a frozen (or partially fine-tuned) backbone."""
    base = base_cls(include_top=False, weights=weights, input_shape=input_shape)
    base.trainable = fine_tune
    if fine_tune:
        # Only fine-tune the last 20% of layers to keep training light.
        freeze_until = int(len(base.layers) * 0.8)
        for layer in base.layers[:freeze_until]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False if not fine_tune else None)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name=base_cls.__name__)
    return model


# StreamlitProgressCallback is defined dynamically inside _ensure_ml_imports()


def evaluate_model(model, test_gen, class_names):
    """Run inference on the test set in small batches to avoid CPU OOM errors."""
    # Use a small fixed batch size for prediction regardless of training batch size.
    # This prevents ResourceExhaustedError on machines with limited RAM.
    PREDICT_BATCH = 4

    test_gen.reset()
    n_samples = test_gen.n
    img_size = test_gen.target_size  # e.g. (224, 224)

    all_probs = []
    samples_seen = 0
    test_gen.batch_size = PREDICT_BATCH  # shrink batch for inference pass
    test_gen.reset()

    for batch_x, _ in test_gen:
        preds = model.predict_on_batch(batch_x)  # single batch, no graph overhead
        all_probs.append(preds)
        samples_seen += len(batch_x)
        if samples_seen >= n_samples:
            break
        gc.collect()  # release intermediate tensors between batches

    probs = np.concatenate(all_probs, axis=0)[:n_samples]
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes[:n_samples]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    acc = float(np.mean(y_pred == y_true))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                    zero_division=0, output_dict=True)
    return {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": cm.tolist(), "report": report,
    }


def save_artifacts(name, model, history_dict, metrics_dict, train_seconds, num_params):
    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    model_path = os.path.join(MODELS_DIR, f"{safe}.h5")
    model.save(model_path)

    hist_path = os.path.join(HISTORY_DIR, f"{safe}.json")
    with open(hist_path, "w") as f:
        json.dump(history_dict, f)

    report_path = os.path.join(REPORTS_DIR, f"{safe}.json")
    metrics_dict = dict(metrics_dict)
    metrics_dict["train_seconds"] = train_seconds
    metrics_dict["num_params"] = num_params
    metrics_dict["model_name"] = name
    with open(report_path, "w") as f:
        json.dump(metrics_dict, f)
    return model_path, hist_path, report_path


@st.cache_data(show_spinner=False)
def load_all_reports():
    rows = []
    for path in glob.glob(os.path.join(REPORTS_DIR, "*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


@st.cache_resource(show_spinner=False)
def load_keras_model(path):
    _ensure_ml_imports()
    return tf.keras.models.load_model(path)


def plot_confusion_matrix_heatmap(cm, class_names, title="Confusion Matrix"):
    cm = np.array(cm)
    fig = px.imshow(
        cm, text_auto=True, aspect="auto",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names, y=class_names, color_continuous_scale="Blues",
        title=title,
    )
    fig.update_layout(height=550, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ==============================================================================
# 2. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🐟 Fish Classifier")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📁 Upload Dataset", "📊 Data Exploration", "🧠 Model Training", "📈 Model Comparison", "🔍 Predict"],
)

st.sidebar.markdown("---")
_default_data_dir = st.session_state.get(
    "dataset_path", "images.cv_jzk6llhf18tm3k0kyttxz/data"
)
data_dir = st.sidebar.text_input(
    "Dataset root folder",
    value=_default_data_dir,
    help="Folder containing train/ val/ test/ sub-folders (one folder per fish species).",
)
# Keep session state in sync with manual sidebar edits
if data_dir != st.session_state.get("dataset_path"):
    st.session_state["dataset_path"] = data_dir
st.sidebar.caption("Change this if your dataset lives somewhere else on disk.")

# ==============================================================================
# 3. PAGE: HOME
# ==============================================================================
if page == "🏠 Home":
    st.title("🐟 Multiclass Fish Image Classification")
    st.markdown(
        """
        This app trains and compares **6 deep-learning models** on a fish-species
        image dataset, then serves the best one behind a live prediction UI.

        | Stage | What happens |
        |---|---|
        | **Data Exploration** | Class balance, image-size / brightness analytics, sample grids, augmentation preview |
        | **Model Training** | CNN from scratch + VGG16 / ResNet50 / MobileNet / InceptionV3 / EfficientNetB0 transfer learning |
        | **Model Comparison** | Accuracy, precision, recall, F1, confusion matrices, training curves, leaderboard |
        | **Predict** | Upload a fish photo and get the class + confidence scores |
        """
    )

    df = scan_dataset(data_dir)
    if df.empty:
        st.warning(
            "Couldn't find a dataset at the path in the sidebar. "
            "Point it to a folder that contains `train/`, `val/` and `test/` sub-folders."
        )
    else:
        classes = sorted(df["class"].unique())
        total_imgs = int(df["count"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total images", f"{total_imgs:,}")
        c2.metric("Classes", len(classes))
        c3.metric("Train images", int(df[df.split == "train"]["count"].sum()))
        c4.metric("Test images", int(df[df.split == "test"]["count"].sum()))

        st.subheader("Sample images")
        preview_classes = classes[:6]
        cols = st.columns(len(preview_classes))
        for col, cls in zip(cols, preview_classes):
            paths = sample_image_paths(data_dir, "train", cls, 1)
            if paths:
                col.image(paths[0], caption=cls, use_container_width=True)

# ==============================================================================
# 3b. PAGE: UPLOAD DATASET
# ==============================================================================
elif page == "📁 Upload Dataset":
    st.title("📁 Upload Your Dataset")

    UPLOAD_DIR = "uploaded_dataset"

    st.markdown(
        """
        Upload a **ZIP file** containing your image dataset.
        The ZIP must follow this folder structure:

        ```
        your_dataset.zip
        └── <any_root_folder>/        ← optional single root folder
            ├── train/
            │   ├── class_A/
            │   └── class_B/
            ├── val/
            │   ├── class_A/
            │   └── class_B/
            └── test/
                ├── class_A/
                └── class_B/
        ```

        Supported image formats: **JPG, JPEG, PNG**
        """
    )

    uploaded_zip = st.file_uploader(
        "Drop your dataset ZIP here",
        type=["zip"],
        help="ZIP file containing train/, val/, test/ folders with class sub-folders.",
    )

    if uploaded_zip is not None:
        st.info(f"📦 Received: **{uploaded_zip.name}** ({uploaded_zip.size / 1024 / 1024:.1f} MB)")

        if st.button("🚀 Extract & Use This Dataset", type="primary"):
            with st.spinner("Extracting dataset…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    tmp.write(uploaded_zip.read())
                    tmp_path = tmp.name

                if os.path.exists(UPLOAD_DIR):
                    shutil.rmtree(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR, exist_ok=True)

                try:
                    with zipfile.ZipFile(tmp_path, "r") as zf:
                        zf.extractall(UPLOAD_DIR)
                    os.remove(tmp_path)

                    # Auto-detect root: recursively traverse until we find a folder containing train/val/test splits
                    detected_root = UPLOAD_DIR
                    for root, dirs, files in os.walk(UPLOAD_DIR):
                        if any(s in dirs for s in ("train", "val", "test")):
                            detected_root = root
                            break

                    found_splits = [s for s in ("train", "val", "test")
                                    if os.path.isdir(os.path.join(detected_root, s))]

                    if not found_splits:
                        st.error(
                            "❌ Could not find `train/`, `val/`, or `test/` folders inside the ZIP. "
                            "Please re-check your folder structure and try again."
                        )
                        shutil.rmtree(UPLOAD_DIR)
                    else:
                        total_imgs = 0
                        class_set = set()
                        split_summary = {}
                        for split in found_splits:
                            sp = os.path.join(detected_root, split)
                            cnt = 0
                            for cls in os.listdir(sp):
                                cp = os.path.join(sp, cls)
                                if os.path.isdir(cp):
                                    class_set.add(cls)
                                    cnt += len([f for f in os.listdir(cp)
                                                if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                            split_summary[split] = cnt
                            total_imgs += cnt

                        st.session_state["dataset_path"] = detected_root
                        scan_dataset.clear()
                        sample_image_paths.clear()
                        analyze_image_properties.clear()

                        st.success("✅ Dataset extracted and ready!")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total images", f"{total_imgs:,}")
                        c2.metric("Classes", len(class_set))
                        c3.metric("Splits found", len(found_splits))
                        c4.metric("Active path", detected_root)

                        st.markdown("#### Split breakdown")
                        for s, n in split_summary.items():
                            st.write(f"- **{s}**: {n:,} images")

                        st.markdown("#### Classes detected")
                        st.write(", ".join(sorted(class_set)))

                except zipfile.BadZipFile:
                    st.error("❌ The uploaded file is not a valid ZIP archive.")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    # ── Current Dataset Status ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Current Dataset Status")
    current_path = st.session_state.get("dataset_path", "")
    if not current_path:
        st.warning("No dataset active yet. Upload a ZIP above.")
    else:
        df_status = scan_dataset(current_path)
        if df_status.empty:
            st.warning(f"Path `{current_path}` found but no images detected. Check the folder structure.")
        else:
            classes_status = sorted(df_status["class"].unique())
            c1, c2, c3 = st.columns(3)
            c1.metric("Classes", len(classes_status))
            c2.metric("Total images", int(df_status["count"].sum()))
            c3.metric("Active path", current_path)
            st.dataframe(
                df_status.pivot_table(index="class", columns="split", values="count", fill_value=0),
                use_container_width=True,
            )

# ==============================================================================
# 4. PAGE: DATA EXPLORATION (EDA)
# ==============================================================================
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration & Analytics")

    df = scan_dataset(data_dir)
    if df.empty:
        st.error("No dataset found at the given path.")
        st.stop()

    classes = sorted(df["class"].unique())

    # --- Class distribution -----------------------------------------------
    st.subheader("1. Class distribution across splits")
    fig = px.bar(df, x="class", y="count", color="split", barmode="group",
                 color_discrete_sequence=PALETTE,
                 title="Image count per class, per split")
    fig.update_layout(xaxis_tickangle=-35, height=450)
    st.plotly_chart(fig, use_container_width=True)

    train_counts = df[df.split == "train"].set_index("class")["count"]
    imbalance_ratio = train_counts.max() / max(train_counts.min(), 1)
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(df[df.split == "train"], names="class", values="count",
                          title="Training set share by class", hole=0.35,
                          color_discrete_sequence=PALETTE)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.markdown("#### Class balance check")
        st.metric("Largest / smallest class ratio (train)", f"{imbalance_ratio:.1f}×")
        if imbalance_ratio > 5:
            st.warning(
                "⚠️ Significant class imbalance detected. Consider using "
                "`class_weight`, oversampling the minority class(es), or "
                "focal loss when training."
            )
        else:
            st.success("Class distribution is reasonably balanced.")
        st.dataframe(train_counts.sort_values(ascending=False).rename("train images"))

    # --- Image property analytics ------------------------------------------
    st.subheader("2. Image property analytics (sampled)")
    n_samp = st.slider("Images sampled per class for analysis", 5, 40, 15)
    props_df = analyze_image_properties(data_dir, "train", n_samp)

    if not props_df.empty:
        colA, colB = st.columns(2)
        with colA:
            fig_dim = px.scatter(props_df, x="width", y="height", color="class",
                                  title="Image dimensions by class",
                                  color_discrete_sequence=PALETTE)
            st.plotly_chart(fig_dim, use_container_width=True)
        with colB:
            fig_bright = px.box(props_df, x="class", y="brightness", color="class",
                                 title="Brightness distribution by class",
                                 color_discrete_sequence=PALETTE)
            fig_bright.update_layout(xaxis_tickangle=-35, showlegend=False)
            st.plotly_chart(fig_bright, use_container_width=True)

        fig_ar = px.histogram(props_df, x="aspect_ratio", nbins=20,
                               title="Aspect-ratio distribution (width / height)",
                               color_discrete_sequence=PALETTE)
        st.plotly_chart(fig_ar, use_container_width=True)

    # --- Sample gallery -------------------------------------------------------
    st.subheader("3. Sample image gallery")
    chosen_cls = st.selectbox("Choose a class to preview", classes)
    paths = sample_image_paths(data_dir, "train", chosen_cls, 6)
    if paths:
        cols = st.columns(len(paths))
        for c, p in zip(cols, paths):
            c.image(p, use_container_width=True)

    # --- Augmentation preview -------------------------------------------------
    st.subheader("4. Data augmentation preview")
    st.caption("This is exactly the augmentation pipeline used during training.")
    if paths:
        base_img = Image.open(paths[0]).convert("RGB").resize(IMG_SIZE)
        arr = np.expand_dims(np.array(base_img), 0)
        _ensure_ml_imports()
        aug = ImageDataGenerator(
            rotation_range=30, zoom_range=0.2, width_shift_range=0.15,
            height_shift_range=0.15, shear_range=0.15, horizontal_flip=True,
            fill_mode="nearest",
        )
        aug_iter = aug.flow(arr, batch_size=1)
        cols = st.columns(6)
        cols[0].image(base_img, caption="Original", use_container_width=True)
        for i in range(1, 6):
            batch = next(aug_iter)[0].astype("uint8")
            cols[i].image(batch, caption=f"Augmented {i}", use_container_width=True)

# ==============================================================================
# 5. PAGE: MODEL TRAINING
# ==============================================================================
elif page == "🧠 Model Training":
    st.title("🧠 Train CNN & Transfer-Learning Models")

    df = scan_dataset(data_dir)
    if df.empty:
        st.error("No dataset found at the given path.")
        st.stop()
    class_names = sorted(df["class"].unique())
    num_classes = len(class_names)

    _ensure_ml_imports()

    st.markdown(
        "Select which architectures to train. Transfer-learning models load "
        "**ImageNet** weights and train a new classification head on top "
        "(optionally fine-tuning the top layers of the backbone)."
    )

    selected = st.multiselect(
        "Models to train", list(MODEL_CATALOG.keys()),
        default=["CNN (Scratch)"],
    )

    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.number_input("Epochs", 1, 100, 10)
    batch_size = c2.number_input("Batch size", 8, 128, BATCH_SIZE_DEFAULT, step=8)
    fine_tune = c3.checkbox("Fine-tune backbone (transfer models)", value=False)
    use_augment = c4.checkbox("Use data augmentation", value=True)

    if st.button("🚀 Start Training", type="primary", disabled=(len(selected) == 0)):
        # Clear any leftover Keras session memory/graphs before starting
        tf.keras.backend.clear_session()
        gc.collect()

        input_shape = IMG_SIZE + (3,)

        for name in selected:
            st.markdown(f"### Training: {name}")
            spec = MODEL_CATALOG[name]

            # Build per-model generators with the correct preprocessing function.
            # Each backbone expects a different input range (e.g. VGG16 expects
            # BGR + mean subtraction, NOT [0,1]).  Using the wrong preprocessor
            # is the most common cause of near-random accuracy on transfer models.
            train_gen, val_gen, test_gen = get_generators(
                data_dir, IMG_SIZE, int(batch_size),
                augment=use_augment,
                preprocessing_fn=spec["preprocess"],  # None → rescale=1/255 for CNN scratch
            )

            # Clear session and collect garbage before building this model to avoid memory accumulation
            tf.keras.backend.clear_session()
            gc.collect()

            if spec["builder"] == "cnn_scratch":
                model = build_cnn_scratch(input_shape, num_classes)
            else:
                try:
                    model = build_transfer_model(spec["base"], input_shape, num_classes, fine_tune, weights="imagenet")
                except Exception as download_err:
                    if "getaddrinfo" in str(download_err) or "URL fetch" in str(download_err) or "urlopen" in str(download_err).lower():
                        st.warning(
                            f"⚠️ Could not download ImageNet weights for **{name}** "
                            f"(no internet access). Falling back to **random weights** — "
                            f"accuracy will be lower but training will still run."
                        )
                        try:
                            tf.keras.backend.clear_session()
                            model = build_transfer_model(spec["base"], input_shape, num_classes, fine_tune, weights=None)
                        except Exception as e2:
                            st.error(f"❌ Failed to build {name} even without pre-trained weights: {e2}")
                            continue
                    else:
                        st.error(f"❌ Failed to build {name}: {download_err}")
                        continue

            model.compile(
                optimizer=optimizers.Adam(learning_rate=1e-4 if fine_tune else 1e-3),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            chart_ph = st.empty()
            status_ph = st.empty()
            cb = StreamlitProgressCallback(int(epochs), chart_ph, status_ph, name)
            early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

            start = time.time()
            model.fit(
                train_gen, validation_data=val_gen, epochs=int(epochs),
                callbacks=[cb, early_stop], verbose=0,
            )
            train_seconds = time.time() - start

            with st.spinner(f"Evaluating {name} on the test set..."):
                metrics = evaluate_model(model, test_gen, class_names)

            num_params = model.count_params()
            save_artifacts(name, model, cb.history, metrics, train_seconds, num_params)

            st.success(
                f"✅ {name} done in {train_seconds:.0f}s — "
                f"test accuracy = {metrics['accuracy']:.3f}, F1 = {metrics['f1']:.3f}"
            )

            # Free memory immediately after we are done training and evaluating this model
            del model
            tf.keras.backend.clear_session()
            gc.collect()

        load_all_reports.clear()
        st.info("All selected models trained and saved to `artifacts/`. "
                "Head to **Model Comparison** to review results.")

    st.markdown("---")
    st.caption(
        "💡 Training runs happen inside this Streamlit process — for large "
        "datasets or many epochs, run this on a machine with a GPU."
    )

# ==============================================================================
# 6. PAGE: MODEL COMPARISON
# ==============================================================================
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison & Leaderboard")

    reports = load_all_reports()
    if not reports:
        st.info("No trained models found yet. Go to **Model Training** first.")
        st.stop()

    comp_df = pd.DataFrame([{
        "Model": r["model_name"],
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1-score": r["f1"],
        "Params (M)": round(r["num_params"] / 1e6, 2),
        "Train time (s)": round(r["train_seconds"], 1),
    } for r in reports]).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    best_model_name = comp_df.iloc[0]["Model"]
    st.success(f"🏆 Best model so far: **{best_model_name}** "
               f"(accuracy = {comp_df.iloc[0]['Accuracy']:.3f})")

    st.subheader("Leaderboard")
    st.dataframe(
        comp_df.style.format({
            "Accuracy": "{:.3f}", "Precision": "{:.3f}",
            "Recall": "{:.3f}", "F1-score": "{:.3f}",
        }).background_gradient(subset=["Accuracy", "F1-score"], cmap="Greens"),
        use_container_width=True,
    )

    st.subheader("Metric comparison")
    metric_long = comp_df.melt(
        id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
        var_name="Metric", value_name="Score",
    )
    fig_bar = px.bar(metric_long, x="Model", y="Score", color="Metric", barmode="group",
                      color_discrete_sequence=PALETTE, title="Accuracy / Precision / Recall / F1 per model")
    fig_bar.update_layout(height=450, xaxis_tickangle=-20)
    st.plotly_chart(fig_bar, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_radar = go.Figure()
        for r in reports:
            fig_radar.add_trace(go.Scatterpolar(
                r=[r["accuracy"], r["precision"], r["recall"], r["f1"]],
                theta=["Accuracy", "Precision", "Recall", "F1"],
                fill="toself", name=r["model_name"],
            ))
        fig_radar.update_layout(title="Radar comparison", height=420,
                                 polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig_radar, use_container_width=True)
    with col2:
        fig_eff = px.scatter(comp_df, x="Params (M)", y="Accuracy", size="Train time (s)",
                              color="Model", title="Accuracy vs. model size (bubble = train time)",
                              color_discrete_sequence=PALETTE)
        fig_eff.update_layout(height=420)
        st.plotly_chart(fig_eff, use_container_width=True)

    st.subheader("Training curves")
    tabs = st.tabs([r["model_name"] for r in reports])
    for tab, r in zip(tabs, reports):
        safe = r["model_name"].replace(" ", "_").replace("(", "").replace(")", "")
        hist_path = os.path.join(HISTORY_DIR, f"{safe}.json")
        with tab:
            if os.path.exists(hist_path):
                with open(hist_path) as f:
                    hist = json.load(f)
                c1, c2 = st.columns(2)
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=hist["epoch"], y=hist["accuracy"], name="train"))
                fig_acc.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_accuracy"], name="val"))
                fig_acc.update_layout(title="Accuracy", height=350)
                c1.plotly_chart(fig_acc, use_container_width=True)

                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=hist["epoch"], y=hist["loss"], name="train"))
                fig_loss.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_loss"], name="val"))
                fig_loss.update_layout(title="Loss", height=350)
                c2.plotly_chart(fig_loss, use_container_width=True)

            cm = r.get("confusion_matrix")
            if cm:
                _ds = scan_dataset(data_dir)
                if "class" not in _ds.columns or _ds.empty:
                    st.warning("Dataset directory not found or has no class folders — cannot render confusion matrix.")
                else:
                    class_names = sorted(_ds["class"].unique())
                    st.plotly_chart(
                        plot_confusion_matrix_heatmap(cm, class_names, f"{r['model_name']} — Confusion Matrix"),
                        use_container_width=True,
                    )
            with st.expander("Per-class precision / recall / F1"):
                rep_df = pd.DataFrame(r["report"]).T
                st.dataframe(rep_df.style.format(precision=3), use_container_width=True)

# ==============================================================================
# 7. PAGE: PREDICT
# ==============================================================================
elif page == "🔍 Predict":
    st.title("🔍 Predict Fish Species")

    reports = load_all_reports()
    if not reports:
        st.info("No trained models found yet. Go to **Model Training** first.")
        st.stop()

    _ensure_ml_imports()

    model_names = [r["model_name"] for r in reports]
    default_idx = int(np.argmax([r["accuracy"] for r in reports]))
    chosen = st.selectbox("Model to use for prediction", model_names, index=default_idx)

    safe = chosen.replace(" ", "_").replace("(", "").replace(")", "")
    model_path = os.path.join(MODELS_DIR, f"{safe}.h5")
    _ds = scan_dataset(data_dir)
    if "class" not in _ds.columns or _ds.empty:
        st.warning("Dataset directory not found or has no class folders. Please configure the data directory first.")
        st.stop()
    class_names = sorted(_ds["class"].unique())

    uploaded = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])
    if uploaded is not None and os.path.exists(model_path):
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 1.4])
        col1.image(image, caption="Uploaded image", use_container_width=True)

        model = load_keras_model(model_path)
        img_resized = image.resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        with st.spinner("Predicting..."):
            probs = model.predict(arr, verbose=0)[0]

        top_idx = np.argsort(probs)[::-1]
        pred_class = class_names[top_idx[0]]
        confidence = float(probs[top_idx[0]])

        with col2:
            st.metric("Predicted species", pred_class, f"{confidence*100:.1f}% confidence")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=confidence * 100,
                title={"text": "Confidence"}, gauge={"axis": {"range": [0, 100]}},
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Top-5 class probabilities")
        top5 = pd.DataFrame({
            "class": [class_names[i] for i in top_idx[:5]],
            "probability": [float(probs[i]) for i in top_idx[:5]],
        })
        fig_top5 = px.bar(top5, x="probability", y="class", orientation="h",
                           color="probability", color_continuous_scale="Blues",
                           title="Top-5 predictions")
        fig_top5.update_layout(yaxis=dict(autorange="reversed"), height=350)
        st.plotly_chart(fig_top5, use_container_width=True)

        # Session-based prediction log for extra analytics
        if "pred_log" not in st.session_state:
            st.session_state.pred_log = []
        st.session_state.pred_log.append({
            "file": uploaded.name, "model": chosen,
            "prediction": pred_class, "confidence": round(confidence, 3),
        })

    if st.session_state.get("pred_log"):
        st.subheader("Prediction history (this session)")
        log_df = pd.DataFrame(st.session_state.pred_log)
        st.dataframe(log_df, use_container_width=True)
        csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download prediction log (CSV)", csv, "prediction_log.csv", "text/csv")

    elif not os.path.exists(model_path):
        st.warning("Selected model file not found — train it first on the Model Training page.")
