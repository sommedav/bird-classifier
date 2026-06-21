import base64
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
import librosa
import librosa.display
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn

# ======================================
# CONFIG
# ======================================

# Modell-Output-Reihenfolge (passend zu data_splits/train.csv)
MODEL_CLASSES = ["Amsel", "Kohlmeise", "Rotkehlchen", "Background"]
BIRD_CLASSES = ["Amsel", "Kohlmeise", "Rotkehlchen"]
NO_BIRD_LABEL = "Kein Vogel"

CLASS_INFO = {
    "Amsel":       {"emoji": "🐦‍⬛", "color": "#2C3E50", "scientific": "Turdus merula"},
    "Kohlmeise":   {"emoji": "🐤",   "color": "#F39C12", "scientific": "Parus major"},
    "Rotkehlchen": {"emoji": "🐦",   "color": "#E74C3C", "scientific": "Erithacus rubecula"},
    "Background":  {"emoji": "🌳",   "color": "#7F8C8D", "scientific": "Hintergrund / kein Zielvogel"},
    NO_BIRD_LABEL: {"emoji": "🤷",   "color": "#95A5A6", "scientific": "Keiner der drei Vögel"},
}

# Info-Map fuer die vier Trainingsklassen (Dashboard)
MODEL_CLASS_INFO = {name: CLASS_INFO[name] for name in MODEL_CLASSES}

# BirdNET → unsere lokalen Namen (per scientific name)
BIRDNET_SCIENTIFIC_TO_LOCAL = {
    "Turdus merula":      "Amsel",
    "Parus major":        "Kohlmeise",
    "Erithacus rubecula": "Rotkehlchen",
}

# Modelle liegen in models/. Mitgeliefert (committed) werden zwei Release-Modelle,
# die sich nur in der Datenpipeline unterscheiden und im Dropdown umschaltbar sind:
#   birdcnn_release_mit_yamnet.pth  → Background per YAMNet bereinigt (empfohlen, Default)
#   birdcnn_release_ohne_yamnet.pth → ältere Pipeline ohne YAMNet (librosa-Heuristik)
# Selbst trainierte Checkpoints (birdcnn_<timestamp>_best.pth) landen ebenfalls hier
# und sind waehlbar. BIRD_MODEL_PATH ueberschreibt die Auswahl mit einem festen Pfad.
MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL = "birdcnn_release_mit_yamnet.pth"
MODEL_LABELS = {
    "birdcnn_release_mit_yamnet.pth": "BirdCNN · mit YAMNet  ⭐ (empfohlen)",
    "birdcnn_release_ohne_yamnet.pth": "BirdCNN · ohne YAMNet (Legacy)",
}
ENV_MODEL_PATH = os.environ.get("BIRD_MODEL_PATH")


def discover_models() -> list[Path]:
    """Alle .pth in models/ — empfohlenes Modell zuerst, danach neueste zuerst."""
    if not MODELS_DIR.is_dir():
        return []
    paths = list(MODELS_DIR.glob("*.pth"))
    return sorted(
        paths,
        key=lambda p: (0 if p.name == DEFAULT_MODEL else 1, -p.stat().st_mtime),
    )


def model_label(p: Path) -> str:
    """Anzeigename im Dropdown."""
    return MODEL_LABELS.get(p.name, p.stem)

TARGET_SR = 32000
TARGET_DURATION = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
TARGET_FRAMES = 313
HOP_LENGTH = 512

# ======================================
# TRAINING-METRIKEN
# Quelle: project.md / notebooks/bird_training.ipynb (Evaluation auf dem Test-Set).
# Diese Zahlen NICHT erfinden oder schaetzen — sie stammen 1:1 aus project.md.
# ======================================

METRICS = {
    "test_acc": 87.87,
    "val_acc": 87.32,
    "n_train": 8760,
    "n_val": 1783,
    "n_test": 2086,
    "n_total": 12629,
    "epochs": 20,
    "batch_size": 32,
}

# Pro Klasse: Accuracy %, Support, Precision, Recall, F1 (aus project.md, Abschnitt 7)
PER_CLASS = {
    "Amsel":       {"acc": 93.62, "support": 580, "precision": 0.949, "recall": 0.936, "f1": 0.943},
    "Kohlmeise":   {"acc": 86.63, "support": 673, "precision": 0.922, "recall": 0.866, "f1": 0.893},
    "Rotkehlchen": {"acc": 88.38, "support": 327, "precision": 0.732, "recall": 0.884, "f1": 0.801},
    "Background":  {"acc": 82.61, "support": 506, "precision": 0.858, "recall": 0.826, "f1": 0.842},
}

MACRO_AVG = {"precision": 0.865, "recall": 0.878, "f1": 0.870}
WEIGHTED_AVG = {"precision": 0.884, "recall": 0.879, "f1": 0.880}

# Confusion Matrix (Zeilen = wahr, Spalten = vorhergesagt), Reihenfolge = MODEL_CLASSES
CONFUSION_MATRIX = np.array([
    [543,   0,  10,  27],
    [  6, 583,  63,  21],
    [  1,  16, 289,  21],
    [ 22,  33,  33, 418],
])

# ======================================
# CNN MODEL  (V3 — passend zu bird_training.ipynb)
# ======================================

class SpecAugment(nn.Module):
    """Im Training aktiv — in der App nur Stub, da model.eval(). Keine Parameter."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class BirdCNN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()
        self.spec_aug = SpecAugment()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# ======================================
# CACHED LOADERS
# ======================================

@st.cache_resource
def load_model(model_path: str):
    m = BirdCNN(num_classes=len(MODEL_CLASSES))
    m.load_state_dict(torch.load(model_path, map_location="cpu"))
    m.eval()
    return m

def birdnetlib_installed() -> bool:
    """Leichter Check — initialisiert nichts."""
    return importlib.util.find_spec("birdnetlib") is not None

# BirdNET laeuft in einem eigenen Subprozess. Grund: PyTorch und das von BirdNET
# genutzte TensorFlow-Lite vertragen sich im selben Prozess nicht zuverlaessig.
_BIRDNET_SUBPROCESS_SCRIPT = textwrap.dedent("""
    import sys, json, contextlib
    # Alle Logs/Prints des Analyzers nach stderr — stdout bleibt sauber fürs JSON.
    with contextlib.redirect_stdout(sys.stderr):
        from birdnetlib import Recording
        from birdnetlib.analyzer import Analyzer
        analyzer = Analyzer()
        rec = Recording(analyzer, sys.argv[1], min_conf=0.05)
        rec.analyze()
        detections = rec.detections
    sys.stdout.write(json.dumps(detections))
""")

@st.cache_data(show_spinner=False)
def load_full_audio(audio_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    y, sr = librosa.load(path, sr=TARGET_SR)
    return y, sr, path

@st.cache_data(show_spinner=False)
def compute_mel(audio_bytes: bytes):
    y, sr, _ = load_full_audio(audio_bytes)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=16000, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max)

@st.cache_data(show_spinner="BirdNET analysiert (Subprocess — kann beim ersten Mal etwas dauern)...")
def run_birdnet_full(audio_bytes: bytes):
    """BirdNET einmal auf die volle Aufnahme — als Subprocess (s. _BIRDNET_SUBPROCESS_SCRIPT)."""
    if not birdnetlib_installed():
        return None

    _, _, full_path = load_full_audio(audio_bytes)

    try:
        result = subprocess.run(
            [sys.executable, "-c", _BIRDNET_SUBPROCESS_SCRIPT, full_path],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        st.warning("BirdNET-Subprocess hat das Zeitlimit überschritten.")
        return None

    if result.returncode != 0:
        st.warning(
            "BirdNET-Subprocess ist fehlgeschlagen. "
            f"stderr (gekürzt): `{result.stderr.strip()[-400:]}`"
        )
        return None

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        st.warning(
            "BirdNET-Output konnte nicht geparst werden. "
            f"stdout: `{result.stdout[:300]}`"
        )
        return None


@st.cache_data(show_spinner=False)
def confusion_matrix_png() -> bytes:
    """Confusion Matrix als PNG — gecached, damit sie nicht bei jedem Rerun neu gebaut wird."""
    cm = CONFUSION_MATRIX
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm_norm, cmap="Purples", vmin=0, vmax=1)

    ax.set_xticks(range(len(MODEL_CLASSES)))
    ax.set_xticklabels(MODEL_CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(MODEL_CLASSES)))
    ax.set_yticklabels(MODEL_CLASSES)
    ax.set_xlabel("Vorhergesagt", fontweight="bold")
    ax.set_ylabel("Wahr", fontweight="bold")

    for i in range(len(MODEL_CLASSES)):
        for j in range(len(MODEL_CLASSES)):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "#2c3e50",
                fontweight="bold", fontsize=11,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Anteil pro wahrer Klasse")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@st.cache_data(show_spinner="Notebook wird gerendert ...")
def render_notebook_html():
    """bird_training.ipynb → HTML (nbconvert). None, wenn nicht moeglich."""
    nb_path = Path(__file__).resolve().parent / "notebooks" / "bird_training.ipynb"
    if not nb_path.exists():
        return None
    try:
        import nbformat
        from nbconvert import HTMLExporter

        nb = nbformat.read(nb_path, as_version=4)
        exporter = HTMLExporter()
        exporter.exclude_input = True          # Code-Zellen ausblenden
        exporter.exclude_input_prompt = True
        exporter.exclude_output_prompt = True
        body, _ = exporter.from_notebook_node(nb)
        return body
    except Exception:
        return None


def detections_in_window(detections, start_sec: float, end_sec: float):
    """BirdNET chunked in 3s; behalte alles was sich mit [start, end] überschneidet."""
    if not detections:
        return []
    return [
        d for d in detections
        if d.get("start_time", 0.0) < end_sec
        and d.get("end_time", 0.0) > start_sec
    ]

# ======================================
# AUDIO HELPERS
# ======================================

def crop_audio(y: np.ndarray, sr: int, start_sec: float) -> np.ndarray:
    start_sample = int(start_sec * sr)
    end_sample = start_sample + TARGET_SAMPLES
    cropped = y[start_sample:end_sample]
    if len(cropped) < TARGET_SAMPLES:
        cropped = np.pad(cropped, (0, TARGET_SAMPLES - len(cropped)), mode="constant")
    return cropped

def audio_to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()

def array_to_mel_for_model(y: np.ndarray, sr: int) -> np.ndarray:
    """Genau wie im Notebook: 313 Frames fixed + (mean=0, std=1) Normalisierung."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=16000, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    n = mel_db.shape[1]
    if n < TARGET_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, TARGET_FRAMES - n)), mode="constant")
    elif n > TARGET_FRAMES:
        mel_db = mel_db[:, :TARGET_FRAMES]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db

# ======================================
# PREDICTION HELPERS
# ======================================

def predict_my_model(model, y_cropped: np.ndarray, sr: int) -> np.ndarray:
    mel = array_to_mel_for_model(y_cropped, sr)
    t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1).numpy()[0]
    return probs  # Reihenfolge: MODEL_CLASSES

def my_model_display_probs(probs: np.ndarray) -> dict:
    """MODEL_CLASSES-Probs → dict mit Anzeige-Labels (Background → Kein Vogel)."""
    p = dict(zip(MODEL_CLASSES, probs.tolist()))
    p[NO_BIRD_LABEL] = p.pop("Background")
    return p

def birdnet_display_probs(detections) -> dict:
    """Max-Confidence pro Spezies + Rest als 'Kein Vogel'."""
    bird_max = {b: 0.0 for b in BIRD_CLASSES}
    if detections:
        for d in detections:
            sci = d.get("scientific_name", "")
            local = BIRDNET_SCIENTIFIC_TO_LOCAL.get(sci)
            if local:
                bird_max[local] = max(bird_max[local], float(d.get("confidence", 0.0)))
    max_bird_conf = max(bird_max.values()) if bird_max else 0.0
    return {**bird_max, NO_BIRD_LABEL: max(0.0, 1.0 - max_bird_conf)}

def birdnet_top_detection(detections):
    """Top-Detection (egal welche Spezies) → (display_name, scientific, conf, is_in_our_3) oder None."""
    if not detections:
        return None
    top = max(detections, key=lambda d: d.get("confidence", 0.0))
    sci = top.get("scientific_name", "")
    common = top.get("common_name", sci)
    conf = float(top.get("confidence", 0.0))
    if sci in BIRDNET_SCIENTIFIC_TO_LOCAL:
        return (BIRDNET_SCIENTIFIC_TO_LOCAL[sci], sci, conf, True)
    return (common, sci, conf, False)

# ======================================
# PAGE CONFIG + CSS
# ======================================

st.set_page_config(
    page_title="Bird Classifier",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    /* Modell-Sidebar dauerhaft ausgeklappt: Collapse-/Expand-Buttons entfernen und
       die Sidebar sichtbar erzwingen, damit sie nie eingeklappt hängen bleibt. */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] {
        visibility: visible !important;
        transform: none !important;
        min-width: 244px !important;
    }
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    /* Theme-adaptive Farben — funktionieren in Light- UND Dark-Mode */
    :root {
        --title-accent: #4754c1;   /* Titel-Akzent (Indigo) */
        --title-border: #d9ddee;
        --muted-text:   #5b6472;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --title-accent: #aab6ff;
            --title-border: #3a3f55;
            --muted-text:   #aab2c5;
        }
    }

    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
           margin-right: 8px; vertical-align: middle; }

    .hero-img {
        position: relative; border-radius: 20px; overflow: hidden;
        margin-bottom: 1.5rem; min-height: 270px;
        display: flex; align-items: flex-end;
        background-size: cover; background-position: center 35%;
        box-shadow: 0 12px 34px rgba(0,0,0,0.28);
    }
    .hero-img::before { content: ""; position: absolute; inset: 0;
        background: linear-gradient(180deg, rgba(12,15,35,0.05) 0%,
                    rgba(12,15,35,0.28) 45%, rgba(12,15,35,0.82) 100%); }
    .hero-img .hero-content { position: relative; z-index: 1; padding: 1.7rem 2rem;
        color: #fff; width: 100%; }
    .hero-img h1 { font-size: 2.7rem; margin: 0; font-weight: 800; letter-spacing: -0.5px;
        text-shadow: 0 2px 14px rgba(0,0,0,0.5); }
    .hero-img p  { margin: 0.45rem 0 0; font-size: 1.15rem; opacity: 0.97;
        text-shadow: 0 1px 10px rgba(0,0,0,0.55); }

    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem; border-radius: 20px; color: white;
        text-align: center; margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.25);
    }
    .hero h1 { font-size: 2.6rem; margin: 0; font-weight: 700; letter-spacing: -0.5px; }
    .hero p  { font-size: 1.1rem; opacity: 0.95; margin: 0.5rem 0 0 0; }

    .card {
        background: white; padding: 1.5rem; border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #eef0f3;
        margin-bottom: 1rem;
    }

    .result-box {
        background: linear-gradient(135deg, #f6f9fc 0%, #eef2f7 100%);
        padding: 1.6rem 1.2rem; border-radius: 16px;
        text-align: center; border: 2px solid #e6ebf1;
        margin-bottom: 1rem;
    }
    .result-tag        { font-size: 0.75rem; font-weight: 700; letter-spacing: 1.5px;
                         text-transform: uppercase; color: #95a5a6; margin-bottom: 0.4rem; }
    .result-emoji      { font-size: 3.2rem; line-height: 1; margin-bottom: 0.3rem; }
    .result-name       { font-size: 1.6rem; font-weight: 700; color: #2c3e50; margin: 0; }
    .result-scientific { font-style: italic; color: #7f8c8d; margin: 0.2rem 0 0.8rem 0; font-size: 0.9rem; }
    .result-conf-label { font-size: 0.75rem; color: #95a5a6; text-transform: uppercase; letter-spacing: 1px; }
    .result-conf-val   { font-size: 1.5rem; font-weight: 700; color: #27ae60; }

    .result-tag.mine    { color: #667eea; }
    .result-tag.birdnet { color: #16a085; }

    .prob-row { display: flex; align-items: center; gap: 12px; padding: 8px 0; }
    .prob-label   { flex: 0 0 145px; font-weight: 600; color: #34495e; font-size: 0.95rem; }
    .prob-bar-bg  { flex: 1; background: #ecf0f1; height: 20px; border-radius: 10px; overflow: hidden; }
    .prob-bar-fill{ height: 100%; border-radius: 10px; transition: width 0.4s ease; }
    .prob-value   { flex: 0 0 60px; text-align: right; font-weight: 700;
                    font-variant-numeric: tabular-nums; color: #2c3e50; }

    .section-title {
        font-size: 1.1rem; font-weight: 700; color: var(--title-accent);
        margin: 1.5rem 0 0.8rem 0; padding-bottom: 0.4rem;
        border-bottom: 2px solid var(--title-border);
    }

    .compare-col-title {
        font-size: 1rem; font-weight: 700;
        padding: 0.5rem 0.8rem; border-radius: 8px;
        margin-bottom: 0.6rem; text-align: center;
    }
    .compare-col-title.mine    { background: #eef0fb; color: #4754c1; }
    .compare-col-title.birdnet { background: #e8f7f2; color: #16a085; }

    /* ---- Dashboard / Modell-Insights ---- */
    .muted { color: var(--muted-text); font-size: 0.92rem; margin: 0 0 1rem 0; }

    .metric-card {
        background: white; border-radius: 16px; padding: 1.2rem 1rem;
        text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #eef0f3; height: 100%;
    }
    .metric-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px;
                    color: #95a5a6; font-weight: 700; }
    .metric-value { font-size: 2.1rem; font-weight: 800; margin: 0.3rem 0; line-height: 1; }
    .metric-sub   { font-size: 0.74rem; color: #7f8c8d; }

    .metrics-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .metrics-table th { background: #f6f9fc; color: #4754c1; padding: 0.55rem 0.4rem;
                        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .metrics-table td { padding: 0.5rem 0.4rem; text-align: center; color: #2c3e50;
                        border-bottom: 1px solid #eef0f3; font-variant-numeric: tabular-nums; }
    .metrics-table tr.avg td { background: #fafbfc; font-weight: 700; color: #2c3e50; }

    .spec-row { display: flex; justify-content: space-between; gap: 1rem;
                padding: 0.55rem 0; border-bottom: 1px solid #f0f2f5; font-size: 0.92rem; }
    .spec-row:last-child { border-bottom: none; }
    .spec-row span { color: #7f8c8d; }
    .spec-row b    { color: #2c3e50; text-align: right; }

    .info-banner { display: flex; align-items: center; gap: 1rem;
        background: linear-gradient(135deg, #f6f9fc, #eef2f7);
        padding: 1.2rem 1.5rem; border-radius: 16px;
        border: 1px solid #e6ebf1; margin-bottom: 1.5rem; }

    /* ---- Projekt-Tab (CRISP-DM Cards) ---- */
    .proj-hero {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem 1.8rem; border-radius: 20px; color: white;
        margin-bottom: 1.4rem; box-shadow: 0 10px 30px rgba(17, 153, 142, 0.22);
    }
    .proj-hero h2 { margin: 0; font-size: 1.7rem; font-weight: 800; letter-spacing: -0.4px; }
    .proj-hero p  { margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.95; max-width: 780px; }

    .pill-row { margin: 0.1rem 0 0.4rem 0; line-height: 2.1; }
    .pill { display: inline-block; background: #eef0fb; color: #4754c1; font-weight: 600;
        font-size: 0.82rem; padding: 0.32rem 0.85rem; border-radius: 999px;
        margin: 0 0.35rem 0.1rem 0; border: 1px solid #e0e4f7; }

    .proj-section-head { display: flex; align-items: center; gap: 0.75rem; margin: 0.1rem 0 0.4rem 0; }
    .proj-badge { width: 30px; height: 30px; border-radius: 50%; color: white;
        font-weight: 800; font-size: 0.9rem; display: flex; align-items: center;
        justify-content: center; flex: 0 0 30px; box-shadow: 0 3px 10px rgba(0,0,0,0.12); }
    .proj-title { font-size: 1.18rem; font-weight: 700; color: var(--title-accent); }
</style>
""", unsafe_allow_html=True)

# ======================================
# HEADER  (Foto-Hero; faellt auf Farbverlauf zurueck, wenn kein Bild vorhanden)
# ======================================

# Lege das Titelbild hier ab: assets/hero_birds.jpg (oder .png). Fehlt es,
# wird automatisch der Farbverlauf-Header genutzt.
HERO_IMAGE_DIR = Path(__file__).resolve().parent / "assets"
HERO_TITLE = "Bird Species Classifier"
HERO_SUBTITLE = "Mein CNN gegen BirdNET — wer erkennt den Vogel besser?"


def _hero_image_path() -> Path | None:
    for name in ("hero_birds.jpg", "hero_birds.jpeg", "hero_birds.png", "hero_birds.webp"):
        p = HERO_IMAGE_DIR / name
        if p.exists():
            return p
    return None


def _render_hero():
    img = _hero_image_path()
    if img is not None:
        mime = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
        }.get(img.suffix.lower(), "image/jpeg")
        b64 = base64.b64encode(img.read_bytes()).decode()
        st.markdown(
            f'<div class="hero-img" style="background-image:'
            f'url(data:{mime};base64,{b64});">'
            f'<div class="hero-content"><h1>{HERO_TITLE}</h1>'
            f'<p>{HERO_SUBTITLE}</p></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="hero"><h1>{HERO_TITLE}</h1><p>{HERO_SUBTITLE}</p></div>',
            unsafe_allow_html=True,
        )


_render_hero()

# ======================================
# MODELL-AUSWAHL + LADEN (mit Fehlerbehandlung)
# ======================================

st.sidebar.markdown("## Modell")

# BIRD_MODEL_PATH (falls gesetzt) hat Vorrang vor der manuellen Auswahl.
if ENV_MODEL_PATH:
    selected_path = ENV_MODEL_PATH
    st.sidebar.info(f"Modell via `BIRD_MODEL_PATH`:\n\n`{ENV_MODEL_PATH}`")
else:
    available_models = discover_models()
    if available_models:
        st.sidebar.caption("Aktives Modell auswählen:")
        choice = st.sidebar.radio(
            "Aktives Modell",
            options=available_models,
            format_func=model_label,
            label_visibility="collapsed",
        )
        selected_path = str(choice)
        st.sidebar.caption(
            f"{len(available_models)} Modell(e) in `models/`. "
            "Eigene Trainings (`birdcnn_<timestamp>_best.pth`) erscheinen hier automatisch."
        )
    else:
        # Kein Modell gefunden — Fallback-Pfad fuer eine klare Fehlermeldung.
        selected_path = str(MODELS_DIR / DEFAULT_MODEL)

try:
    model = load_model(selected_path)
    model_error = None
except Exception as e:
    model = None
    model_error = str(e)

if model is None:
    st.error(
        f"Mein Modell konnte nicht geladen werden.\n\n"
        f"Pfad: `{selected_path}`\n\n"
        f"Fehler: `{model_error}`\n\n"
        f"Erwartet wird `models/{DEFAULT_MODEL}`. Trainiere alternativ das Notebook "
        f"`notebooks/bird_training.ipynb` durch — das legt einen Checkpoint in `models/` ab."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(f"Aktiv: `{Path(selected_path).name}`")


# ======================================
# TAB: MODELL & TRAINING (Dashboard)
# ======================================

def render_model_insights():
    st.markdown('<div class="section-title" style="margin-top:0;">Modell-Steckbrief</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">Alle Kennzahlen stammen direkt aus dem Trainings-Notebook '
        '<code>notebooks/bird_training.ipynb</code> bzw. <code>project.md</code> — '
        'gemessen auf dem zuvor unberührten Test-Set.</p>',
        unsafe_allow_html=True,
    )

    # ---- Headline-Metriken ----
    n_total = f"{METRICS['n_total']:,}".replace(",", ".")
    split_sub = (
        f"Train {METRICS['n_train']:,} · Val {METRICS['n_val']:,} · Test {METRICS['n_test']:,}"
    ).replace(",", ".")
    metric_cards = [
        ("Test-Accuracy", f"{METRICS['test_acc']:.2f} %", "2.086 ungesehene Clips", "#27ae60"),
        ("Val-Accuracy", f"{METRICS['val_acc']:.2f} %", "bester Checkpoint", "#667eea"),
        ("Klassen", "4", "3 Vögel + Background", "#16a085"),
        ("Clips gesamt", n_total, split_sub, "#e67e22"),
    ]
    cols = st.columns(4)
    for col, (label, val, sub, color) in zip(cols, metric_cards):
        col.markdown(f"""
        <div class="metric-card" style="border-top:4px solid {color};">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Pro-Klasse Accuracy ----
    st.markdown('<div class="section-title">Genauigkeit pro Klasse</div>', unsafe_allow_html=True)
    rows = []
    for name, m in PER_CLASS.items():
        info = MODEL_CLASS_INFO[name]
        pct = m["acc"]
        rows.append(
            f'<div class="prob-row">'
            f'<div class="prob-label"><span class="dot" style="background:{info["color"]};"></span>{name}</div>'
            f'<div class="prob-bar-bg">'
            f'<div class="prob-bar-fill" style="width:{pct}%; background:{info["color"]};"></div>'
            f'</div>'
            f'<div class="prob-value">{pct:.2f}%</div>'
            f'</div>'
        )
    st.markdown(f'<div class="card">{"".join(rows)}</div>', unsafe_allow_html=True)

    # ---- Classification Report + Confusion Matrix ----
    col_rep, col_cm = st.columns(2, gap="large")

    with col_rep:
        st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
        header = ("<tr><th style='text-align:left;'>Klasse</th><th>Precision</th>"
                  "<th>Recall</th><th>F1</th><th>Support</th></tr>")
        body = ""
        for name, m in PER_CLASS.items():
            info = MODEL_CLASS_INFO[name]
            body += (
                f"<tr><td style='text-align:left;'>"
                f"<span class='dot' style='background:{info['color']};'></span>{name}</td>"
                f"<td>{m['precision']:.3f}</td><td>{m['recall']:.3f}</td>"
                f"<td>{m['f1']:.3f}</td><td>{m['support']}</td></tr>"
            )
        body += (
            f"<tr class='avg'><td style='text-align:left;'>macro avg</td>"
            f"<td>{MACRO_AVG['precision']:.3f}</td><td>{MACRO_AVG['recall']:.3f}</td>"
            f"<td>{MACRO_AVG['f1']:.3f}</td><td>{METRICS['n_test']}</td></tr>"
        )
        body += (
            f"<tr class='avg'><td style='text-align:left;'>weighted avg</td>"
            f"<td>{WEIGHTED_AVG['precision']:.3f}</td><td>{WEIGHTED_AVG['recall']:.3f}</td>"
            f"<td>{WEIGHTED_AVG['f1']:.3f}</td><td>{METRICS['n_test']}</td></tr>"
        )
        st.markdown(
            f'<div class="card"><table class="metrics-table">{header}{body}</table></div>',
            unsafe_allow_html=True,
        )

    with col_cm:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        st.image(confusion_matrix_png(), use_container_width=True)
        st.caption(
            "Zeilen = wahre Klasse, Spalten = Vorhersage. "
            "Häufigste Verwechslung: Kohlmeise → Rotkehlchen (63)."
        )

    # ---- Architektur + Hyperparameter ----
    col_arch, col_hyper = st.columns(2, gap="large")

    with col_arch:
        st.markdown('<div class="section-title">Architektur — BirdCNN</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <div class="spec-row"><span>Input</span><b>1 × 128 × 313 (Mel-Spektrogramm)</b></div>
            <div class="spec-row"><span>Conv-Kanäle</span><b>1 → 32 → 64 → 128 → 256</b></div>
            <div class="spec-row"><span>Block-Aufbau</span><b>2× (Conv 3×3 → BN → ReLU) → MaxPool</b></div>
            <div class="spec-row"><span>Pooling</span><b>Global Average Pooling</b></div>
            <div class="spec-row"><span>Kopf</span><b>Dropout 0.5 → Linear 256 → 4</b></div>
            <div class="spec-row"><span>Augmentation</span><b>SpecAugment (Freq/Zeit-Masking)</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_hyper:
        st.markdown('<div class="section-title">Training-Setup</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
            <div class="spec-row"><span>Epochen</span><b>{METRICS['epochs']}</b></div>
            <div class="spec-row"><span>Batch-Size</span><b>{METRICS['batch_size']}</b></div>
            <div class="spec-row"><span>Loss</span><b>CrossEntropy (Label Smoothing 0.1)</b></div>
            <div class="spec-row"><span>Optimizer</span><b>AdamW (lr 1e-3, wd 1e-4)</b></div>
            <div class="spec-row"><span>Scheduler</span><b>CosineAnnealing</b></div>
            <div class="spec-row"><span>Checkpoint</span><b>bester Val-Acc → models/…_best.pth</b></div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Optional: volles Notebook rendern ----
    st.markdown('<div class="section-title">Trainings-Notebook</div>', unsafe_allow_html=True)
    with st.expander("Notebook-Ergebnisse anzeigen (Plots & Ausgaben aus bird_training.ipynb, ohne Code)"):
        if st.button("Ergebnisse aus bird_training.ipynb laden"):
            html = render_notebook_html()
            if html is None:
                st.warning(
                    "Notebook konnte nicht gerendert werden — `nbconvert`/`nbformat` fehlt "
                    "oder `notebooks/bird_training.ipynb` wurde nicht gefunden."
                )
            else:
                components.html(html, height=820, scrolling=True)


# ======================================
# TAB: ÜBER DAS PROJEKT (project.md)
# ======================================

# Akzentfarbe pro CRISP-DM-Abschnitt (Match per Schlagwort im Titel)
_SECTION_STYLE = [
    ("idea",               "#f39c12"),
    ("business",           "#e74c3c"),
    ("data understanding", "#3498db"),
    ("related",            "#9b59b6"),
    ("preparation",        "#16a085"),
    ("modeling",           "#667eea"),
    ("evaluation",         "#27ae60"),
    ("deployment",         "#e67e22"),
    ("reflection",         "#34495e"),
]


def _parse_md_sections(md_text: str):
    """project.md → Liste von (titel, body) anhand der '## '-Überschriften."""
    sections, title, body = [], None, []
    for line in md_text.splitlines():
        if line.startswith("## "):
            if title is not None:
                sections.append((title, "\n".join(body).strip()))
            title, body = line[3:].strip(), []
        elif line.startswith("# "):
            continue  # H1 überspringen — wir haben einen eigenen Hero
        elif title is not None:
            body.append(line)
    if title is not None:
        sections.append((title, "\n".join(body).strip()))
    return sections


def _section_color(title: str) -> str:
    t = title.lower()
    for key, color in _SECTION_STYLE:
        if key in t:
            return color
    return "#7f8c8d"


def render_project_info():
    # ---- Hero ----
    st.markdown("""
    <div class="proj-hero">
        <h2>Bird Species Classifier — ML4B</h2>
        <p>Ein Proof of Concept, das Vogelgesang aus 5-Sekunden-Aufnahmen erkennt:
        Amsel, Kohlmeise und Rotkehlchen — plus eine Background-Klasse für „kein Zielvogel".
        Unten die ganze Projekt-Story entlang des CRISP-DM-Prozesses.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Quick Facts ----
    facts = [
        ("Zielklassen", "3 + 1", "Amsel · Kohlmeise · Rotkehlchen · Background", "#667eea"),
        ("Datensatz", f"{METRICS['n_total']:,}".replace(",", "."), "Clips von Xeno-Canto", "#16a085"),
        ("Test-Accuracy", f"{METRICS['test_acc']:.2f} %", "auf ungesehenen Clips", "#27ae60"),
        ("Vergleich", "BirdNET", "Referenzsystem in der App", "#e67e22"),
    ]
    cols = st.columns(4)
    for col, (label, val, sub, color) in zip(cols, facts):
        col.markdown(f"""
        <div class="metric-card" style="border-top:4px solid {color};">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- CRISP-DM Abschnitte als Cards ----
    project_md = Path(__file__).resolve().parent / "project.md"
    if not project_md.exists():
        st.warning("`project.md` wurde nicht gefunden.")
        return

    st.markdown('<div class="section-title">Die Projekt-Story (CRISP-DM)</div>', unsafe_allow_html=True)
    sections = _parse_md_sections(project_md.read_text(encoding="utf-8"))
    for i, (title, body) in enumerate(sections, start=1):
        color = _section_color(title)
        # Führende Nummer aus dem Titel (z. B. "7. Evaluation") für das Badge nutzen
        num = title.split(".", 1)[0].strip() if title[:1].isdigit() else str(i)
        name = title.split(".", 1)[1].strip() if "." in title and title[:1].isdigit() else title
        with st.container(border=True):
            st.markdown(
                f'<div class="proj-section-head">'
                f'<span class="proj-badge" style="background:{color};">{num}</span>'
                f'<span class="proj-title">{name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(body)

    # ---- Tech-Stack Pills (am Ende) ----
    st.markdown('<div class="section-title">Bausteine</div>', unsafe_allow_html=True)
    tools = ["Xeno-Canto", "YAMNet", "BirdNET", "librosa", "PyTorch", "scikit-learn", "Streamlit"]
    pills = "".join(f'<span class="pill">{t}</span>' for t in tools)
    st.markdown(f'<div class="pill-row">{pills}</div>', unsafe_allow_html=True)


# ======================================
# TAB: KLASSIFIZIEREN
# ======================================

def render_classifier():
    # ---- Input ----
    tab_upload, tab_record = st.tabs(["Datei hochladen", "Live aufnehmen"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "WAV-Datei", type=["wav"], label_visibility="collapsed"
        )

    with tab_record:
        recorded = st.audio_input("Drück den Knopf und nimm den Vogel auf")

    audio_source = uploaded_file if uploaded_file is not None else recorded

    # ---- BirdNET-Vergleich Toggle ----
    if birdnetlib_installed():
        birdnet_enabled = st.checkbox(
            "BirdNET-Vergleich aktiv",
            value=True,
            help="Beim ersten Mal pro Datei dauert die Analyse ~5–30s. "
                 "Danach reagiert der Slider sofort (Detektionen werden nur noch gefiltert).",
        )
    else:
        birdnet_enabled = False

    # ---- Kein Input → Species Cards ----
    if audio_source is None:
        st.markdown('<div class="section-title">Erkennbare Arten</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for col, name in zip(cols, BIRD_CLASSES):
            info = CLASS_INFO[name]
            with col:
                st.markdown(f"""
                <div class="card" style="text-align:center;">
                    <div style="font-size:3.5rem; line-height:1;">{info['emoji']}</div>
                    <div style="font-weight:700; font-size:1.2rem; color:{info['color']}; margin-top:0.5rem;">{name}</div>
                    <div style="font-style:italic; color:#7f8c8d; font-size:0.9rem;">{info['scientific']}</div>
                </div>
                """, unsafe_allow_html=True)

        if not birdnetlib_installed():
            st.info(
                "**BirdNET nicht aktiv.** `birdnetlib` ist nicht installiert. "
                "Abhängigkeiten synchronisieren, damit die App beide Modelle "
                "nebeneinander vergleichen kann:  \n"
                "```bash\nuv sync\n```"
            )
        return

    # ---- Audio Input → Analyse ----
    audio_bytes = audio_source.getvalue() if hasattr(audio_source, "getvalue") else audio_source.read()

    y_full, sr, _ = load_full_audio(audio_bytes)
    full_duration = len(y_full) / sr
    mel_full = compute_mel(audio_bytes)

    # ---- Volle Aufnahme ----
    st.markdown('<div class="section-title">Aufnahme</div>', unsafe_allow_html=True)
    st.audio(audio_bytes)
    st.caption(f"Länge: {full_duration:.2f}s")

    # ---- Slider ----
    if full_duration > TARGET_DURATION:
        st.markdown(
            '<div class="section-title">Wähle den 5-Sekunden-Ausschnitt</div>',
            unsafe_allow_html=True,
        )
        start_sec = st.slider(
            "Startzeit",
            min_value=0.0,
            max_value=float(full_duration - TARGET_DURATION),
            value=0.0,
            step=0.1,
            format="%.1f s",
            label_visibility="collapsed",
        )
        st.caption(f"Analysefenster: **{start_sec:.1f}s – {start_sec + TARGET_DURATION:.1f}s**")
    else:
        start_sec = 0.0
        if full_duration < TARGET_DURATION:
            st.info(f"Aufnahme ist nur {full_duration:.2f}s — wird auf 5s mit Stille aufgefüllt.")

    # ---- Crop ----
    y_cropped = crop_audio(y_full, sr, start_sec)
    cropped_wav = audio_to_wav_bytes(y_cropped, sr)

    # ---- Layout ----
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Mel-Spektrogramm</div>', unsafe_allow_html=True)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1e1e2e")
        ax.set_facecolor("#1e1e2e")

        librosa.display.specshow(
            mel_full, sr=sr, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel", cmap="magma", ax=ax, fmax=16000
        )

        rect = patches.Rectangle(
            (start_sec, 0), TARGET_DURATION, sr // 2,
            linewidth=2.5, edgecolor="#2ecc71", facecolor="#2ecc71",
            alpha=0.25, zorder=10,
        )
        ax.add_patch(rect)
        ax.axvline(start_sec, color="#2ecc71", linewidth=2, zorder=11)
        ax.axvline(start_sec + TARGET_DURATION, color="#2ecc71", linewidth=2, zorder=11)

        ax.set_xlabel("Zeit (s)", color="white")
        ax.set_ylabel("Mel-Frequenz", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div class="section-title">Nur der Ausschnitt</div>', unsafe_allow_html=True)
        st.audio(cropped_wav, format="audio/wav")

    # ---- Mein Modell ----
    my_raw_probs = predict_my_model(model, y_cropped, sr)
    my_probs = my_model_display_probs(my_raw_probs)
    my_top_label = max(my_probs, key=my_probs.get)
    my_top_conf = my_probs[my_top_label] * 100
    my_top_info = CLASS_INFO[my_top_label]

    # ---- BirdNET (volle Aufnahme einmal analysieren, dann pro Fenster filtern) ----
    if birdnet_enabled and birdnetlib_installed():
        all_detections = run_birdnet_full(audio_bytes)
        birdnet_available = all_detections is not None
        birdnet_detections = (
            detections_in_window(all_detections, start_sec, start_sec + TARGET_DURATION)
            if birdnet_available else None
        )
    else:
        birdnet_available = False
        birdnet_detections = None

    bn_probs = birdnet_display_probs(birdnet_detections) if birdnet_available else None
    bn_top = birdnet_top_detection(birdnet_detections) if birdnet_available else None

    # ---- Result Cards (rechts) ----
    with col_right:
        st.markdown('<div class="section-title">Vergleich</div>', unsafe_allow_html=True)

        # ---- MEIN MODELL ----
        st.markdown(f"""
        <div class="result-box" style="border-top:5px solid {my_top_info['color']};">
            <div class="result-tag mine">Mein Modell</div>
            <div class="result-name">{my_top_label}</div>
            <div class="result-scientific">{my_top_info['scientific']}</div>
            <div class="result-conf-label">Konfidenz</div>
            <div class="result-conf-val">{my_top_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ---- BIRDNET ----
        if not birdnet_available:
            st.markdown(f"""
            <div class="result-box">
                <div class="result-tag birdnet">BirdNET</div>
                <div class="result-name" style="font-size:1.1rem;">birdnetlib nicht installiert</div>
                <div class="result-scientific">uv sync</div>
            </div>
            """, unsafe_allow_html=True)
        elif bn_top is None:
            st.markdown(f"""
            <div class="result-box">
                <div class="result-tag birdnet">BirdNET</div>
                <div class="result-name">{NO_BIRD_LABEL}</div>
                <div class="result-scientific">{CLASS_INFO[NO_BIRD_LABEL]['scientific']}</div>
                <div class="result-conf-label">Keine Detektion</div>
                <div class="result-conf-val" style="color:#95a5a6;">—</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            bn_display_name, bn_sci, bn_conf, is_ours = bn_top
            if is_ours:
                bn_info = CLASS_INFO[bn_display_name]
                accent = bn_info["color"]
                sci = bn_info["scientific"]
            else:
                accent = "#16a085"
                sci = bn_sci or "andere Art"
            st.markdown(f"""
            <div class="result-box" style="border-top:5px solid {accent};">
                <div class="result-tag birdnet">BirdNET</div>
                <div class="result-name">{bn_display_name}</div>
                <div class="result-scientific">{sci}</div>
                <div class="result-conf-label">Konfidenz</div>
                <div class="result-conf-val">{bn_conf * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Wahrscheinlichkeiten im Vergleich (volle Breite) ----
    st.markdown(
        '<div class="section-title">Wahrscheinlichkeiten im Vergleich</div>',
        unsafe_allow_html=True,
    )

    def render_prob_bars(probs_dict):
        items = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)
        rows = []
        for name, p in items:
            info = CLASS_INFO[name]
            pct = p * 100
            rows.append(f"""
            <div class="prob-row">
                <div class="prob-label"><span class="dot" style="background:{info['color']};"></span>{name}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct}%; background:{info['color']};"></div>
                </div>
                <div class="prob-value">{pct:.1f}%</div>
            </div>
            """)
        return "".join(rows)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(
            '<div class="compare-col-title mine">Mein Modell</div>',
            unsafe_allow_html=True,
        )
        st.markdown(render_prob_bars(my_probs), unsafe_allow_html=True)

    with col_b:
        st.markdown(
            '<div class="compare-col-title birdnet">BirdNET</div>',
            unsafe_allow_html=True,
        )
        if not birdnet_available:
            st.info("`uv sync` für den Vergleich.")
        else:
            st.markdown(render_prob_bars(bn_probs), unsafe_allow_html=True)
            if birdnet_detections:
                with st.expander("Alle BirdNET-Detektionen"):
                    for d in sorted(birdnet_detections, key=lambda x: -x.get("confidence", 0.0)):
                        st.write(
                            f"**{d.get('common_name','?')}** "
                            f"_({d.get('scientific_name','?')})_ — "
                            f"{d.get('confidence',0)*100:.1f}% "
                            f"@ {d.get('start_time',0):.1f}–{d.get('end_time',0):.1f}s"
                        )


# ======================================
# TOP-LEVEL NAVIGATION
# ======================================

tab_classify, tab_model, tab_about = st.tabs([
    "Klassifizieren",
    "Modell & Training",
    "Über das Projekt",
])

with tab_classify:
    render_classifier()

with tab_model:
    render_model_insights()

with tab_about:
    render_project_info()
