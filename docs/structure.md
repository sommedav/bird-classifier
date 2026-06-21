# Projektstruktur — Bird Species Classifier

## Verzeichnisbaum

```
bird-classifier/
├── app.py
├── project.md
├── pyproject.toml
├── uv.lock
├── setup_check.py
│
├── models/
│   ├── birdcnn_release_mit_yamnet.pth    ← Default, mit YAMNet (committed)
│   └── birdcnn_release_ohne_yamnet.pth   ← Legacy, ohne YAMNet (committed)
│
├── data/
│   └── .gitkeep
│
├── data_splits/
│   └── .gitkeep
│
├── docs/
│   ├── structure.md        ← diese Datei
│   └── crisp-dm.md
│
├── notebooks/
│   └── bird_training.ipynb
│
└── src/
    ├── bird_data.py
    ├── cut_audio.py
    ├── yamnet_worker.py
    └── build_dataset.py
```

> `data/` und `data_splits/` sind gitignored und enthalten lokal die Rohdaten
> bzw. die CSV-Splits. Im Repository liegt jeweils nur ein `.gitkeep`.
> Modelle liegen in `models/`. Committet werden die beiden mitgelieferten
> Release-Modelle (`birdcnn_release_mit_yamnet.pth`, `birdcnn_release_ohne_yamnet.pth`);
> selbst trainierte Checkpoints (`birdcnn_<timestamp>_best.pth`) sind gitignored.

---

## Dateien und Ordner im Detail

### Einstiegspunkte

| Pfad | Zweck |
| --- | --- |
| `app.py` | Streamlit-Web-App. Lädt standardmäßig `models/birdcnn_release_mit_yamnet.pth` (per Sidebar-Dropdown sind das zweite Release-Modell und jedes weitere `models/*.pth` wählbar), nimmt eine WAV-Datei entgegen (Upload oder Live-Aufnahme), zeigt Mel-Spektrogramm mit wählbarem 5-s-Ausschnitt, gibt CNN-Vorhersage und optionalen BirdNET-Vergleich aus. Einstiegspunkt: `uv run streamlit run app.py`. |
| `setup_check.py` | Einfacher Umgebungstest: gibt Python-Version und Versionsnummern der wichtigsten Bibliotheken aus. Kein Unittest-Framework, nur manueller Smoke-Test. |

### Modell-Artefakte (`models/`)

| Pfad | Zweck |
| --- | --- |
| `models/birdcnn_release_mit_yamnet.pth` | Mitgeliefertes BirdCNN-State-Dict, Daten mit YAMNet bereinigt (committed). App-Default. |
| `models/birdcnn_release_ohne_yamnet.pth` | Mitgeliefertes BirdCNN-State-Dict der älteren Pipeline ohne YAMNet (committed). Nur zum Vergleich, per Dropdown wählbar. |
| `models/birdcnn_<timestamp>_best.pth` | Selbst trainierte Checkpoints (bester Val-Wert pro Lauf). Gitignored; in der App per Dropdown wählbar. |

### Konfiguration & Doku

| Pfad | Zweck |
| --- | --- |
| `pyproject.toml` | Abhängigkeitsdefinition des UV-Projekts (numpy, librosa, torch, tensorflow/-hub, streamlit, birdnetlib u. a.). |
| `uv.lock` | Reproduzierbares Lockfile — `uv sync` installiert exakt diese Versionen. |
| `project.md` | Ausführliche ML4B-Projektdokumentation auf Deutsch: Idee, Business Understanding, Daten, Modell, Ergebnisse, Reflexion. |

### Datenpipeline — `src/`

Die drei Skripte müssen in dieser Reihenfolge ausgeführt werden:

| Pfad | Schritt | Zweck |
| --- | --- | --- |
| `src/bird_data.py` | 1 | Lädt MP3-Aufnahmen von der Xeno-Canto API herunter — alle konfigurierten Arten parallel in einem Lauf. Konfigurierbar: Zielarten (`DOWNLOAD_SPECIES`), Background-Arten (`BACKGROUND_SPECIES`), maximale Dateianzahl pro Art (`MAX_FILES`), Mindestlänge in Sekunden (`MIN_SECONDS`); die Suchbegriffe je Art stehen im dict `BIRD_CONFIG`. Speichert Zielarten nach `data/<Art>/files/`, Background-Arten nach `data/Background/files/`. |
| `src/cut_audio.py` | 2 | Schneidet lange MP3s in 5-Sekunden-WAV-Clips (32 kHz). YAMNet (isolierter Subprozess, `src/yamnet_worker.py`) bewertet pro Zielart-Clip die Vogel-Präsenz: Score ≥ 0,20 → `data/<Art>/clips/`, sonst (Stille/Rauschen) → `data/Background/clips/`. Background-Arten (Krähe/Taube/Spatz) gehen komplett ohne YAMNet in den Background. |
| `src/yamnet_worker.py` | (Helfer zu 2) | Wird **nicht** direkt aufgerufen, sondern von `cut_audio.py` als isolierter Subprozess gestartet. Lädt YAMNet (TensorFlow Hub) und gibt pro 16-kHz-Clip einen Vogel-Score [0, 1] als JSON zurück. Eigener Prozess, weil sich TensorFlow und PyTorch im selben Prozess nicht zuverlässig vertragen. |
| `src/build_dataset.py` | 3 | Sammelt alle WAV-Clips, vergibt numerische Labels und erzeugt Train/Val/Test-Splits (70 / 15 / 15, Seed=42). Split erfolgt nach `recording_id`, nicht nach einzelnen Clips — verhindert Data Leakage. Ausgabe: `data_splits/train.csv`, `val.csv`, `test.csv`. |

Die drei Pipeline-Skripte (`bird_data.py`, `cut_audio.py`, `build_dataset.py`) enthalten je
einen konfigurierbaren Block am Anfang (`# --- ANPASSEN ---`), in dem Arten, Pfade und
Schwellwerte angepasst werden. `yamnet_worker.py` wird nur intern von `cut_audio.py` genutzt.

### Training — `notebooks/`

| Pfad | Zweck |
| --- | --- |
| `notebooks/bird_training.ipynb` | Vollständiges Trainings-Notebook. Liest die CSV-Splits aus `data_splits/`, berechnet Mel-Spektrogramme, definiert BirdCNN (V3), trainiert 20 Epochen mit AdamW + CosineAnnealing, speichert den besten Checkpoint als `models/birdcnn_<timestamp>_best.pth`, gibt Classification Report und Confusion Matrix aus. |

### Daten — `data/`

| Pfad | Zweck |
| --- | --- |
| `data/` | Basisordner für alle Rohdaten. Nicht committed (nur `.gitkeep`). |
| `data/<Klasse>/files/` | MP3-Dateien direkt von Xeno-Canto, erzeugt von `src/bird_data.py`. Klassen: `Amsel`, `Kohlmeise`, `Rotkehlchen`, `Background`. |
| `data/<Klasse>/clips/` | 5-s-WAV-Clips, erzeugt von `src/cut_audio.py`. Dateiname-Schema: `<Art>_<AufnahmeID>_<Offset>.wav` (erkannter Zielvogel), `…_nobird_bg.wav` (Zielart-Segment ohne Vogel → Background) bzw. `…_bg.wav` (Background-Art). |

### Splits — `data_splits/`

| Pfad | Zweck |
| --- | --- |
| `data_splits/` | CSV-Dateien mit Pfad, Klasse, Label und Aufnahme-ID pro Clip. Nicht committed (nur `.gitkeep`). |
| `data_splits/train.csv` | Trainingsdaten (≈ 70 % der Aufnahmen, 3.797 Clips). |
| `data_splits/val.csv` | Validierungsdaten (≈ 15 % der Aufnahmen, 1.585 Clips). |
| `data_splits/test.csv` | Testdaten (≈ 15 % der Aufnahmen, 964 Clips). |

Spalten je CSV: `path`, `label`, `class_name`, `recording_id`.

### Dokumentation — `docs/`

| Pfad | Zweck |
| --- | --- |
| `docs/structure.md` | Diese Datei — Verzeichnisbaum und Datei-Erklärungen. |
| `docs/crisp-dm.md` | CRISP-DM-Dokumentation der sechs Projektphasen mit Pfadverweisen. |

---

## Ablage-Konventionen

| Was | Wo |
| --- | --- |
| Python-Skripte der Datenpipeline | `src/` |
| Jupyter-Notebooks | `notebooks/` |
| Rohdaten (MP3, WAV) | `data/<Klasse>/files/` bzw. `data/<Klasse>/clips/` — **nicht committen** |
| CSV-Splits | `data_splits/` — **nicht committen** |
| Modell-Checkpoints (`.pth`) | `models/` — die zwei Release-Modelle (`birdcnn_release_mit_yamnet.pth`, `birdcnn_release_ohne_yamnet.pth`) committed, eigene Trainings gitignored |
| Projektdokumentation (Markdown) | `docs/` |

> Große Binärdateien (Audiodaten, Modellgewichte > wenige MB) gehören nicht
> ins Repository. Für Modell-Artefakte sollte langfristig ein Artefakt-Store
> (z. B. DVC, MLflow) in Betracht gezogen werden.
