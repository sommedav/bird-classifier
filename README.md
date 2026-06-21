# Bird Species Classifier

Ein Proof-of-Concept-System zur automatischen Erkennung von Vogelgesang aus kurzen
Audioaufnahmen (5 Sekunden). Das Modell unterscheidet drei heimische Vogelarten —
**Amsel**, **Kohlmeise** und **Rotkehlchen** — sowie eine vierte Klasse
**Background** (kein Zielvogel hörbar). Eine Streamlit-App stellt die eigene
CNN-Vorhersage dem bekannten System BirdNET direkt gegenüber.

---

## Quickstart for Reviewers

Dieses Repository enthält eine lauffähige Streamlit-App und zwei bereits trainierte Modelle im Ordner `models/` (`birdcnn_release_mit_yamnet.pth` und `birdcnn_release_ohne_yamnet.pth`, beide committed). Die Trainingsdaten sind nicht enthalten, die App kann aber direkt gestartet werden.

Für die Installation siehe Abschnitt "Voraussetzungen" und "Installation".

```bash
git clone https://github.com/sommedav/bird-classifier.git
cd bird-classifier
uv python install 3.11
uv sync --locked
uv run python setup_check.py
uv run streamlit run app.py
```
Wichtig: Der Quickstart soll kompakt bleiben. Die ausführlichen Windows/macOS-Befehle kommen darunter bei **Voraussetzungen** und **Installation**!

Nach dem Start der App kann eine WAV-Datei hochgeladen oder direkt im Browser eine Audioaufnahme erstellt werden. Die App wählt daraus ein 5-sekündiges Audiofenster aus, wandelt dieses in ein Mel-Spektrogramm um und sagt anschließend eine von vier Klassen vorher: Amsel, Kohlmeise, Rotkehlchen oder Background.

Die trainierten Modelle liegen im Ordner `models/`. Mitgeliefert werden zwei Varianten, die sich nur in der Datenpipeline unterscheiden und in der App per Sidebar-Dropdown umschaltbar sind: `birdcnn_release_mit_yamnet.pth` (Background per YAMNet bereinigt, **empfohlen und Default**) und `birdcnn_release_ohne_yamnet.pth` (ältere Pipeline ohne YAMNet, nur zum Vergleich). Die ursprünglichen Trainingsdaten sind nicht im Repository enthalten, da sie zu groß sind und die Audiodateien den jeweiligen Lizenzen der Originalaufnahmen unterliegen. Die Datenpipeline kann mit den Skripten im Ordner `src/` nachvollzogen werden.

---

## Features

- Klassifikation von 5-Sekunden-Audioausschnitten in 4 Klassen
- Erkennung von „kein Vogel" durch explizite Background-Klasse
- Interaktive Streamlit-App: WAV-Datei hochladen oder live aufnehmen
- Mel-Spektrogramm-Visualisierung mit wählbarem Analysefenster
- Direktvergleich mit BirdNET (via `birdnetlib`)
- Reproduzierbare Datenpipeline: Download → Zuschneiden → Split → Training
- Aufnahme-basierter Train/Val/Test-Split verhindert Data Leakage

---

## Tech-Stack

| Bereich | Werkzeuge |
|---|---|
| Deep Learning | PyTorch |
| Audio-Features | librosa, soundfile |
| Daten-Bereinigung | TensorFlow Hub, YAMNet |
| Daten-Split / Metriken | scikit-learn |
| Web-App | Streamlit |
| BirdNET-Vergleich | birdnetlib |
| Daten-Download | requests (Xeno-Canto API) |
| Visualisierung | matplotlib, plotly |
| Notebooks | JupyterLab |

---

## Voraussetzungen (Prerequisites)

Für die Nutzung der Anwendung wird keine manuelle Installation einzelner Python-Pakete benötigt. Das Projekt verwendet `uv` als Paketmanager. `uv` erstellt die virtuelle Umgebung automatisch und installiert die im Projekt festgelegten Abhängigkeiten aus `pyproject.toml` und `uv.lock`.

### Benötigte Software

- Python 3.11 oder neuer. Empfohlen ist Python 3.11, da die Projektumgebung über `.python-version` auf 3.11 festgelegt ist.
- [uv](https://docs.astral.sh/uv/) (Python-Paketmanager) — `brew install uv` oder `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Mindestens ein `.pth`-Modell in `models/` (zwei Modelle sind enthalten; eigene Trainings landen ebenfalls hier)
- Für die reine Nutzung der App wird kein Xeno-Canto-API-Key benötigt
- Für den Xeno-Canto-Download: kostenloser API-Key von [xeno-canto.org](https://xeno-canto.org) 
- BirdNET-Vergleich (`birdnetlib`) ist standardmäßig in `uv sync` enthalten

### uv installieren

`uv` ist der Paketmanager, mit dem die Projektumgebung erstellt wird. Die offiziellen Installationsbefehle unterscheiden sich je nach Betriebssystem.

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Falls curl nicht verfügbar ist, kann alternativ wget verwendet werden:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
Auf macOS kann uv alternativ auch über Homebrew installiert werden:

```bash
brew install uv
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Danach das Terminal beziehungsweise VS Code einmal schließen und neu öffnen.

### Instalation prüfen

Nach der Installation sollte geprüft werden, ob uv korrekt verfügbar ist:

```bash
uv --version
```

Wenn eine Versionsnummer ausgegeben wird, ist uv korrekt installiert.

Falls der Befehl unter Windows nicht gefunden wird, ist uv wahrscheinlich noch nicht im PATH. In diesem Fall das Terminal neu öffnen oder Windows neu starten.

---

## Installation

```bash
# 1. Repository klonen
git clone https://github.com/sommedav/bird-classifier.git
cd bird-classifier

# 2. Abhängigkeiten installieren (erzeugt .venv automatisch)
#    Enthält auch birdnetlib für den BirdNET-Vergleich.
uv sync

# 3. Umgebung prüfen
uv run python setup_check.py
```

---

## Nutzung

### App starten (Schnellstart)

Beide Modelle liegen bereits im Repo — die App ist sofort nutzbar:

```bash
uv run streamlit run app.py
```

Der Browser öffnet sich automatisch. Du kannst eine WAV-Datei hochladen oder
direkt aufnehmen. Mit dem Slider wählst du den 5-Sekunden-Ausschnitt;
die App zeigt Mel-Spektrogramm, CNN-Vorhersage und (optional) BirdNET-Vergleich.
Über das Dropdown in der Sidebar lässt sich zwischen dem YAMNet-Modell (Default)
und der Variante ohne YAMNet umschalten.

Alternativer Modell-Pfad via Umgebungsvariable:

- macOS / Linux
```bash
BIRD_MODEL_PATH=/pfad/zu/modell.pth uv run streamlit run app.py
```
- Windows PowerShell
  ```powershell
  $env:BIRD_MODEL_PATH="C:\Pfad\zu\modell.pth" uv run streamlit run app.py
  ```

- Windows CMD
  ```cmd
  set BIRD_MODEL_PATH=C:\Pfad\zu\modell.pth
  uv run streamlit run app.py
  ```

---

### Modell selbst trainieren

Führe die folgenden Schritte der Reihe nach aus:

```bash
# Schritt 1 — Rohdaten herunterladen (API-Key in src/bird_data.py eintragen)
uv run python src/bird_data.py
#-> Lädt alle konfigurierten Arten in einem Lauf herunter.
#   Zielarten stehen in DOWNLOAD_SPECIES (-> data/<Art>/files/),
#   die Fremdvögel für den Hintergrund in BACKGROUND_SPECIES (-> data/Background/files/).
#   Beide Listen stehen im Block "# --- ANPASSEN ---" am Anfang von src/bird_data.py.

# Schritt 2 — Aufnahmen in 5-s-WAV-Clips schneiden + YAMNet-Filterung
uv run python src/cut_audio.py

# Schritt 3 — Train/Val/Test-Splits erzeugen
uv run python src/build_dataset.py

# Schritt 4 — Notebook öffnen und alle Zellen ausführen
uv run jupyter lab notebooks/bird_training.ipynb
```

Das Training speichert nur den besten Val-Checkpoint unter einem eindeutigen Namen
`models/birdcnn_<timestamp>_best.pth` (gitignored). Die beiden mitgelieferten
Release-Modelle werden dabei nie überschrieben; ein eigenes Modell wird bewusst
zum Release, indem man es z. B. nach `models/birdcnn_release_mit_yamnet.pth` umbenennt.

---

## Projektstruktur

Eine vollständige Übersicht aller Ordner und Dateien mit Erklärungen:
→ [docs/structure.md](docs/structure.md)

```
bird-classifier/
├── src/                    # Datenpipeline-Skripte
├── notebooks/              # Trainings-Notebook
├── data/                   # Rohdaten (nicht committed)
├── data_splits/            # CSV-Splits (nicht committed)
├── docs/                   # Projektdokumentation
├── app.py                  # Streamlit-App
├── models/                 # Modell-Artefakte (zwei Release-Modelle committed)
├── pyproject.toml          # Abhängigkeitsdefinition (UV)
├── uv.lock                 # Reproduzierbares Lockfile
└── ...
```

---

## Modell & Daten

### Architektur — BirdCNN

Eigenes Convolutional Neural Network in PyTorch:

- **Input:** Mel-Spektrogramm `(1, 128, 313)` — 32 kHz, 128 Mel-Bänder, 313 Frames
- **Feature-Extraktion:** 4 Conv-Blöcke (je 2× Conv3×3 → BatchNorm → ReLU → MaxPool),
  Kanaltiefe 1 → 32 → 64 → 128 → 256
- **Klassifikationskopf:** Global Average Pooling → Dropout(0,5) → Linear(256 → 4)
- **Augmentierung:** SpecAugment (Frequenz- und Zeit-Masking, nur beim Training)
- **Training:** 20 Epochen, AdamW (lr=1e-3, weight decay=1e-4),
  CosineAnnealingLR, CrossEntropyLoss (label smoothing=0,1)

### Datensatz

- **Quelle:** [Xeno-Canto](https://xeno-canto.org) — freie Vogelgesang-Aufnahmen
- **Arten:** Amsel, Kohlmeise, Rotkehlchen (Zielklassen) + Taube, Krähe, Spatz (Background)
- **Clip-Länge:** 5 Sekunden WAV, 32 kHz
- **Datensatzgröße (YAMNet-Pipeline):** 6.346 Clips (Train 3.797 / Val 1.585 / Test 964)
- **Split-Strategie:** Aufnahme-basiert (70 % / 15 % / 15 %, fester Seed) → kein Data Leakage
- **Background-Erzeugung:** YAMNet prüft pro Clip, ob ein Vogel hörbar ist (Maximum
  über die Zeitfenster, ab Score 0,20). Zielart-Segmente ohne erkannten Vogel sowie
  die bewusst geladenen Fremdvögel (Taube, Krähe, Spatz) bilden die Background-Klasse.

Vollständige CRISP-DM-Dokumentation: → [docs/crisp-dm.md](docs/crisp-dm.md)

---

## Evaluationsergebnisse

Empfohlenes Modell **`birdcnn_release_mit_yamnet.pth`**, bewertet auf dem unberührten
Test-Set (964 Clips):

| Klasse | Precision | Recall | F1 | Anzahl |
|---|---|---|---|---|
| Amsel | 0,753 | 0,936 | 0,834 | 78 |
| Kohlmeise | 0,907 | 0,772 | 0,834 | 127 |
| Rotkehlchen | 0,755 | 0,923 | 0,831 | 234 |
| Background | 0,947 | 0,853 | 0,898 | 525 |
| **Gesamt (weighted)** | **0,880** | **0,866** | **0,868** | **964** |

**Test-Accuracy: 86,62 %** · Best-Val-Accuracy 93,50 % (Epoche 8 von 20) ·
ROC-AUC 0,96–0,99 je Klasse · ECE 0,098 (brauchbar kalibriert).

### Modellvergleich: mit vs. ohne YAMNet

Im Repo liegen zwei Modelle, die dieselbe BirdCNN-Architektur nutzen, aber auf
unterschiedlich bereinigten Daten trainiert wurden:

| Modell | Datenpipeline | Background-Quelle | Hinweis |
|---|---|---|---|
| `birdcnn_release_mit_yamnet.pth` ⭐ | mit YAMNet (`cut_audio.py`) | YAMNet-Score < 0,20 + Fremdvögel | empfohlen, ausgewogenes Per-Class-Profil, App-Default |
| `birdcnn_release_ohne_yamnet.pth` | ältere Pipeline ohne YAMNet | librosa-Heuristik (Energieanteil > 1 kHz) | Legacy; auf dem YAMNet-Test-Set unausgewogen (z. B. Kohlmeise-Recall ≈ 0,52) |

Das YAMNet-Modell ist die empfohlene Variante: Die erste, YAMNet-lose Version hatte
das Problem, vor allem Rauschen zu „lernen". Erst die YAMNet-Bereinigung und die
explizite Background-Klasse haben dieses Verhalten korrigiert. Das Legacy-Modell
bleibt nur zum direkten Vergleich in der App erhalten.

---

## Tests

Automatisierte Unit-Tests sind in diesem Projekt nicht enthalten. Zur Überprüfung der grundlegenden Lauffähigkeit kann jedoch `setup_check.py` als Smoke-Test ausgeführt werden. Zusätzlich kann die Streamlit-Anwendung lokal mit `uv run streamlit run app.py` gestartet werden, um die Modellinferenz über die Weboberfläche zu testen.


Umgebungscheck:

```bash
uv run python setup_check.py
```

---

## Bekannte Einschränkungen

- Nur drei Vogelarten erkennbar; alle anderen Arten landen in „Background"
- Trainiert auf Studioqualität-Aufnahmen von Xeno-Canto — Feldaufnahmen können abweichen
- Rotkehlchen wird tendenziell überschätzt (Precision 0,76); Kohlmeise hat den
  schwächsten Recall (0,77) — beide werden am ehesten mit Background verwechselt
- BirdNET und PyTorch laufen aus Kompatibilitätsgründen in getrennten Prozessen
- Kein Docker-Image / kein Cloud-Deployment vorhanden

---

## Lizenz

Der Quellcode dieses Projekts wurde für die Nutzung im Rahmen des ML4B-Kurses erstellt. Die verwendeten Audiodateien stammen von Xeno-Canto und unterliegen weiterhin den jeweiligen Lizenzen der Originalaufnahmen. Die Lizenz des Quellcodes ist daher getrennt von den Lizenzen der Audiodaten zu betrachten.


## Quellen / Danksagungen

- [Xeno-Canto](https://xeno-canto.org) — Audiodaten
- [YAMNet](https://tfhub.dev/google/yamnet/1) (Google) — Qualitätskontrolle der Background-Clips
- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) / `birdnetlib` — Vergleichssystem
- [librosa](https://librosa.org) — Audio-Analyse und Mel-Spektrogramme
- [PyTorch](https://pytorch.org) — Modelltraining und Inferenz
- [Streamlit](https://streamlit.io) — Web-App
