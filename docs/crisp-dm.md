# CRISP-DM Dokumentation — Bird Species Classifier

Dieses Dokument beschreibt den ML-Entwicklungsprozess entlang der sechs
CRISP-DM-Phasen. Alle Aussagen beziehen sich auf konkrete Dateien im Repository.

---

## 1. Business Understanding

### Problemstellung

Vögel am Gesang zu bestimmen erfordert Vorwissen und ist für Laien schwierig.

### Ziel

Proof-of-Concept-Klassifikator, der aus einer kurzen Audioaufnahme (5 Sekunden)
automatisch eine Einschätzung der Vogelart liefert — ohne Expertenwissen.
Das System erkennt drei häufige Gartenvögel und sagt explizit, wenn kein Zielvogel
zu hören ist (statt eine falsche Art zu erzwingen).

### Zielklassen

| Label | Art | Wissenschaftlicher Name |
| --- | --- | --- |
| 0 | Amsel | Turdus merula |
| 1 | Kohlmeise | Parus major |
| 2 | Rotkehlchen | Erithacus rubecula |
| 3 | Background | (kein Zielvogel) |

### Zielgruppe

Hobby-Ornithologen, Naturinteressierte, Gartenbesitzer — keine Fachkenntnisse
erforderlich.

### Bewertungskriterium

Die Gesamtaccuracy auf dem unberührten Test-Set sowie die Per-Class-Accuracy
entscheiden über die Modellqualität. Die Klasse „Background" ist besonders wichtig,
damit das System nicht immer eine Vogelart ausgibt.

### Abgrenzung

Kein Produktivsystem — bewusster Machbarkeitsnachweis für drei Arten.

**Hauptdatei:** `project.md` (Abschnitte 1–2)

---

## 2. Data Understanding

### Datenquelle

Alle Aufnahmen stammen von [Xeno-Canto](https://xeno-canto.org) —
einer öffentlichen Plattform mit tausenden frei verfügbaren Vogelaufnahmen.
Der Download erfolgt über die Xeno-Canto API v3.

**Skript:** `src/bird_data.py`

```
API-Endpunkt: https://xeno-canto.org/api/3/recordings
Filter: type:song q:A (Gesang, beste Qualität), Mindestlänge 20 s
```

### Heruntergeladene Arten

| Verwendung | Art | Xeno-Canto-Suchbegriff |
| --- | --- | --- |
| Zielklasse | Amsel | `en:"Common Blackbird"` |
| Zielklasse | Kohlmeise | `en:"Great Tit"` |
| Zielklasse | Rotkehlchen | `en:"European Robin"` |
| Background (Hard Negatives) | Krähe | `en:"Carrion Crow"` |
| Background (Hard Negatives) | Taube | `en:"Wood Pigeon"` |
| Background (Hard Negatives) | Spatz | `en:"House Sparrow"` |

Konfiguration je Art in `src/bird_data.py`, dict `BIRD_CONFIG`.

### Rohformat

MP3-Dateien, teils mehrere Minuten lang, Qualitätsstufe A.
Abgelegt nach dem Download unter `data/<Klasse>/files/`.

### Features

Aus dem Audiosignal werden **Mel-Spektrogramme** berechnet:

| Parameter | Wert |
| --- | --- |
| Sampling-Rate | 32.000 Hz (alle Clips); YAMNet-Bewertung intern bei 16.000 Hz |
| Mel-Bänder | 128 |
| fmax | 16.000 Hz |
| Zeitfenster | feste Breite von 313 Frames (Padding / Crop) |
| Normalisierung | mean=0, std=1 pro Clip |
| Eingabeform CNN | `(1, 128, 313)` |

Feature-Berechnung: `notebooks/bird_training.ipynb` (Funktion `audio_to_mel`),
Inferenz: `app.py` (Funktion `array_to_mel_for_model`)

### Labels

- **Zielarten:** Klasse ergibt sich direkt aus dem Download (was als Amsel gesucht
  wird, erhält das Label Amsel).
- **Background:** Kein manuelles Labeling. Das vortrainierte Google-Modell
  **YAMNet** erkennt pro Clip, ob ein Vogel hörbar ist (Skript: `src/cut_audio.py`).
  Zielart-Segmente ohne erkannten Vogel (Stille/Rauschen) wandern automatisch in
  die Background-Klasse; die Background-Arten (Krähe/Taube/Spatz) liefern zusätzlich
  klare Fremdvogel-Negative.

### Datensatzgröße (nach Vorverarbeitung)

| Split | Clips |
| --- | --- |
| Train | 3.797 |
| Validation | 1.585 |
| Test | 964 |
| **Gesamt** | **6.346** |

Klassenverteilung im Trainingsset: 602 Amsel, 895 Kohlmeise,
838 Rotkehlchen, 1.462 Background.

**Hauptdateien:** `src/bird_data.py`, `project.md` (Abschnitt 3)

### Identifizierte Risiken

- Verrauschte Clips trotz Qualitätsstufe A → gelöst durch YAMNet-Filterung
- Label-Rauschen durch die selbst gesetzte YAMNet-Schwelle
- Generalisierungslücke: Daten aus relativ sauberen Einzelaufnahmen,
  keine Feldaufnahmen
- Data Leakage (wenn Clips derselben Aufnahme in Train und Test landen)
  → gelöst durch aufnahmebasierten Split (siehe Phase 3)

---

## 3. Related Work

Dieses Projekt baut auf bestehenden Werkzeugen und Datensätzen zur audiobasierten Klassifikation von Vogelarten auf.

* **Xeno-Canto** stellt öffentlich zugängliche Vogelstimmenaufnahmen bereit und dient als zentrale Datenquelle für dieses Projekt.
* **YAMNet** wird als vortrainierter Audio-Event-Klassifikator genutzt, um Hintergrundaufnahmen zu filtern und Segmente mit vogelähnlichen Geräuschen zu erkennen.
* **BirdNET** wird in der Streamlit-Anwendung als externes Referenzsystem verwendet. BirdNET wird nicht zum Training unseres Modells genutzt, sondern dient dazu, die Vorhersagen unseres CNN-Modells mit einem etablierten System zur Vogelstimmenerkennung zu vergleichen.
* **librosa** wird zum Laden der Audiodateien und zur Erzeugung der Mel-Spektrogramme verwendet.
* **PyTorch** wird zur Implementierung und zum Training des eigenen CNN-Modells genutzt.
* **Streamlit** wird verwendet, um das Modell über eine interaktive Webanwendung nutzbar zu machen.

Der wichtigste Unterschied zu BirdNET besteht darin, dass unser Modell bewusst auf drei lokale Vogelarten und eine Background-Klasse beschränkt ist. Dadurch ist die Klassifikationsaufgabe kleiner, besser interpretierbar und für einen Proof of Concept im Rahmen des Kurses geeignet.

---

## 4. Data Preparation

### Schritt 1 — Zuschneiden in Clips

**Skript:** `src/cut_audio.py`

Lange MP3-Aufnahmen werden in **5-Sekunden-WAV-Segmente** zerschnitten (32 kHz,
identisch zu Training und Inferenz). Kurze Rest-Segmente am Ende werden verworfen.

**YAMNet** läuft in einem **isolierten Subprozess** (`src/yamnet_worker.py`), da
sich TensorFlow und PyTorch im selben Prozess nicht zuverlässig vertragen. Es
bewertet pro Segment, ob ein Vogel hörbar ist — als **Maximum über die Zeitfenster**
des Clips, damit auch kurze Gesangsphasen erkannt werden. Davon hängt das Routing ab:

**Zielarten** (Amsel, Kohlmeise, Rotkehlchen):

| YAMNet-Vogel-Score | Zuweisung | Ziel |
| --- | --- | --- |
| ≥ 0,20 (`BIRD_PRESENCE_THRESHOLD`) | Vogel hörbar | `data/<Art>/clips/` |
| < 0,20 | kein Vogel (Stille/Rauschen) | `data/Background/clips/` (`…_nobird_bg.wav`) |

**Background-Arten** (Krähe, Taube, Spatz): **alle** Segmente → `data/Background/clips/`
(`…_bg.wav`), **ohne** YAMNet — diese Arten sind selbst Vögel, nur eben nicht unsere
Zielarten, und dienen als klare Fremdvogel-Negative.

Wichtig: Es wird **nichts verworfen** — Stille und Rauschen aus Zielart-Aufnahmen
gehören bewusst in die Background-Klasse, damit das Modell „kein Zielvogel" lernt.
YAMNet bestimmt **nicht** die Vogelart, sondern nur, ob überhaupt ein Vogel da ist.

### Schritt 2 — Feature-Berechnung

Mel-Spektrogramme werden zur Ladezeit im Notebook berechnet (kein separates
Vorverarbeitungs-Skript). Funktion `audio_to_mel` in
`notebooks/bird_training.ipynb`:

1. Laden mit `librosa.load` (sr=32.000 Hz)
2. `librosa.feature.melspectrogram` (n_mels=128, fmax=16.000)
3. Konvertierung zu Dezibel: `librosa.power_to_db`
4. Auf 313 Frames bringen (Padding mit Nullen oder Abschneiden)
5. Z-Normalisierung: `(x - mean) / (std + 1e-6)`

Dieselbe Pipeline ist identisch in `app.py` (`array_to_mel_for_model`)
implementiert — wichtig für konsistente Inferenz.

### Schritt 3 — Augmentierung

**SpecAugment** (Frequenz- und Zeit-Masking) wird im Training angewendet:

| Parameter | Wert |
| --- | --- |
| Frequenz-Masken | 2 Masken, max. 24 Bänder |
| Zeit-Masken | 2 Masken, max. 40 Frames |

Implementiert als `SpecAugment`-Klasse in `notebooks/bird_training.ipynb`
(und als Stub in `app.py`, da beim Inference `model.eval()` aktiv ist).

### Schritt 4 — Train / Val / Test-Split

**Skript:** `src/build_dataset.py`

- Alle WAV-Clips werden gesammelt, Labels vergeben (`LABEL_MAP`)
- Clips werden nach `recording_id` (erste Ziffernfolge im Dateinamen) gruppiert
- Split der **Aufnahmen** (nicht der Clips): 70 % / 15 % / 15 %, Seed=42
- So landen Clips derselben Originalaufnahme nie gleichzeitig in Train und Test

Ausgabe: `data_splits/train.csv`, `data_splits/val.csv`, `data_splits/test.csv`

Spalten: `path`, `label`, `class_name`, `recording_id`

**Hauptdateien:** `src/cut_audio.py`, `src/build_dataset.py`,
`notebooks/bird_training.ipynb`

---

## 5. Modeling

### Architektur — BirdCNN (V3)

Eigenes Convolutional Neural Network in PyTorch.
Definiert in `notebooks/bird_training.ipynb` und `app.py`.

```
Input: (B, 1, 128, 313)  ← Mel-Spektrogramm, Batch × Kanal × Freq × Zeit
│
├── SpecAugment (nur Training)
│
├── Conv-Block 1: Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → MaxPool2d
│   Kanäle: 1 → 32
├── Conv-Block 2: wie oben
│   Kanäle: 32 → 64
├── Conv-Block 3: wie oben
│   Kanäle: 64 → 128
├── Conv-Block 4: wie oben
│   Kanäle: 128 → 256
│
├── AdaptiveAvgPool2d(1)   ← Global Average Pooling
├── Flatten
├── Dropout(0.5)
└── Linear(256 → 4)        ← 4 Klassen
```

### Hyperparameter

| Parameter | Wert |
| --- | --- |
| Epochen | 20 |
| Batch-Größe | 32 |
| Optimizer | AdamW |
| Lernrate | 1e-3 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=20) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Dropout | 0,5 |
| Device | MPS (Apple Silicon) / CPU |

### Modell-Checkpointing

Nach jeder Epoche wird auf der Validierung evaluiert. Wenn `val_acc` den
bisherigen Bestwert übertrifft, wird der State-Dict gespeichert — unter einem
eindeutigen Namen `models/birdcnn_<timestamp>_best.pth`, sodass ein neuer Lauf die
mitgelieferten Release-Modelle nie überschreibt. Gespeichert wird nur
der beste Checkpoint (keine separate Datei für die letzte Epoche).

### Erste Modellversion (Baseline)

Eine erste Notebook-Version (vgl. `project.md`, Abschnitt 6) hatte
ein kritisches Problem: Viele Clips enthielten Stille oder Rauschen und waren
trotzdem als Vogel gelabelt. Das Modell lernte Rauschen statt Vogelgesang.
Daraus entstand die Idee zur YAMNet-Bereinigung und der Background-Klasse.

**Hauptdatei:** `notebooks/bird_training.ipynb`, `app.py`

---

## 6. Evaluation

### Methode

Bewertet wird das empfohlene Modell `models/birdcnn_release_mit_yamnet.pth` auf dem
zuvor unberührten Test-Set (964 Clips). Bester Val-Checkpoint: Epoche 8 mit 93,50 %.

Metriken berechnet mit `sklearn.metrics` in `notebooks/bird_training.ipynb`.

### Ergebnisse (aus Notebook-Output)

**Test-Accuracy: 86,62 %**

#### Per-Class-Accuracy (= Recall)

| Klasse | Accuracy | Anzahl Clips |
| --- | --- | --- |
| Amsel | 93,59 % | 78 |
| Kohlmeise | 77,17 % | 127 |
| Rotkehlchen | 92,31 % | 234 |
| Background | 85,33 % | 525 |

#### Classification Report

| Klasse | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Amsel | 0,753 | 0,936 | 0,834 | 78 |
| Kohlmeise | 0,907 | 0,772 | 0,834 | 127 |
| Rotkehlchen | 0,755 | 0,923 | 0,831 | 234 |
| Background | 0,947 | 0,853 | 0,898 | 525 |
| **accuracy** | | | **0,866** | **964** |
| macro avg | 0,841 | 0,871 | 0,849 | 964 |
| weighted avg | 0,880 | 0,866 | 0,868 | 964 |

#### Confusion Matrix (Zeilen = wahr, Spalten = vorhergesagt)

|  | Amsel | Kohlmeise | Rotkehlchen | Background |
| --- | --- | --- | --- | --- |
| **Amsel** | 73 | 0 | 3 | 2 |
| **Kohlmeise** | 0 | 98 | 19 | 10 |
| **Rotkehlchen** | 1 | 4 | 216 | 13 |
| **Background** | 23 | 6 | 48 | 448 |

Ergänzend (One-vs-Rest): ROC-AUC 0,963–0,989 je Klasse, PR-AP 0,881–0,972,
Kalibrierung ECE = 0,098 (Ø-Konfidenz 0,772).

### Interpretation

- Amsel und Rotkehlchen werden am zuverlässigsten erkannt
- Rotkehlchen hat die niedrigste Precision (0,755) → wird tendenziell überschätzt,
  vor allem aus Background (48 Fälle) und Kohlmeise (19)
- Kohlmeise hat den schwächsten Recall (0,772)
- Background ist die breite Restklasse, aus der Fehler in die Vogelarten überlaufen
- Für ein selbst trainiertes CNN auf 4 Klassen ist das Ergebnis solide

### Vergleich: mit vs. ohne YAMNet

Im Repo liegen zwei Modelle mit identischer BirdCNN-Architektur. Das oben bewertete
`birdcnn_release_mit_yamnet.pth` nutzt die YAMNet-bereinigte Pipeline und ist der
App-Default. `birdcnn_release_ohne_yamnet.pth` stammt aus der älteren Pipeline, in der
die Background-Clips noch per librosa-Heuristik (Energieanteil oberhalb 1 kHz) statt per
YAMNet erzeugt wurden; auf dem heutigen YAMNet-Test-Set wirkt es unausgewogen
(z. B. Kohlmeise-Recall ≈ 0,52) und bleibt nur zum direkten Vergleich erhalten.

**Hauptdatei:** `notebooks/bird_training.ipynb` (Evaluation-Zelle am Ende),
`project.md` (Abschnitt 7)

---

## 7. Deployment

### Anwendung

Die Anwendung ist eine **Streamlit-App** (`app.py`), die lokal gestartet wird.

```bash
uv run streamlit run app.py
```

Kein Docker-Image, kein Cloud-Deployment vorhanden.

### Funktionsumfang der App

| Feature | Details |
| --- | --- |
| Audio-Eingabe | WAV-Datei hochladen oder Live-Aufnahme via Browser |
| Mel-Spektrogramm | Visualisierung der gesamten Aufnahme mit farbig markiertem Analysefenster |
| Fenster-Auswahl | Slider zur Wahl des 5-s-Ausschnitts (bei Aufnahmen > 5 s) |
| CNN-Vorhersage | Konfidenz-Wahrscheinlichkeiten für alle 4 Klassen; Background → „Kein Vogel" |
| BirdNET-Vergleich | Optional via `birdnetlib`; läuft in separatem Subprocess |
| Arten-Info | Karten mit Emoji, deutschem und wissenschaftlichem Name |

### Modell-Pfad

Die App lädt standardmäßig `models/birdcnn_release_mit_yamnet.pth`. In der Sidebar kann
per Dropdown das zweite Release-Modell (`birdcnn_release_ohne_yamnet.pth`) oder jeder
weitere `models/*.pth` (z. B. ein selbst trainierter Checkpoint) gewählt werden. Ein
fester Pfad lässt sich erzwingen via:

```bash
BIRD_MODEL_PATH=/pfad/zu/modell.pth uv run streamlit run app.py
```

### Konsistenz Training ↔ Inferenz

Die Vorverarbeitungs-Parameter sind in `app.py` und `notebooks/bird_training.ipynb`
identisch implementiert:

| Parameter | Wert |
| --- | --- |
| Sampling-Rate | 32.000 Hz |
| Mel-Bänder | 128 |
| fmax | 16.000 Hz |
| Frames | 313 (Padding / Crop) |
| Normalisierung | mean=0, std=1 |

### BirdNET-Subprocess

BirdNET (`birdnetlib`) nutzt TensorFlow-Lite, das sich mit PyTorch im selben
Prozess nicht zuverlässig verträgt. Lösung: BirdNET wird in einem separaten
Python-Subprocess aufgerufen (`subprocess.run`). Das JSON-Ergebnis wird über
stdout zurückgegeben. Implementiert in `app.py` als `_BIRDNET_SUBPROCESS_SCRIPT`.

> TODO: Docker-Image für reproduzierbares Deployment erstellen.
> TODO: Cloud-Deployment (z. B. Streamlit Cloud, Hugging Face Spaces) evaluieren.

**Hauptdatei:** `app.py`, `project.md` (Abschnitt 8)

---

## 8. Future Work

Mögliche Erweiterungen wären ein Docker-Image für eine reproduzierbare Laufzeitumgebung und ein Cloud-Deployment, zum Beispiel über Streamlit Cloud oder Hugging Face Spaces. Für die aktuelle Kursabgabe wird die Anwendung lokal über `uv run streamlit run app.py` ausgeführt.

