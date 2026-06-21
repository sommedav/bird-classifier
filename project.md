# ML4B Project Documentation – Bird Species Classifier

## 1. Project Idea

Wir haben ein eigenes Machine-Learning-System gebaut, das Vogelarten anhand von
Audioaufnahmen erkennt. Unterschieden werden drei heimische Arten – Amsel, Kohlmeise
und Rotkehlchen – plus eine vierte Klasse "Background", die greift, wenn gar kein
relevanter Vogelgesang zu hören ist (Stille, Wind, Rauschen oder andere Vögel).
Dazu kommt eine kleine Streamlit-App, in der man eine Aufnahme hochladen oder direkt
aufnehmen kann; sie zeigt das Mel-Spektrogramm und gibt die wahrscheinliche Vogelart
aus und vergleicht sie mit dem bekannten System BirdNET.

Bei der technischen Umsetzung haben wir KI-gestützte Werkzeuge als Unterstützung genutzt, insbesondere bei der Strukturierung des Trainingscodes und bei der Fehlersuche. Die zentralen fachlichen Entscheidungen lagen jedoch bei uns: Auswahl der Datenquelle, Definition der Zielklassen, Aufbau der Datenpipeline, Umgang mit fehlerhaften oder stillen Audioclips, Einführung der Background-Klasse, Vermeidung von Data Leakage durch aufnahmebasiertes Splitting sowie Interpretation der Evaluationsergebnisse. Gerade dadurch wurde deutlich, dass die Qualität der Daten und der Aufbau der Pipeline für dieses Projekt wichtiger waren als die reine Modellarchitektur.


## 2. Business Understanding

Das ist kein fertiges Produkt, sondern ein Proof of Concept. Wir wollten vor allem
zeigen, dass so etwas grundsätzlich funktioniert: Aus einer kurzen Aufnahme lässt
sich für ein paar häufige Gartenvögel automatisch eine Einschätzung gewinnen.

### Problem

Vögel am Gesang zu bestimmen, ist ohne Vorwissen schwierig und mühsam.

### Target Group

Für alle, die so etwas interessant finden – Hobby-Ornithologen, Naturinteressierte,
Gartenbesitzer – und die ohne Expertenwissen schnell eine grobe Einordnung bekommen
wollen.

### Expected Value

Eine Aufnahme wird in Sekunden grob eingeordnet, und durch die Klasse "Background"
sagt das System auch, wenn gar kein Zielvogel zu hören ist, statt einen Vogel zu
erzwingen. Mehr als ein Machbarkeitsnachweis für drei Arten ist es bewusst nicht.

## 3. Data Understanding

### Data Source

Als Datenquelle haben wir die öffentliche Plattform **Xeno-Canto** genutzt, die
tausende frei verfügbare Vogelaufnahmen enthält. Den Download haben wir über die
Xeno-Canto-API mit dem Skript `bird_data.py` automatisiert und dabei gezielt nach
Gesang in guter Qualität (`type:song q:A`) mit mindestens 20 Sekunden Länge gesucht.

Geladen haben wir die drei Zielarten – Amsel (Common Blackbird), Kohlmeise (Great Tit)
und Rotkehlchen (European Robin) – sowie zusätzlich Taube, Krähe und Spatz, aus denen
wir später schwierige Negativbeispiele für die Klasse "Background" gemacht haben. Die
Rohdaten lagen als MP3-Dateien vor und waren oft mehrere Minuten lang.

### Sensors or Features

Wir arbeiten mit reinem Audio. Aus dem Signal berechnen wir mit `librosa`
**Mel-Spektrogramme** – eine bildähnliche Darstellung, die zeigt, welche Frequenzen
wann und wie stark vorkommen. Dieses "Bild" ist das eigentliche Feature für unser CNN.
Einstellungen: 32 kHz Sampling-Rate, 128 Mel-Bänder, `fmax` = 16 kHz, feste Breite von
313 Zeit-Frames.

### Labels

Die Labels für die drei Vogelarten ergeben sich direkt aus dem Download – was wir als
Amsel suchen, bekommt das Label Amsel usw. (numerisch: Amsel = 0, Kohlmeise = 1,
Rotkehlchen = 2, Background = 3). Für "Background" haben wir nicht von Hand gelabelt,
sondern das vortrainierte Google-Modell **YAMNet** als Qualitätskontrolle eingesetzt
(`cut_audio.py`, in einem isolierten Subprozess). YAMNet erkennt pro Clip, ob ein Vogel
hörbar ist (Maximum über die Zeitfenster, ab Score 0.20). Die Background-Klasse speist
sich daraus aus zwei Quellen: Zielart-Segmente **ohne** erkannten Vogel (Stille,
Rauschen) und die bewusst geladenen Fremdvögel (Taube, Krähe, Spatz), die als
"Hard Negatives" komplett in den Background gehen. Wichtig: YAMNet bestimmt nicht die
Vogelart, sondern nur, ob überhaupt ein Vogel da ist.

### Risks

Die größten Risiken waren: verrauschte Clips, die trotzdem als Vogel gelabelt sind
(genau das ist uns am Anfang passiert), Label-Rauschen durch die selbst gesetzte
YAMNet-Schwelle, ein anfängliches Ungleichgewicht zugunsten des Hintergrunds, und die
begrenzte Generalisierung – nur drei Arten, Daten überwiegend aus relativ sauberen
Einzelaufnahmen. Außerdem das Risiko von Data Leakage, wenn Clips derselben
Originalaufnahme in Train und Test landen (siehe Abschnitt 5).

## 4. Related Work

Statt eigener Forschung haben wir bestehende Werkzeuge kombiniert. Die wichtigsten:

- **Xeno-Canto** (Plattform/API) – Quelle aller Aufnahmen.
- **YAMNet** von Google (vortrainiertes Modell) – Qualitätskontrolle und Erzeugung der
  Background-Clips.
- **BirdNET** über `birdnetlib` (Referenzsystem) – Vergleichsmaßstab in der App.
- **librosa** – Laden der Audiodaten und Berechnung der Mel-Spektrogramme.
- **PyTorch** – Aufbau und Training des eigenen CNN.
- **scikit-learn** – Daten-Split und Evaluations-Metriken.
- **Streamlit** – die Benutzeroberfläche der App.

## 5. Data Preparation

**Schneiden in Clips:** Die langen MP3s sind als Ganzes nicht zum Trainieren geeignet.
Mit `cut_audio.py` schneiden wir sie in 5-Sekunden-Clips und speichern sie als WAV.
Kurze Segmente sind ein gängiger Standard für Audio-Klassifikation und reduzieren
unnötigen Hintergrund. Im selben Schritt entscheidet YAMNet pro Zielart-Clip, ob ein
Vogel hörbar ist: Segmente ohne Vogel (Stille/Rauschen) wandern in die Background-Klasse.

**Feature-Berechnung:** Jeder Clip wird in ein Mel-Spektrogramm umgewandelt (32 kHz,
128 Mel-Bänder, `fmax` = 16 kHz). Die Breite wird fest auf 313 Frames gebracht (zu
kurze Clips werden mit Nullen aufgefüllt, zu lange abgeschnitten), danach normalisieren
wir auf Mittelwert 0 und Standardabweichung 1. Die Eingabeform pro Clip ist
`(1, 128, 313)`.

**Splitting (file-based):** Mit `build_dataset.py` teilen wir in Train/Validation/Test.
Wichtig: Wir gruppieren zuerst nach der ursprünglichen Aufnahme-ID und teilen dann
diese Aufnahmen auf (70 % / 15 % / 15 %, fester Seed). So landen Clips derselben
Originalaufnahme nie gleichzeitig in Train und Test – das verhindert Data Leakage. Das
Ergebnis liegt als `train.csv`, `val.csv` und `test.csv` vor (mit Pfad, Klasse,
Label und Aufnahme-ID).

Am Ende hatten wir (YAMNet-Pipeline) 3.797 Clips zum Trainieren, 1.585 für die
Validierung und 964 für den Test (zusammen 6.346). Im Trainingsset verteilt sich das
auf 602 Amsel, 895 Kohlmeise, 838 Rotkehlchen und 1.462 Background.

## 6. Modeling

**Erstes Modellproblem (Baseline):** Eine erste Notebook-Version lief zwar,
hatte aber ein klares Problem: Weil viele Clips Stille oder Rauschen enthielten und
trotzdem als Vogel gelabelt waren, hat das Modell falsche Muster gelernt – selbst bei
stillen Aufnahmen wurde mit hoher Sicherheit ein Vogel vorhergesagt. Aus diesem Fehler
ist die Idee entstanden, die Daten mit YAMNet zu bereinigen und "Background" als vierte
Klasse einzuführen.

**Finales Modell:** Das finale Modell (`notebooks/bird_training.ipynb`) ist ein eigenes
Convolutional Neural Network (BirdCNN) in PyTorch. Es bekommt ein Mel-Spektrogramm und
gibt Wahrscheinlichkeiten für die vier Klassen aus. Es besteht aus vier
Convolution-Blöcken (jeweils zweimal Conv 3×3 → BatchNorm → ReLU, dann MaxPooling;
Kanäle 1 → 32 → 64 → 128 → 256), einem Global Average Pooling und einem Klassifikations-
kopf aus Dropout (0.5) und einer Linear-Schicht (256 → 4). Beim Training kommt zusätzlich
SpecAugment (Frequenz- und Zeit-Masking) zum Einsatz, um Overfitting zu reduzieren.

Trainiert wurde über 20 Epochen mit Batch-Größe 32, CrossEntropy-Loss (Label Smoothing
0.1), dem AdamW-Optimizer (lr = 1e-3, weight decay = 1e-4) und einem CosineAnnealing-
Scheduler. Nach jeder Epoche haben wir auf der Validierung gemessen und jeweils das beste
Modell unter einem eindeutigen Namen `models/birdcnn_<timestamp>_best.pth` gespeichert
(so überschreibt ein neuer Lauf die mitgelieferten Release-Modelle nie). Es wird nur
der beste Checkpoint gespeichert. Das hier dokumentierte, mitgelieferte Modell ist
`models/birdcnn_release_mit_yamnet.pth` (bestes Val-Ergebnis 93,50 % in Epoche 8).

## 7. Evaluation

Bewertet wird das empfohlene Modell `birdcnn_release_mit_yamnet.pth` auf dem zuvor
unberührten Test-Set (964 Clips). Ausgewählt wurde der Checkpoint mit der besten
Validation-Accuracy (93,50 % in Epoche 8). Auf dem Test-Set erreicht er eine
**Gesamt-Accuracy von 86,62 %**.

Pro Klasse sieht das so aus – Amsel wird mit Abstand am zuverlässigsten erkannt,
Kohlmeise am schwächsten (Recall):

```
Klasse         Accuracy (= Recall)   Anzahl
Amsel           93.59 %                78
Kohlmeise       77.17 %               127
Rotkehlchen     92.31 %               234
Background      85.33 %               525
```

Der Classification Report (Precision / Recall / F1):

```
              precision   recall   f1-score   support
Amsel            0.753    0.936     0.834        78
Kohlmeise        0.907    0.772     0.834       127
Rotkehlchen      0.755    0.923     0.831       234
Background       0.947    0.853     0.898       525

accuracy                            0.866       964
macro avg        0.841    0.871     0.849       964
weighted avg     0.880    0.866     0.868       964
```

Confusion Matrix (Zeilen = wahr, Spalten = vorhergesagt):

```
              Amsel  Kohlmeise  Rotkehlchen  Background
Amsel           73       0           3            2
Kohlmeise        0      98          19           10
Rotkehlchen      1       4         216           13
Background      23       6          48          448
```

Zusätzlich (One-vs-Rest): **ROC-AUC** Amsel 0,989 · Kohlmeise 0,985 · Rotkehlchen 0,968 ·
Background 0,963; **PR-AP** zwischen 0,881 und 0,972. Die Kalibrierung ist mit einem
**ECE von 0,098** (bei Ø-Konfidenz 0,772) brauchbar.

**Interpretation:** Amsel und Rotkehlchen werden gut getroffen. Rotkehlchen hat die
niedrigste Precision (0,755) – das Modell sagt "Rotkehlchen" tendenziell zu oft, vor
allem aus der Background-Klasse (48 Fälle) und aus Kohlmeise (19). Kohlmeise hat den
schwächsten Recall (0,772). Background ist die breite "Restklasse", aus der Fehler in
die Vogelarten überlaufen. Für ein selbst gebautes CNN auf vier Klassen ist das ein
solides Ergebnis.

### Vergleich: mit vs. ohne YAMNet

Im Repo liegen zwei mitgelieferte Modelle mit identischer BirdCNN-Architektur, die sich
nur in der Datenpipeline unterscheiden:

- **`birdcnn_release_mit_yamnet.pth`** (empfohlen, App-Default): Background mit YAMNet
  bereinigt; die oben dokumentierten Werte (86,62 % Test-Accuracy, ausgewogenes
  Per-Class-Profil) sind über `notebooks/bird_training.ipynb` reproduzierbar.
- **`birdcnn_release_ohne_yamnet.pth`** (Legacy): das ältere Modell der ursprünglichen
  Pipeline, in der die Background-Clips noch über eine librosa-Heuristik (Energieanteil
  oberhalb 1 kHz) statt über YAMNet erzeugt wurden. Auf dem heutigen, YAMNet-gefilterten
  Test-Set wirkt es unausgewogen (z. B. Kohlmeise-Recall ≈ 0,52), weil es auf anders
  definierten Background-Daten trainiert wurde. Es bleibt nur zum direkten Vergleich in
  der App erhalten.

Der Wechsel von der librosa-Heuristik zu YAMNet (plus der expliziten Background-Klasse)
war der entscheidende Schritt: Die erste Version hatte vor allem Rauschen "gelernt",
erst die YAMNet-Bereinigung hat dieses Verhalten korrigiert.

## 8. Deployment

Die Anwendung ist eine Streamlit-App (`app.py`) und wird lokal gestartet mit:

```bash
uv run streamlit run app.py
```

In der App kann man eine WAV-Datei hochladen oder live aufnehmen, sieht das
Mel-Spektrogramm, wählt einen 5-Sekunden-Ausschnitt und bekommt die Vorhersage unseres
eigenen CNN (Default `models/birdcnn_release_mit_yamnet.pth`) – die Klasse "Background" wird dabei als "Kein Vogel"
angezeigt. Zusätzlich wird dieselbe Aufnahme von BirdNET analysiert und das Ergebnis
gegenübergestellt. BirdNET läuft bewusst in einem eigenen Subprozess, weil sich PyTorch
und TensorFlow-Lite sonst in die Quere kommen.

Damit Training und App zusammenpassen, ist die Vorverarbeitung in der App identisch zum
Training (32 kHz, 128 Mel-Bänder, 313 Frames, gleiche Normalisierung). Die benötigten
Bibliotheken sind in `pyproject.toml` definiert und über `uv sync` reproduzierbar
installierbar.

## 9. Reflection

Die wichtigste Lektion war, dass saubere Daten mehr bringen als ein komplexeres Netz:
Der eigentliche Fortschritt kam nicht durch ein größeres Modell, sondern durch das
Aufräumen der Daten mit YAMNet und die Einführung der Background-Klasse – die erste
Version hatte vor allem Rauschen "gelernt". Auch die bewussten Hard Negatives (Taube,
Krähe, Spatz) haben geholfen, Zielvögel von anderen Vögeln zu trennen, und das
file-based Splitting nach Aufnahme-ID war nötig, um zu optimistische Ergebnisse durch
Data Leakage zu vermeiden.

Offen geblieben sind ein paar Schwächen: Rotkehlchen wird zu oft vorhergesagt, und
Background ist die schwierigste Klasse – mehr und vielfältigere Daten würden hier
wahrscheinlich am meisten bringen. Technisch hat uns BirdNET zusammen mit PyTorch
zunächst Konflikte beschert, die wir über einen Subprozess gelöst haben. Und auch wenn
wir den ML-Teil stark mit KI erarbeitet haben, haben wir gerade an den Daten- und
Problemstellen viel darüber gelernt, worauf es bei so einem Projekt wirklich ankommt.
