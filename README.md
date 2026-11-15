# COMP4471
# Detailed project plan: real-time push-up and sit-up repetition counting

## Environment setup

The repository uses Python 3.10 or newer. Create the local environment and
install the training, pose-extraction, notebook, and test dependencies with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e '.[notebooks,dev]'
# Use the headless OpenCV wheel in Linux containers without libGL.
.venv/bin/pip install --force-reinstall --no-deps 'opencv-python-headless>=4.9,<6'
```

The initial environment is CPU/GPU agnostic and uses PyTorch for the temporal
models. The `data/` directories are created as part of the repository layout;
raw videos and generated features remain local and are intentionally ignored by
Git. Run the smoke tests with:

```bash
.venv/bin/python -m pytest -q
```

## 1. Final project objective

Build a real-time webcam system that:

1. Detects a person using MediaPipe Pose.
2. Extracts pose landmarks and motion features.
3. Identifies whether the exercise is a push-up or sit-up.
4. Classifies movement phases: `up`, `down`, or `transition`.
5. Counts repetitions using a temporal state machine.
6. Compares learned prediction against an angle-threshold baseline.
7. Measures accuracy, robustness, and real-time speed.

Recommended project framing:

> We use a frozen pretrained pose-estimation model and train a lightweight temporal model on top of pose sequences. Public data are used for initial training, while our manually annotated multi-view recordings are used for fine-tuning and subject-independent evaluation.

---

# 2. System architecture

```text
Webcam/video
    ↓
MediaPipe Pose Landmarker
    ↓
33 landmarks × x/y/z/visibility
    ↓
Coordinate normalization and missing-value handling
    ↓
Feature extraction
    ├── Raw normalized landmarks
    ├── Joint angles
    ├── Velocities
    └── Confidence values
    ↓
Temporal window
    ↓
1D CNN or GRU/LSTM
    ↓
Phase probabilities
    ├── up
    ├── down
    └── transition
    ↓
Smoothing and state machine
    ↓
Repetition count
    ↓
Form-quality indicators and annotated video
```

Keep MediaPipe frozen. Train only the temporal classifier and, optionally, the exercise classifier.

---

# 3. Phase 0 — Define the exact scope

## 3.1 Initial exercise scope

Start with only:

- Push-ups
- Sit-ups

Do not add squats, lunges, or planks until the first two exercises work reliably.

## 3.2 Define one repetition

### Push-up

A repetition is:

```text
Top position → lowering → bottom position → rising → top position
```

### Sit-up

A repetition is:

```text
Lying/low position → rising → upright/high position → returning to low position
```

Write these definitions before annotation so every group member labels videos consistently.

## 3.3 Define the primary research question

Recommended question:

> Does a lightweight temporal model trained on pose sequences improve repetition-counting accuracy over fixed joint-angle thresholds under changes in subject, camera angle, and movement speed?

## 3.4 Define hypotheses

Example hypotheses:

- H1: The temporal model achieves lower repetition-count MAE than angle thresholding.
- H2: The temporal model performs better than thresholding when exercise speed varies.
- H3: Subject-independent performance is lower than within-subject performance.
- H4: Sit-ups benefit more from temporal modeling because their motion is more variable.

---

# 4. Phase 1 — Repository and project structure

Use `pushup-tracker` as an engineering reference, but do not blindly copy it. First inspect:

- pose extraction,
- feature preprocessing,
- model architecture,
- data format,
- train/test split,
- repetition state machine,
- license,
- dependency versions.

Recommended project structure:

```text
exercise-rep-counter/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   └── splits/
├── notebooks/
│   ├── 01_pose_extraction.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_baseline.ipynb
│   └── 04_model_training.ipynb
├── src/
│   ├── pose/
│   │   ├── extractor.py
│   │   └── normalization.py
│   ├── features/
│   │   ├── angles.py
│   │   └── motion.py
│   ├── models/
│   │   ├── cnn.py
│   │   └── gru.py
│   ├── counting/
│   │   ├── thresholds.py
│   │   └── state_machine.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── plots.py
│   └── demo/
│       └── realtime.py
├── tests/
├── README.md
├── requirements.txt
└── LICENSE
```

The current notebook should become an exploratory notebook, not the final application.

---

# 5. Phase 2 — Collect and organize data

## 5.1 Public data

Use public datasets for experimentation and possible pretraining:

- exercise-recognition video datasets,
- UI-PRMD,
- KIMORE,
- public pose/keypoint datasets,
- permitted Kaggle exercise datasets.

Before using any dataset, record:

```text
Dataset name
URL
License
Number of subjects
Exercises
Camera views
Available labels
Whether repetition boundaries exist
```

Do not assume that a dataset permits redistribution or publication of extracted videos.

## 5.2 Own recording protocol

Record 5–8 volunteers if possible.

For each volunteer:

- 2–3 push-up recordings,
- 2–3 sit-up recordings,
- front view,
- side view,
- 45-degree view,
- slow speed,
- normal speed,
- fast speed,
- good form,
- deliberately imperfect form if form analysis is included.

Keep the camera fixed during each recording.

Recommended recording conditions:

```text
Resolution: 720p or 1080p
Frame rate: 30 FPS
Duration: 20–60 seconds
Distance: enough to show the full body
Lighting: normal indoor lighting
Background: varied but not visually crowded
```

## 5.3 Target dataset size

Minimum viable dataset:

```text
5 subjects × 2 exercises × 3 views × 2 recordings
```

Better target:

```text
8 subjects × 2 exercises × 3 views × 3 recordings
```

Aim for at least:

- 1,000 total repetitions,
- 500 push-ups,
- 500 sit-ups,
- multiple repetitions per video,
- at least one unseen subject for final testing.

---

# 6. Phase 3 — Annotation design

## 6.1 Annotation format

Use one row per repetition:

```csv
video_id,subject_id,exercise,view,start_frame,bottom_frame,end_frame,quality
v001,s01,pushup,side,120,145,170,good
v001,s01,pushup,side,171,198,225,good
```

For phase-level training, also create frame labels:

```csv
video_id,frame,exercise,phase,rep_id
v001,120,pushup,up,1
v001,121,pushup,down,1
v001,145,pushup,bottom,1
v001,170,pushup,up,1
```

You can simplify the model labels to:

```text
0 = up
1 = down
2 = transition
```

## 6.2 Annotation rules

Create a short annotation guide:

- What counts as the start of a repetition?
- What counts as the bottom position?
- How are incomplete repetitions labeled?
- How are missed frames handled?
- What happens when the pose detector loses the person?
- How is poor form marked?

Two people should independently annotate a subset of videos. Compare their labels to measure annotation consistency.

## 6.3 Avoid pseudo-label dependence

The current notebook creates labels from angle thresholds. That is acceptable only for a preliminary demonstration.

For the final experiment:

```text
Human annotations = ground truth
Angle thresholds = baseline only
```

---

# 7. Phase 4 — Improve pose and feature preprocessing

## 7.1 Pose extraction

For every frame, store:

```text
frame index
timestamp
33 landmarks
x, y, z
visibility/presence confidence
```

Do not store only angles. Preserve the raw pose data so you can test different feature sets later.

## 7.2 Normalize coordinates

Raw pixel or normalized image coordinates are sensitive to camera distance and body position.

Use:

1. Hip midpoint as the origin.
2. Shoulder width or torso length as the scale.
3. Optional body orientation normalization.
4. Interpolation for short missing-pose gaps.
5. Sequence rejection for long missing-pose intervals.

Example:

```text
center = midpoint(left_hip, right_hip)
scale = distance(left_shoulder, right_shoulder)
normalized_point = (point - center) / scale
```

## 7.3 Feature groups

Create three feature configurations:

### Feature set A: engineered features

```text
elbow angle
shoulder angle
hip angle
knee angle
body centroid height
centroid velocity
elbow-angle velocity
```

### Feature set B: normalized landmarks

```text
33 landmarks × x/y/z
```

### Feature set C: combined features

```text
normalized landmarks
+ joint angles
+ velocities
+ visibility values
```

This allows an ablation study.

---

# 8. Phase 5 — Implement baseline methods

You need strong baselines before training a neural model.

## 8.1 Baseline 1: raw angle threshold

For push-ups:

```text
elbow angle below threshold → down
elbow angle above threshold → up
```

For sit-ups:

```text
hip or torso angle below/above threshold → phase
```

## 8.2 Baseline 2: smoothed threshold plus hysteresis

Add:

- moving-average smoothing,
- separate entering and exiting thresholds,
- minimum phase duration,
- minimum time between repetitions.

This is more realistic than a single threshold.

## 8.3 Baseline 3: peak detection

Use a motion signal such as:

- elbow angle,
- hip angle,
- torso height.

Detect local peaks and valleys with minimum distance constraints.

## 8.4 Baseline 4: dynamic time warping

Compare each motion cycle against a reference repetition using DTW. This provides a classical temporal baseline.

Every baseline should produce:

```text
predicted phase sequence
predicted repetition count
processing speed
```

---

# 9. Phase 6 — Train the learned model

## 9.1 Window design

Start with:

```text
Window length: 30–60 frames
Stride: 5 frames
```

At 30 FPS, a 60-frame window represents about 2 seconds.

Test several window sizes:

```text
30 frames
45 frames
60 frames
90 frames
```

The window should capture enough motion to distinguish phases without creating excessive latency.

## 9.2 Model progression

Train models in this order.

### Model A: MLP on individual frames

Purpose:

- sanity check,
- demonstrates that frame-only information is insufficient.

### Model B: 1D CNN

Purpose:

- fast temporal baseline,
- suitable for real-time inference.

### Model C: GRU

Purpose:

- captures sequence direction and temporal order,
- lighter than many LSTM configurations.

### Model D: CNN-GRU

Purpose:

- convolution extracts local motion patterns,
- GRU models temporal progression.

Do not begin with a very large model.

## 9.3 Recommended initial model

```text
Input: 60 frames × feature dimension
Conv1D(64)
Batch normalization
Conv1D(64)
GRU(64)
Dense(32)
Dropout
Dense(3)
```

Output classes:

```text
up
down
transition
```

## 9.4 Optional multi-task model

After phase classification works, add an exercise output:

```text
Shared temporal encoder
    ├── phase classification head
    └── exercise classification head
```

This allows the network to learn both:

- what exercise is being performed,
- which phase the exercise is in.

---

# 10. Phase 7 — Transfer-learning strategy

## 10.1 What is reused

Reuse:

- MediaPipe Pose Landmarker,
- pose extraction code,
- optionally a temporal encoder trained on public exercise data,
- preprocessing and model architecture from the base project.

## 10.2 What is trained

Train:

- input projection layer,
- 1D CNN/GRU temporal layers,
- phase-classification head,
- optional exercise-classification head.

## 10.3 Training stages

### Stage A: public-data training

Train the temporal model to recognize exercise classes or movement phases using public data.

### Stage B: private-data head training

Freeze the temporal encoder and train a new phase head using your own labels.

### Stage C: partial fine-tuning

Unfreeze only the final temporal layer and train with a small learning rate.

Example schedule:

```text
Stage A: learning rate 1e-3
Stage B: learning rate 1e-3, encoder frozen
Stage C: learning rate 1e-5, final encoder layer unfrozen
```

## 10.4 Compare against training from scratch

Run three experiments:

```text
Experiment 1: train from scratch on own data
Experiment 2: public-data pretraining + own-data fine-tuning
Experiment 3: frozen encoder + new classification head
```

Compare performance and training-data requirements.

---

# 11. Phase 8 — Correct repetition-counting logic

The neural model should classify phases. It should not directly increment the count on every prediction.

Use a state machine:

```text
START
  ↓
UP
  ↓ when stable down prediction
DOWN
  ↓ when stable up prediction
UP + 1 repetition
```

Add safeguards:

- require several consecutive predictions,
- minimum time between counts,
- reject incomplete cycles,
- use confidence thresholds,
- reset after long pose loss,
- optionally require a valid bottom position.

Example logic:

```text
UP → DOWN → UP = one repetition
```

Do not count:

```text
UP → DOWN → TRANSITION → DOWN
```

as a completed repetition.

---

# 12. Phase 9 — Fix the notebook’s current technical issues

The current notebook should be updated in these ways:

## 12.1 Correct window indexing

Store the center frame index for every window:

```python
idx.append(start + WINDOW // 2)
```

Use `idx` to reconstruct the frame-level phase signal. Do not assume predictions occur every `WINDOW` frames when the stride is 5.

## 12.2 Avoid training/test leakage

Split by video or subject:

```python
train_videos, test_videos = ...
```

Never randomly split overlapping windows from the same recording.

## 12.3 Use training-only normalization

Fit normalization values on the training set only, then apply them to validation and test sets.

## 12.4 Handle missing landmarks

Do not let NaNs silently become labels or invalid model input. Use:

- interpolation for short gaps,
- confidence features,
- sequence rejection for long gaps.

## 12.5 Validate count output independently

A high phase F1 is not enough. Always compare:

```text
ground-truth repetition count
predicted repetition count
absolute count error
```

---

# 13. Phase 10 — Evaluation protocol

## 13.1 Dataset split

Recommended subject-independent split:

```text
Training: 60% of subjects
Validation: 20% of subjects
Testing: 20% of subjects
```

For example:

```text
Train: subjects 1–5
Validation: subjects 6–7
Test: subject 8
```

Keep all recordings from one person in only one split.

## 13.2 Main metrics

### Phase classification

- accuracy,
- macro precision,
- macro recall,
- macro F1,
- confusion matrix.

### Repetition counting

- mean absolute error,
- mean absolute percentage error where appropriate,
- exact count accuracy,
- precision and recall of detected repetitions,
- missed repetitions,
- false repetitions.

### Real-time performance

- FPS,
- average latency,
- CPU/GPU usage,
- pose extraction time,
- model inference time.

## 13.3 Robustness experiments

Evaluate separately for:

- front view,
- side view,
- 45-degree view,
- slow motion,
- normal motion,
- fast motion,
- good form,
- incomplete form,
- poor lighting,
- partial pose loss.

## 13.4 Required comparisons

At minimum compare:

```text
Threshold baseline
Smoothed threshold baseline
1D CNN
GRU/LSTM
Pretrained temporal model + fine-tuning
```

---

# 14. Phase 11 — Ablation studies

Ablation studies will make the project academically stronger.

## 14.1 Feature ablation

Compare:

```text
angles only
landmarks only
angles + velocities
landmarks + angles + velocities
```

## 14.2 Model ablation

Compare:

```text
1D CNN
GRU
LSTM
CNN-GRU
```

## 14.3 Data ablation

Train with:

```text
25% of data
50% of data
75% of data
100% of data
```

This tests whether transfer learning reduces the amount of required labeled data.

## 14.4 Viewpoint ablation

Train and test:

```text
same-view evaluation
cross-view evaluation
```

---

# 15. Suggested team division

## Member 1 — Data and annotation

Responsible for:

- recording protocol,
- dataset organization,
- annotation format,
- annotation quality checks,
- train/validation/test splits.

## Member 2 — Pose and feature pipeline

Responsible for:

- MediaPipe extraction,
- normalization,
- missing-pose handling,
- angle and velocity features,
- data serialization.

## Member 3 — Baselines and counting

Responsible for:

- threshold counter,
- smoothing,
- hysteresis,
- state machine,
- DTW or peak-detection baseline.

## Member 4 — Neural models and transfer learning

Responsible for:

- 1D CNN,
- GRU/LSTM,
- pretraining,
- fine-tuning,
- hyperparameter experiments.

## Member 5 — Evaluation and demo

Responsible for:

- metrics,
- plots,
- confusion matrices,
- annotated videos,
- real-time application,
- final presentation.

If there are fewer members, combine data with pose preprocessing and combine evaluation with the application.

---

# 16. Suggested eight-week schedule

## Week 1 — Specification and base setup

Deliverables:

- finalized research question,
- exercise definitions,
- repository structure,
- base-project review,
- license review,
- initial notebook cleaned up.

## Week 2 — Data collection protocol

Deliverables:

- recording instructions,
- annotation guide,
- pilot recordings,
- public dataset inventory,
- metadata table.

## Week 3 — Pose extraction pipeline

Deliverables:

- extracted landmarks,
- normalized landmark files,
- feature-generation script,
- missing-pose report,
- sample visualization.

## Week 4 — Baselines

Deliverables:

- threshold counter,
- smoothed/hysteresis counter,
- state machine,
- baseline evaluation on pilot videos.

## Week 5 — Manual annotation and dataset finalization

Deliverables:

- annotated repetition boundaries,
- phase labels,
- subject-based splits,
- quality-control report.

## Week 6 — Neural models

Deliverables:

- 1D CNN,
- GRU/LSTM,
- training curves,
- validation metrics,
- initial comparison with baselines.

## Week 7 — Transfer learning and robustness

Deliverables:

- pretrained/fine-tuned model,
- from-scratch comparison,
- viewpoint and subject-independent evaluation,
- ablation results.

## Week 8 — Demo and final report

Deliverables:

- real-time webcam demo,
- annotated output videos,
- final plots and tables,
- error analysis,
- presentation,
- reproducibility instructions.

---

# 17. Minimum viable version

If time becomes limited, complete this version:

```text
MediaPipe Pose
    ↓
normalized joint angles
    ↓
threshold baseline
    ↓
1D CNN phase classifier
    ↓
state-machine repetition counter
    ↓
subject-independent evaluation
```

Use:

- 5 subjects,
- push-ups and sit-ups,
- side and front views,
- manually annotated phase labels,
- threshold versus CNN comparison.

This is sufficient for a coherent project.

---

# 18. Stretch goals

Only attempt these after the core system works:

- poor-form detection,
- repetition-quality score,
- automatic exercise classification,
- adaptive thresholds per person,
- multi-person tracking,
- mobile deployment,
- quantized TensorFlow Lite model,
- voice feedback,
- support for squats and lunges,
- online personalization using a few calibration repetitions.

---

# 19. Expected final results

Your final report should contain:

1. System architecture diagram.
2. Dataset and annotation description.
3. MediaPipe preprocessing method.
4. Baseline threshold method.
5. Temporal model architecture.
6. Transfer-learning procedure.
7. Subject-independent split strategy.
8. Phase-classification results.
9. Repetition-counting results.
10. FPS and latency measurements.
11. Error analysis.
12. Ablation study.
13. Limitations and future work.

The central comparison should be:

```text
Rule-based angle thresholding
versus
Learned temporal phase recognition
```

The most defensible implementation strategy is:

```text
Use MediaPipe as the frozen pose backbone.
Use the existing push-up tracker as an engineering reference.
Use your notebook as the experimental starting point.
Train a small CNN/GRU temporal model.
Fine-tune it with manually annotated videos.
Use a state machine to convert phase predictions into repetitions.
Evaluate on unseen subjects and camera angles.
```