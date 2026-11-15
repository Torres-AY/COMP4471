# Introduction {#sec:intro}

Exercise recognition has become an important application of computer
vision and artificial intelligence, particularly in fitness tracking,
personal training, and rehabilitation systems. Traditional approaches
that rely solely on raw video frames often struggle with variations in
camera angle, background, lighting, and clothing. A more robust
alternative is to first extract the human skeleton (pose keypoints) and
then perform action recognition on the resulting skeletal sequence. This
two-stage pipeline reduces the influence of appearance-related noise and
focuses the model on the essential motion patterns of the exercise.\
\
In this project, we aim to develop an AI system that can automatically
recognise common bodyweight exercises, with a primary focus on push-ups.
The system follows a skeleton-based approach: a pose estimation model
first detects the human body joints in each video frame, producing a
temporal sequence of skeletal keypoints. These keypoints are then fed
into a downstream recognition model that classifies the performed
exercise.

## Roadmap

Our overall plan consists of three main stages. First, we will select
and prepare suitable datasets that contain push-up and related exercise
videos. Second, we will extract skeletal representations using an
off-the-shelf pose estimation model. Third, we will train and evaluate
sequence models (such as LSTM, Transformer, or graph-based networks) on
the extracted skeletons to classify the exercises. At this milestone, we
focus on dataset preparation, skeleton extraction, and initial
recognition experiments to validate the feasibility of the proposed
pipeline.

# Problem statement {#sec:problem}

The task of this project is to recognise bodyweight exercises, with a
primary focus on push-ups, from video input using a skeleton-based
approach. Given a video sequence of a person performing an exercise, the
system must first extract the human skeleton (pose keypoints) from each
frame and then classify the action based on the temporal sequence of
these keypoints.

## Dataset

We will use three complementary datasets. First, we will collect our own
multi-view webcam dataset from approximately 5--8 volunteers performing
push-ups and sit-ups from front, side, and 45-degree angles, targeting
at least 1,000 manually annotated repetitions. Second, we will use the
Kaggle Real-Time Exercise Recognition
Dataset[@riccio_realtime_exercise_dataset], which contains videos of
exercises---push-ups. This dataset will be used to pretrain the
exercise-recognition and temporal feature-extraction components because
it may not contain precise repetition boundaries, we will not use it as
the sole source of repetition-count ground truth. Third, we will use the
UI-PRMD dataset[@vakanski_ui_prmd], containing 10 subjects performing 10
repetitions of 10 rehabilitation movements with Vicon and Kinect
skeletal data, segmented repetitions, joint positions, angles, and
correct/incorrect examples, for auxiliary temporal-model pretraining and
baseline analysis. Our own dataset will be divided by subject into
training ($60\%$), validation ($20\%$), and testing ($20\%$) sets for
final evaluation. Public datasets will be used for pretraining, while
the final push-up and sit-up repetition-counting results will be
reported primarily on our manually annotated webcam test set.

## Expected results

The system is expected to output the predicted exercise class for a
given input video. For the current milestone, the primary target is
binary or multi-class recognition of push-ups (and possibly a small set
of related exercises such as squat or sit-up). The model should
correctly classify the action based on the extracted skeleton sequence.\
\
Performance will be measured using standard classification metrics on a
held-out test set:

- **Accuracy**: Overall percentage of correctly classified videos.

- **Precision, Recall, and F1-score**: Computed per class (especially
  for the push-up class) and as macro/micro averages to account for
  possible class imbalance.

- **Confusion Matrix**: To analyse common misclassifications between
  similar exercises.

Success at this milestone will be indicated by a reasonable recognition
accuracy (target $> 70$--$80\%$ on the test set) using the
skeleton-based pipeline, demonstrating that the extracted pose sequences
contain sufficient information for exercise classification.

# Technical Approach

## Method

Given a video sequence of a person performing a bodyweight exercise, the
system extracts a human skeleton from each frame and classifies the
action from the temporal sequence of keypoints. The pipeline has five
stage. pose extraction with a frozen pretrained model, coordinate
normalization and missing-value handling,feature extraction,temporal
classification of movement phases, and conversion of phase predictions
into a repetition count via a state machine.

We use MediaPipe Pose Landmarker in RunningMode.VIDEO, called per frame
with a timestamp computed from the frame index and the video frame rate.
The model returns 33 landmarks per person, each with normalized x and y
in \[0, 1\], a relative depth z, and a visibility score.

## Feature sets

Feature sets We define three feature configurations so that the
contribution of each can be isolated in the ablation study. Feature set
A contains seven physically interpretable channels per frame: elbow
angle, shoulder angle, hip angle, knee angle , body centroid height,
centroid velocity, and elbow-angle velocity. Joint angles are computed
with the arccos dot-product formula at each joint triple. Feature set B
contains the 33 normalized landmarks × x/y/z, giving 99 channels that
let the model discover its own spatial relationships. Feature set C
concatenates normalized landmarks, joint angles, velocities, and
visibility scores. Set A is the most sample-efficient and interpretable;
set B is the most expressive; set C is the candidate for the final
model, with the ablation deciding whether the extra dimensionality pays
for itself on a small dataset.

## models

The classifiers operate on sliding windows of the feature sequence and
predict an exercise label for each window. A window of 60 frames covers
roughly two seconds at 30 FPS, which is enough to see a full repetition
without introducing excessive latency. We test window lengths of 30, 45,
60, and 90 frames with a stride of 5. Window-level predictions are
aggregated to a video-level label by majority vote.

We train three models in order of increasing complexity, so that a
failure at any level is diagnosable:

MLP on individual frames --- a sanity check that should underperform,
demonstrating that frame-only information cannot resolve whether the
body is rising or lowering.

1D CNN --- the fast real-time baseline; convolution over time captures
local motion patterns such as the sharp change of direction at the
bottom of a push-up.

GRU --- captures the direction and order of the sequence, which matters
because the same joint angle occurs on the way down and on the way up;
it is lighter than an LSTM at comparable accuracy.

# Intermediate results

We built and validated the full pose-extraction and feature-processing
pipeline on real video. MediaPipe Pose Landmarker extracted 33 landmarks
per frame from a 560-frame screen recording, producing a landmark tensor
of shape \[560, 33, 4\] with normalized x, y, z coordinates and a
visibility score per keypoint. From this tensor we computed a 7-channel
feature matrix of shape \[560, 7\] containing elbow, shoulder, hip, and
knee angles (each averaged over left and right sides), body centroid
height, centroid velocity, and elbow-angle velocity. Joint angles use
the arccos dot-product formula at each joint triple, and velocity
channels are frame-aligned with the first frame set to zero.

We implemented the rule-based angle-threshold counter. For push-ups,
elbow angle below 90° is labelled down and above 160° is labelled up
after moving-average smoothing, and a repetition is counted on each down
up transition. An exercise-type heuristic compares the range of
elbow-angle swing against the range of hip-angle swing, with a larger
elbow swing indicating a push-up. On the pilot clip the heuristic
correctly identified the recording as a push-up,and successfully count
the correct number.
