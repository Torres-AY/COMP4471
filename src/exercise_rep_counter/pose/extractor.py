"""MediaPipe Pose Landmarker extraction for video files."""

from dataclasses import replace
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from exercise_rep_counter.data.manifest import SampleRecord, read_manifest, write_manifest
from exercise_rep_counter.data.video_splits import split_by_record
from .features import normalize_landmarks


def extract_video(
    video_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
) -> tuple[int, float, float]:
    """Extract one person's landmarks and save them as a compressed NPZ.

    Returns ``(frame_count, fps, missing_fraction)``. Missing frames remain in
    the sequence as zero-valued landmarks with zero visibility.
    """

    video = Path(video_path)
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"MediaPipe model asset does not exist: {model}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Video has invalid FPS: {video}")

    landmarks: list[np.ndarray] = []
    missing = 0
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect_for_video(
                image, int(frame_index * 1000 / fps)
            )
            if not result.pose_landmarks:
                values = np.zeros((33, 4), dtype=np.float32)
                missing += 1
            else:
                detected = result.pose_landmarks[0]
                if len(detected) != 33:
                    capture.release()
                    raise ValueError(f"Expected 33 landmarks, got {len(detected)}")
                values = np.array(
                    [[point.x, point.y, point.z, point.visibility or 0.0] for point in detected],
                    dtype=np.float32,
                )
            landmarks.append(values)
            frame_index += 1
    capture.release()
    if not landmarks:
        raise ValueError(f"Video contains no readable frames: {video}")

    raw = np.stack(landmarks)
    normalized = normalize_landmarks(raw)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, landmarks=raw, normalized_landmarks=normalized)
    return len(raw), fps, missing / len(raw)


def extract_manifest(
    manifest_path: str | Path,
    source_root: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    output_manifest: str | Path,
) -> list[SampleRecord]:
    """Extract all videos in a Kaggle manifest and write processed metadata."""

    source_base = Path(source_root)
    destination = Path(output_root)
    records: list[SampleRecord] = []
    for record in read_manifest(manifest_path):
        output_name = f"{record.sample_id}.npz"
        output_path = destination / output_name
        num_frames, fps, missing_fraction = extract_video(
            source_base / record.source_path, model_path, output_path
        )
        if missing_fraction > 0.5:
            raise ValueError(
                f"More than half of frames lost pose for {record.sample_id}: "
                f"{missing_fraction:.1%}"
            )
        records.append(
            replace(
                record,
                feature_path=str(Path(output_root) / output_name),
                num_frames=num_frames,
                fps=fps,
            )
        )
    records = split_by_record(records)
    write_manifest(records, output_manifest)
    return records
