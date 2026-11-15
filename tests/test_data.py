import numpy as np
import pytest
import cv2

from exercise_rep_counter.data.kaggle_loader import discover_videos
from exercise_rep_counter.data.manifest import SampleRecord, read_manifest, write_manifest
from exercise_rep_counter.data.splits import split_by_subject
from exercise_rep_counter.data.uiprmd_loader import discover_sequences, load_sequence


def test_kaggle_discovery_infers_exercise(tmp_path):
    video = tmp_path / "push-ups" / "sample.avi"
    video.parent.mkdir()
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (16, 16)
    )
    writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()

    records = discover_videos(tmp_path)

    assert records[0].exercise == "pushup"
    assert records[0].dataset == "kaggle"


def test_uiprmd_discovery_extracts_subject_and_movement(tmp_path):
    sequence = tmp_path / "Subject01" / "Movement05" / "rep03.npy"
    sequence.parent.mkdir(parents=True)
    np.save(sequence, np.ones((10, 8), dtype=np.float32))

    records = discover_sequences(tmp_path)

    assert records[0].subject_id == "s01"
    assert records[0].exercise == "m05"
    assert records[0].repetition_id == "3"


def test_uiprmd_loader_reads_comma_delimited_text(tmp_path):
    sequence = tmp_path / "m01_s01_e01_angles.txt"
    sequence.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n", encoding="utf-8")

    result = load_sequence(sequence)

    assert result.shape == (2, 3)
    assert result.dtype == np.float32


def test_uiprmd_loader_rejects_nonfinite_values(tmp_path):
    sequence = tmp_path / "sequence.npy"
    np.save(sequence, np.array([[1.0, np.nan]], dtype=np.float32))

    with pytest.raises(ValueError, match="NaN"):
        load_sequence(sequence)


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.csv"
    records = [
        SampleRecord(
            sample_id="sample",
            dataset="kaggle",
            subject_id="unknown",
            exercise="pushup",
            source_path="sample.mp4",
        )
    ]

    write_manifest(records, path)

    assert read_manifest(path) == records


def test_subject_split_keeps_subjects_together():
    records = [
        SampleRecord(str(index), "uiprmd", f"s{subject}", "m01", "")
        for subject in range(1, 6)
        for index in range(subject * 2, subject * 2 + 2)
    ]

    split_records = split_by_subject(records, seed=7)
    assignments = {}
    for record in split_records:
        assignments.setdefault(record.subject_id, record.split)
        assert assignments[record.subject_id] == record.split

    assert {record.split for record in split_records} == {"train", "validation", "test"}
