import pytest

from exercise_rep_counter.annotation.phase_csv import (
    labels_for_video,
    read_phase_intervals,
)


def test_phase_annotations_expand_inclusive_intervals(tmp_path):
    path = tmp_path / "phases.csv"
    path.write_text(
        "video_id,start_frame,end_frame,phase,rep_id\n"
        "v1,0,1,down,1\n"
        "v1,2,3,up,1\n",
        encoding="utf-8",
    )

    intervals = read_phase_intervals(path)
    phases, repetitions = labels_for_video(intervals, "v1", 4)

    assert phases == ["down", "down", "up", "up"]
    assert repetitions == ["1", "1", "1", "1"]


def test_phase_annotations_reject_overlap(tmp_path):
    path = tmp_path / "phases.csv"
    path.write_text(
        "video_id,start_frame,end_frame,phase,rep_id\n"
        "v1,0,2,down,1\n"
        "v1,2,3,up,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Overlapping"):
        read_phase_intervals(path)
