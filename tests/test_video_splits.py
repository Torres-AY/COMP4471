from exercise_rep_counter.data.manifest import SampleRecord
from exercise_rep_counter.data.video_splits import split_by_record


def test_video_split_assigns_every_record_once():
    records = [
        SampleRecord(str(index), "kaggle", "unknown", "pushup", "")
        for index in range(10)
    ]

    split_records = split_by_record(records, seed=3)

    assert len(split_records) == 10
    assert {record.split for record in split_records} == {"train", "validation", "test"}
