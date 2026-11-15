from exercise_rep_counter.config import (
    EXERCISE_LABELS,
    PHASE_LABELS,
    TrainingConfig,
    data_root,
)


def test_training_defaults_match_project_plan():
    config = TrainingConfig()

    assert config.window_lengths == (30, 45, 60, 90)
    assert config.stride == 5
    assert config.target_fps == 30
    assert PHASE_LABELS == ("up", "down", "transition")
    assert EXERCISE_LABELS == ("pushup", "situp")


def test_data_root_is_inside_repository():
    assert data_root().name == "data"
