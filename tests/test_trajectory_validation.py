import pandas as pd
import pytest

from hawkshot.data.validation import validate_engine_trajectories


def test_accepts_unordered_cycles():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [3, 1, 2, 2, 1],
        }
    )

    validate_engine_trajectories(df)


def test_accepts_multiple_engines_and_short_histories():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 3, 3],
            "cycle": [5, 6, 1, 2, 3],
        }
    )
    validate_engine_trajectories(df)


def test_rejects_duplicate_cycles():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [1, 2, 2],
        }
    )

    with pytest.raises(ValueError, match="Duplicate"):
        validate_engine_trajectories(df)


def test_rejects_missing_cycles():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [1, 3, 4],
        }
    )

    with pytest.raises(ValueError, match="Missing cycles"):
        validate_engine_trajectories(df)


def test_rejects_non_integer_cycles():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1.0, 2.5],
        }
    )

    with pytest.raises(ValueError, match="integer type"):
        validate_engine_trajectories(df)


def test_validation_does_not_modify_dataframe():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [3, 1, 2],
        }
    )

    df_copy = df.copy()
    validate_engine_trajectories(df)
    pd.testing.assert_frame_equal(df, df_copy)
