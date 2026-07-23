from pathlib import Path

import pandas as pd
import pytest

from hawkshot.data.cmapss import load_fd001, filter_constant_sensors, SENSOR_COLUMNS


def test_load_fd001_assigns_expected_columns(tmp_path: Path) -> None:
    data_file = tmp_path / "train_FD001.txt"

    # One artificial row of data with the expected number of columns.
    row = " ".join(str(i) for i in range(26))
    data_file.write_text(f"{row}\n", encoding="utf-8")

    df = load_fd001(tmp_path)

    expected_columns = (
        ["engine_id", "cycle"]
        + [f"operational_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    assert df.columns.tolist() == expected_columns
    assert df.shape == (1, 26)


def test_load_fd001_raises_if_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError, match=r"FD001 training file not found : .*train_FD001.txt"
    ):
        load_fd001(tmp_path)


def test_filter_constant_sensors_removes_expected_sensors():
    sensor_data = {sensor: [0.0, 1.0] for sensor in SENSOR_COLUMNS}

    # Reproduce the stricly contsant sensors identified in FD001.
    striclty_constant_sensors = [
        "sensor_1",
        "sensor_5",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    ]

    for sensor in striclty_constant_sensors:
        sensor_data[sensor] = [42.0, 42.0]

    # sensor varies slighlty but is explicitly treated as effectively constant.
    sensor_data["sensor_6"] = [26.60, 26.61]

    df = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 2],
            **sensor_data,
        }
    )

    filtered_df, removed_sensors = filter_constant_sensors(df)

    expected_removed_sensors = [
        "sensor_1",
        "sensor_5",
        "sensor_6",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    ]

    assert removed_sensors == expected_removed_sensors

    for sensor in expected_removed_sensors:
        assert sensor not in filtered_df.columns


def test_filter_constant_sensors_does_not_modify_input() -> None:
    sensor_data = {sensor: [0.0, 1.0] for sensor in SENSOR_COLUMNS}

    sensor_data["sensor_1"] = [42.0, 42.0]
    sensor_data["sensor_6"] = [26.60, 26.61]

    df = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 2],
            **sensor_data,
        }
    )

    original_df = df.copy(deep=True)

    filter_constant_sensors(df)

    pd.testing.assert_frame_equal(df, original_df)
