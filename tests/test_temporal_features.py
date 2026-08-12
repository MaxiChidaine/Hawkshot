import pytest
import pandas as pd

from hawkshot.features.temporal import compute_slope, add_temporal_features


def test_compute_slope_returns_zero_for_single_value():
    values = [5]
    slope = compute_slope(values)
    assert slope == 0.0


def test_compute_slope_returns_expected_slope():
    values = [1, 2, 3, 4, 5]
    slope = compute_slope(values)
    assert slope == pytest.approx(1.0)


def test_add_temporal_features_does_not_mix_engines():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "sensor_2": [10.0, 12.0, 14.0, 100.0, 120.0, 140.0],
        }
    )

    config = {
        "mean": [3],
        "delta": [3],
        "slope": [3],
    }

    result = add_temporal_features(df, ["sensor_2"], config)

    engine_2_first_cycle = result[
        (result["engine_id"] == 2) & (result["cycle"] == 1)
    ].iloc[0]

    assert engine_2_first_cycle["sensor_2_mean_3"] == 100.0
    assert engine_2_first_cycle["sensor_2_delta_3"] == 0.0
    assert engine_2_first_cycle["sensor_2_slope_3"] == 0.0


def test_add_temporal_features_do_not_use_future_value():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "sensor_2": [10.0, 12.0, 14.0],
        }
    )

    config = {
        "mean": [3],
        "delta": [3],
        "slope": [3],
    }

    result_before = add_temporal_features(df, ["sensor_2"], config)

    df_modified = df.copy()
    df_modified.loc[df_modified["cycle"] == 4, "sensor_2"] = 1000.0

    result_after = add_temporal_features(df_modified, ["sensor_2"], config)

    cycle_3_before = result_before.loc[
        result_before["cycle"] == 3,
        [
            "sensor_2_mean_3",
            "sensor_2_delta_3",
            "sensor_2_slope_3",
        ],
    ]

    cycle_3_after = result_after.loc[
        result_after["cycle"] == 3,
        [
            "sensor_2_mean_3",
            "sensor_2_delta_3",
            "sensor_2_slope_3",
        ],
    ]

    pd.testing.assert_frame_equal(
        cycle_3_before.reset_index(drop=True),
        cycle_3_after.reset_index(drop=True),
    )


def test_temporal_features_use_independent_windows():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5],
            "sensor_2": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    config = {
        "mean": [3],
        "delta": [2],
        "slope": [4],
    }

    result = add_temporal_features(df, ["sensor_2"], config)

    cycle_5 = result.loc[result["cycle"] == 5].iloc[0]

    assert cycle_5["sensor_2_mean_3"] == pytest.approx(40.0)
    assert cycle_5["sensor_2_delta_2"] == pytest.approx(10.0)
    assert cycle_5["sensor_2_slope_4"] == pytest.approx(10.0)


def test_add_temporal_features_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "sensor_2": [10.0, 12.0, 14.0],
        }
    )

    config = {
        "mean": [3],
        "delta": [3],
        "slope": [3],
    }

    df_copy = df.copy(deep=True)

    add_temporal_features(df, ["sensor_2"], config)

    pd.testing.assert_frame_equal(df, df_copy)
