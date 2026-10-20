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
            "engine_id": [1, 1, 1, 1],
            "cycle": [1, 2, 3, 4],
            "sensor_2": [10.0, 12.0, 14.0, 16.0],
        }
    )

    config = {
        "mean": [3],
        "delta": [3],
        "slope": [3],
    }

    feature_cols = [
        "sensor_2_mean_3",
        "sensor_2_delta_3",
        "sensor_2_slope_3",
    ]

    # Calculate features before modifying future data.
    result_before = add_temporal_features(df, ["sensor_2"], config)

    # Modify the sensor value at cycle 4.
    df_modified = df.copy()
    df_modified.loc[df_modified["cycle"] == 4, "sensor_2"] = 1000.0

    # Recalculate features after the modification.
    result_after = add_temporal_features(df_modified, ["sensor_2"], config)

    # Past features must remain unchanged.
    past_before = result_before.loc[result_before["cycle"] <= 3, feature_cols]
    past_after = result_after.loc[result_after["cycle"] <= 3, feature_cols]

    pd.testing.assert_frame_equal(
        past_before.reset_index(drop=True),
        past_after.reset_index(drop=True),
    )

    # The modified future value must affect its own features.
    future_before = result_before.loc[result_before["cycle"] == 4, feature_cols]
    future_after = result_after.loc[result_after["cycle"] == 4, feature_cols]

    assert not future_before.equals(future_after)


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


def test_temporal_features_sort_cycles_before_calculation():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [3, 1, 2],
            "sensor_2": [30.0, 10.0, 20.0],
        }
    )

    config = {
        "mean": [2],
        "delta": [2],
        "slope": [2],
    }

    result = add_temporal_features(df, ["sensor_2"], config)

    assert result["cycle"].tolist() == [1, 2, 3]

    assert result["sensor_2_mean_2"].tolist() == (pytest.approx([10.0, 15.0, 25.0]))
    assert result["sensor_2_delta_2"].tolist() == (pytest.approx([0.0, 10.0, 10.0]))
    assert result["sensor_2_slope_2"].tolist() == (pytest.approx([0.0, 10.0, 10.0]))


def test_temporal_features_handle_short_histories():
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "sensor_2": [10.0, 20.0, 30.0],
        }
    )

    config = {
        "mean": [1, 5],
        "delta": [1, 5],
        "slope": [1, 5],
    }

    result = add_temporal_features(df, ["sensor_2"], config)

    assert result["sensor_2_mean_1"].tolist() == (pytest.approx([10.0, 20.0, 30.0]))
    assert result["sensor_2_mean_5"].tolist() == (pytest.approx([10.0, 15.0, 20.0]))

    assert result["sensor_2_delta_1"].tolist() == (pytest.approx([0.0, 0.0, 0.0]))
    assert result["sensor_2_delta_5"].tolist() == (pytest.approx([0.0, 10.0, 20.0]))

    assert result["sensor_2_slope_1"].tolist() == (pytest.approx([0.0, 0.0, 0.0]))
    assert result["sensor_2_slope_5"].tolist() == (pytest.approx([0.0, 10.0, 10.0]))
