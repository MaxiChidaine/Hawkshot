import numpy as np


def compute_slope(values):
    """Compute the slope of the first-degree linear fit for the given values."""
    if len(values) < 2:
        return 0.0

    time_steps = np.arange(len(values))

    slope, _ = np.polyfit(
        time_steps,
        values,
        1,
    )

    return slope


def add_temporal_features(df, sensors, config):
    """Add temporal features to the DataFrame based on the specified sensors and configuration."""
    df_features = df.sort_values(["engine_id", "cycle"]).copy()

    for sensor in sensors:
        grouped_sensor = df_features.groupby("engine_id")[sensor]

        for window in config.get("mean", []):
            df_features[f"{sensor}_mean_{window}"] = grouped_sensor.transform(
                lambda values: values.rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )
        for window in config.get("delta", []):
            df_features[f"{sensor}_delta_{window}"] = grouped_sensor.transform(
                lambda values: values.rolling(
                    window=window,
                    min_periods=1,
                ).apply(
                    lambda window_values: window_values.iloc[-1] - window_values.iloc[0]
                )
            )

        for window in config.get("slope", []):
            df_features[f"{sensor}_slope_{window}"] = grouped_sensor.transform(
                lambda values: values.rolling(
                    window=window,
                    min_periods=1,
                ).apply(compute_slope)
            )
    return df_features
