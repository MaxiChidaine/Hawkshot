import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. RUL target construction

    The objective of this notebook is to transform the run-to-failure trajectories into a supervised regression dataset.

    In supervised learning, each observaion is associated with:

    - **feature**, which represent the information available to the model at the time of prediction.
    - **target**, which represents the value that the model must learn to predict.

    For FD001, the target is the Remaining Useful Life (RUL), defined as the number of operating cycles remaining before the end of an engine trajectories.

    Because the training trajectories are complete run-to-failure histories, the final cycle of each engine is known and can be used to construct the target.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import numpy as np

    from hawkshot.data.cmapss import (
        SENSOR_COLUMNS,
        OPERATIONAL_COLUMNS,
        filter_constant_sensors,
        load_fd001,
    )

    df_raw = load_fd001("data/raw/cmapss")

    df_filtered, removed_sensors = filter_constant_sensors(df_raw)

    available_sensors = [
        sensor for sensor in SENSOR_COLUMNS if sensor in df_filtered.columns
    ]

    df_rul = df_filtered.copy()
    return (
        OPERATIONAL_COLUMNS,
        available_sensors,
        df_rul,
        mo,
        np,
        pd,
        plt,
        train_test_split,
    )


@app.cell
def _(df_rul):
    df_rul["max_cycle"] = df_rul.groupby("engine_id")["cycle"].transform("max")

    df_rul["rul"] = df_rul["max_cycle"] - df_rul["cycle"]
    return


@app.cell
def _(df_rul):
    df_rul["rul"].describe()
    return


@app.cell
def _(df_rul, plt):
    _fig, _ax = plt.subplots(figsize=(10, 6))

    _ax.hist(
        df_rul["rul"],
        bins=30,
    )

    _ax.set_xlabel("Remaining Useful Life")
    _ax.set_ylabel("Number of observations")
    _ax.set_title("Distribution of linear RUL")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Linear and capped RUL

    The linear RUL provides the true number of cycles remaining before failure for each training observation. However, the previous sensor analysis showed that the relationship between sensor measurements and engine age is considerably weaker during early life.

    A capped RUL is therefore introduced as a modelling assumption. Values above 125 cycles are replaced by 1é5, while lower values remain unchanged.

    This does not imply that engines with more than 125 remaining cycles have the same physical condition. Instead, it reflects the decision not to require the model to distinguish precisely between high-RUL states for which the available sensor measurements provide relatively limited degradation information.

    The threshold of 125 cycles is treated as an initial modelling choice rather than a physically validated boundary and may later be reassessed through model performance.
    """)
    return


@app.cell
def _(df_rul):
    rul_cap = 125

    capped_share = (df_rul["rul"] > rul_cap).mean()

    capped_share
    return (rul_cap,)


@app.cell
def _(df_rul, rul_cap):
    df_rul["rul_capped"] = df_rul["rul"].clip(upper=rul_cap)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Predictor definition and leakage analysis

    Before preparing the modelling dataset, each varaible must be classified according to the information that would be available at prediction time.

    A valid predictor must be observable without requiring knowledge of the engine's future trajectory.

    The retained sensor measurements and operational setting satisfy this condition and are therefore considered candidate features. The sensors will be used as they proved to have a strong correlation with engine age but the same cannot be said for the operational setting as, although they do not represent a data leakage, there is no information regarding their usefulness to predict RUL. Their potential as features will be assessed below.

    `engine_id` is required to identify trajectories and to create engine-level data splits, but it is not used as a predictive feature because its numerical value is only an arbitrary identifier.

    The RUL target and `max_cycle` must never be provided to the model. `max_cycle` directly depends on the future failure time and would therefore introduce target leakage.

    The operating `cycle` is different. It is known at prediction time and does not constitute data leakage. However, because it represents engine age, a model may rely heavily on it instead of extracting degradation information from sensor measurements. It's predictive contribution will therefore be evaluated separately.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Operational setting assessment

    The three operational settings were evaluated before defining the initial predictive feature set.

    `operational_setting_3` is strictly constant across all observations and therefore contains no predictive information.

    `operational_setting_1` and `operational_setting_2` vary within individual engine trajectories. However, their median absolute Spearman correlations with operating cycle are both approximately `0.04`, indicating no meaningful monotonic relationship with engine age.

    Their relationships with the 14 retained sensors are similarly weak. Across all setting-sensor combinations, median absolute correlations remain close to zero, with the strongest value only reaching `0.06`.

    These results provide no evidence that the operational settings either track engine ageing directly or explain a substantial proportion of the observed sensor evolution. They are therefore excluded from the initial modelling feature set.

    This conclusion concerns monotonic relationships only. Possible nonlinear effect or interactions cannot be ruled out from correlation analysis alone and may later be reassessed though model comparison.
    """)
    return


@app.cell
def _(OPERATIONAL_COLUMNS, df_rul):
    df_rul.groupby("engine_id")[OPERATIONAL_COLUMNS].nunique().describe().T
    return


@app.cell
def _():
    active_settings = [
        "operational_setting_1",
        "operational_setting_2",
    ]
    return (active_settings,)


@app.cell
def _(active_settings, df_rul, pd):
    _setting_spearman_results = []

    for _setting in active_settings:
        for _engine_id in df_rul["engine_id"].unique():
            _engine_data = df_rul[df_rul["engine_id"] == _engine_id]

            _rho = _engine_data["cycle"].corr(
                _engine_data[_setting],
                method="spearman",
            )

            _setting_spearman_results.append(
                {"engine_id": _engine_id, "setting": _setting, "spearman_rho": _rho}
            )

    setting_spearman_by_engine = pd.DataFrame(_setting_spearman_results)

    return (setting_spearman_by_engine,)


@app.cell
def _(setting_spearman_by_engine):
    setting_spearman_summary = (
        setting_spearman_by_engine.assign(
            absolute_rho=lambda df: df["spearman_rho"].abs()
        )
        .groupby("setting")
        .agg(
            median_rho=("spearman_rho", "median"),
            median_absolute_rho=("absolute_rho", "median"),
            min_rho=("spearman_rho", "min"),
            max_rho=("spearman_rho", "max"),
        )
    )

    setting_spearman_summary.round(3)
    return


@app.cell
def _(active_settings, available_sensors, df_rul, pd):
    _setting_sensor_results = []

    for _setting in active_settings:
        for _sensor in available_sensors:
            for _engine_id in df_rul["engine_id"].unique():
                _engine_data = df_rul[df_rul["engine_id"] == _engine_id]

                _rho = _engine_data[_setting].corr(
                    _engine_data[_sensor],
                    method="spearman",
                )

                _setting_sensor_results.append(
                    {
                        "engine_id": _engine_id,
                        "setting": _setting,
                        "sensor": _sensor,
                        "spearman_rho": _rho,
                    }
                )

    setting_sensor_spearman_by_engine = pd.DataFrame(_setting_sensor_results)
    return (setting_sensor_spearman_by_engine,)


@app.cell
def _(setting_sensor_spearman_by_engine):
    setting_sensor_summary = (
        setting_sensor_spearman_by_engine.assign(
            absolute_rho=lambda df: df["spearman_rho"].abs()
        )
        .groupby(["setting", "sensor"])
        .agg(
            median_rho=("spearman_rho", "median"),
            median_absolute_rho=("absolute_rho", "median"),
            q1_rho=("spearman_rho", lambda values: values.quantile(0.25)),
            q3_rho=("spearman_rho", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )

    setting_sensor_summary["iqr_rho"] = (
        setting_sensor_summary["q3_rho"] - setting_sensor_summary["q1_rho"]
    )

    setting_sensor_summary.sort_values(
        "median_absolute_rho",
        ascending=False,
    ).round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Operating cycle as predictor

    The operating cycle requires separate considaration. Unlike `max_cycle`, it is known at prediction time and therefore does not introduce data leakage.

    However, `cycle` represents the current operational age of the engine and may already provide substantial information about its remaining lifetime. A model including this variable could therefore acheive improved performance partly by exploiting engine age rather than degradation signals measured by the sensors.

    For this reasong, `cycle` is not discarded. Instead, separate feature sets will be prepared so that its contribution can later be evaluated explicitly :

    -**Sensor only**, using the 14 retained sensor measurements.
    - **Sensors + cycle**, adding the current engine age.

    A cycle-only baseline may also be evaluated during modelling. Comparing these configurations will make it possible to determine whether the sensor measurements provide prodictive information beyong engine age alone.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Engine-level train/validation split

    The modelling data must be separated into training and validation sets before any data-dependent preprocessing is learned.

    A random row-level split would be inappropriate for FD001 because multiple observations belong to the same engine trajectory. Such as split could place different cycles from the same engine in both training and validation data, allowing the model to be evaluated on engines that it has already partially observed.

    The split is therefore performed at the engine-level. All observations from one engine are assigned exclusively to either the training or validation set.

    Eighty engines are used for training and twenty are reserved for validation. The validation engines remain completely unseen during model fitting, making the evlation more representative of prediction on new engines.

    The official FD001 test trajectories will remain separate and will be used later for final model evaluation.
    """)
    return


@app.cell
def _(df_rul, train_test_split):
    engine_ids = df_rul["engine_id"].unique()

    train_engines, validation_engines = train_test_split(
        engine_ids, test_size=0.2, random_state=42
    )
    return train_engines, validation_engines


@app.cell
def _(df_rul, train_engines, validation_engines):
    df_train = df_rul[df_rul["engine_id"].isin(train_engines)].copy()

    df_validation = df_rul[df_rul["engine_id"].isin(validation_engines)].copy()
    return df_train, df_validation


@app.cell
def _(df_train, df_validation, pd):
    train_lifetime = df_train.groupby("engine_id")["max_cycle"].first()

    validation_lifetime = df_validation.groupby("engine_id")["max_cycle"].first()

    pd.DataFrame(
        {
            "train": train_lifetime.describe(),
            "validation": validation_lifetime.describe(),
        }
    ).round(1)
    return


@app.cell
def _(available_sensors):
    sensor_features = available_sensors

    sensor_cycle_features = [
        "cycle",
        *available_sensors,
    ]
    return sensor_cycle_features, sensor_features


@app.cell
def _(df_train, df_validation, sensor_cycle_features, sensor_features):
    X_train_sensors = df_train[sensor_features]
    X_validation_sensors = df_validation[sensor_features]

    X_train_sensors_cycle = df_train[sensor_cycle_features]
    X_validation_sensors_cycle = df_validation[sensor_cycle_features]

    Y_train = df_train["rul_capped"]
    Y_validation = df_validation["rul_capped"]
    return (
        X_train_sensors,
        X_validation_sensors,
        X_train_sensors_cycle,
        X_validation_sensors_cycle,
        Y_train,
        Y_validation,
    )


@app.cell
def _():
    temporal_config = {
        "mean": [5, 10, 20],
        "delta": [5, 10],
        "slope": [10, 20],
    }
    return (temporal_config,)


@app.cell
def _(np):
    def compute_slope(values):
        if len(values) < 2:
            return 0.0

        time_steps = np.arange(len(values))

        slope, _ = np.polyfit(
            time_steps,
            values,
            1,
        )

        return slope

    return (compute_slope,)


@app.cell
def _(compute_slope):
    def add_temporal_features(df, sensors, config):
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
                        lambda window_values: window_values.iloc[-1]
                        - window_values.iloc[0]
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

    return (add_temporal_features,)


@app.cell
def _(
    add_temporal_features,
    available_sensors,
    df_train,
    df_validation,
    temporal_config,
):
    df_train_temporal = add_temporal_features(
        df_train,
        available_sensors,
        temporal_config,
    )

    df_validation_temporal = add_temporal_features(
        df_validation,
        available_sensors,
        temporal_config,
    )

    cycle_features = ["cycle"]

    raw_sensor_features = available_sensors

    raw_sensor_cycle_features = [*available_sensors, "cycle"]

    temporal_features = [
        column
        for column in df_train_temporal.columns
        if any(suffix in column for suffix in ["_mean_", "_delta_", "_slope_"])
    ]

    raw_temporal_features = [
        *available_sensors,
        *temporal_features,
    ]
    return (
        df_train_temporal,
        df_validation_temporal,
        cycle_features,
        raw_sensor_features,
        raw_sensor_cycle_features,
        temporal_features,
        raw_temporal_features,
    )


@app.cell
def _(df_train_temporal, train_engines):
    df_train_temporal.loc[
        df_train_temporal["engine_id"] == train_engines[0],
        [
            "engine_id",
            "cycle",
            "sensor_2",
            "sensor_2_mean_10",
            "sensor_2_delta_5",
            "sensor_2_slope_20",
        ],
    ].head(25).round(3)
    return


if __name__ == "__main__":
    app.run()
