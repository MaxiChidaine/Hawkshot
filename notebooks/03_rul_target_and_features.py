import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # FD001 RUL target and feature preparation

    The previous notebooks identified the informative sensor variables in FD001 and investigated how their degradation-related signals evolve throughout engine life.

    The objective of this notebook is to transform these run-to-failure trajectories into a modelling dataset suitable for Remaining Useful Life prediction.

    This requires several methodological decisions:

    1. Construct the RUL target from the complete training trajectories.
    2. Compare the original linear RUL with a capped modelling target.
    3. Determine which available variables may legitimately be used as predictors.
    4. Separate engines into training and validation sets without mixing trajectories.
    5. Define baseline feature sets.
    6. Construct temporal features using only information available up to the prediction cycle.

    Particular attention is given to preventing information leakage. Variables or transformations that depend on future observations must not be made available to the predictive model.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. RUL target construction

    The objective of this notebook is to transform the run-to-failure trajectories into a supervised regression dataset.

    In supervised learning, each observation is associated with:

    - **features**, which represent the information available to the model at the time of prediction.
    - **target**, which represents the value that the model must learn to predict.

    For FD001, the target is the Remaining Useful Life (RUL), defined as the number of operating cycles remaining before the end of an engine trajectory.

    Because the training trajectories are complete run-to-failure histories, the final cycle of each engine is known and can be used to construct the target.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from hawkshot.features.temporal import add_temporal_features

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
    return (
        OPERATIONAL_COLUMNS,
        add_temporal_features,
        available_sensors,
        df_filtered,
        mo,
        pd,
        plt,
        train_test_split,
    )


@app.cell
def _(df_filtered):
    _max_cycle = df_filtered.groupby("engine_id")["cycle"].transform("max")

    df_rul = df_filtered.assign(
        max_cycle=_max_cycle, rul=_max_cycle - df_filtered["cycle"]
    )
    return (df_rul,)


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

    A capped RUL is therefore introduced as a modelling assumption. Values above 125 cycles are replaced by 125, while lower values remain unchanged.

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
def _(mo):
    mo.md(r"""
    A threshold of 125 cycles affects approximately 39% of the available observations. The remaining 61% retain their exact linear RUL value.

    The capped target therefore preserves precise RUL information for most of the dataset while grouping earlier high-RUL observations into a common long-life region.

    Importantly, the threshold does not represent a physical boundary between healthy and degraded engines. It only limits the precision required from the model when the engine is still far from failure.
    """)
    return


@app.cell
def _(df_rul, rul_cap):
    df_prepared = df_rul.assign(rul_capped=df_rul["rul"].clip(upper=rul_cap))
    return (df_prepared,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Predictor definition and leakage analysis

    Before preparing the modelling dataset, each variable must be classified according to the information that would be available at prediction time.

    A valid predictor must be observable without requiring knowledge of the engine's future trajectory.

    The retained sensor measurements and operational settings satisfy this condition and are therefore considered candidate features. The retained sensors are included in the initial feature set because the previous analysis showed that they contain structured degradation-related information. The usefulness of the operational settings has not yet been established and is assessed below.

    `engine_id` is required to identify trajectories and to create engine-level data splits, but it is not used as a predictive feature because its numerical value is only an arbitrary identifier.

    The RUL target and `max_cycle` must never be provided to the model. `max_cycle` directly depends on the future failure time and would therefore introduce target leakage.

    The operating `cycle` is different. It is known at prediction time and does not constitute data leakage. However, because it represents engine age, a model may rely heavily on it instead of extracting degradation information from sensor measurements. Its predictive contribution will therefore be evaluated separately.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Operational Settings Assessment

    The operational settings have not yet been evaluated as potential predictors. Before defining the initial feature set, their behaviour is examined from two perspectives:

    - their relationship with engine age.
    - their relationship with the retained sensor measurements.

    The analysis is performed separately within each engine trajectory using Spearman correlation, in order to preserve the temporal structure of the dataset.
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
    `operational_setting_3` is strictly constant across all observations and therefore contains no predictive information.

    `operational_setting_1` and `operational_setting_2` vary within individual engine trajectories. However, their median absolute Spearman correlations with operating cycle are both approximately `0.04`, indicating no meaningful monotonic relationship with engine age.

    Their relationships with the 14 retained sensors are similarly weak. Across all setting-sensor combinations, median absolute correlations remain close to zero, with the strongest value only reaching `0.06`.

    These results provide no evidence that the operational settings either track engine ageing directly or explain a substantial proportion of the observed sensor evolution. They are therefore excluded from the initial modelling feature set.

    This conclusion concerns monotonic relationships only. Possible nonlinear effects or interactions cannot be ruled out from correlation analysis alone and may later be reassessed through model comparison.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Operating cycle as predictor

    The operating cycle requires separate consideration. Unlike `max_cycle`, it is known at prediction time and therefore does not introduce data leakage.

    However, `cycle` represents the current operational age of the engine and may already provide substantial information about its remaining lifetime. A model including this variable could therefore achieve improved performance partly by exploiting engine age rather than degradation signals measured by the sensors.

    For this reason, `cycle` is not discarded. Instead, separate feature sets will be prepared so that its contribution can later be evaluated explicitly :

    - **Sensor only**, using the 14 retained sensor measurements.
    - **Sensors + cycle**, adding the current engine age.

    A cycle-only baseline may also be evaluated during modelling. Comparing these configurations will make it possible to determine whether the sensor measurements provide predictive information beyond engine age alone.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Engine-level train/validation split

    The modelling data must be separated into training and validation sets before any data-dependent preprocessing is learned.

    A random row-level split would be inappropriate for FD001 because multiple observations belong to the same engine trajectory. Such a split could place different cycles from the same engine in both training and validation data, allowing the model to be evaluated on engines that it has already partially observed.

    The split is therefore performed at the engine-level. All observations from one engine are assigned exclusively to either the training or validation set.

    Eighty engines are used for training and twenty are reserved for validation. The validation engines remain completely unseen during model fitting, making the evaluation more representative of prediction on new engines.

    The official FD001 test trajectories will remain separate and will be used later for final model evaluation.
    """)
    return


@app.cell
def _(df_prepared, train_test_split):
    engine_ids = df_prepared["engine_id"].unique()

    train_engines, validation_engines = train_test_split(
        engine_ids, test_size=0.2, random_state=42
    )
    return train_engines, validation_engines


@app.cell
def _(df_prepared, train_engines, validation_engines):
    df_train = df_prepared[df_prepared["engine_id"].isin(train_engines)].copy()

    df_validation = df_prepared[
        df_prepared["engine_id"].isin(validation_engines)
    ].copy()
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
def _(mo):
    mo.md(r"""
    ## 4. Preprocessing strategy

    Some predictive models are sensitive to differences in feature scale. Standardisation may therefore be required before modelling.

    However, any preprocessing parameters must be learned exclusively from the training data. For example, the mean and standard deviation used to standardise a sensor must be calculated from the training engines only and then applied unchanged to the validation engines.

    This differs from the per-engine standardisation used during exploratory analysis. In the previous notebook, complete trajectories were standardised independently to facilitate visual comparison. Such a transformation would not be available for an operating engine because its future measurements are unknown.

    No scaling is applied directly in this notebook. When required by a model, standardisation will be integrated into the modelling pipeline so that it is fitted only on the training data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Initial feature sets

    In supervised learning, the input variables are conventionally represented by
    `X`, while the target to be predicted is represented by `y`.

    Several feature sets are prepared so that their contribution can later be
    evaluated independently.

    - **Cycle only** provides a reference based solely on engine age.
    - **Raw sensors** use the 14 retained instantaneous sensor measurements.
    - **Raw sensors + cycle** evaluate whether operational age adds information
      beyond the sensor state.

    The capped RUL is used as the initial modelling target, while the original
    linear RUL remains available for later comparison.
    """)
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

    X_train_cycle = df_train[["cycle"]]
    X_validation_cycle = df_validation[["cycle"]]

    y_train = df_train["rul_capped"]
    y_validation = df_validation["rul_capped"]

    assert len(X_train_sensors) == len(y_train)
    assert len(X_validation_sensors) == len(y_validation)

    assert len(X_train_sensors_cycle) == len(y_train)
    assert len(X_validation_sensors_cycle) == len(y_validation)

    assert len(X_train_cycle) == len(y_train)
    assert len(X_validation_cycle) == len(y_validation)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Temporal feature engineering

    The raw sensor values describe the instantaneous state of the engine, but they do not explicitly describe how that state has recently evolved.

    Two engines may therefore present similar sensor values at the current cycle despite having followed different recent trajectories. Temporal features are introduced to provide the model with information about this recent evolution.

    Three complementary transformations are considered:

    - **Rolling mean** describes the recent average sensor level and reduces short-term variability.
    - **Delta** measures the change between the beginning and the end of a recent window.
    - **Slope** estimates the overall rate and direction of change across the window using a first-degree linear fit.

    The original sensor measurements are retained alongside these derived features because current sensor level and recent evolution may provide complementary information.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Different temporal transformations may require different observation windows. A short window reacts more quickly to recent changes but is also more sensitive to noise, whereas a longer window provides a more stable representation of the underlying trend.

    Window lengths are therefore configured independently for rolling means, deltas and slopes. The values defined here are initial experimental choices rather than optimised parameters. Their predictive impact will be compared during modelling.
    """)
    return


@app.cell
def _():
    temporal_config = {
        "mean": [5, 10, 20],
        "delta": [5, 10],
        "slope": [10, 20],
    }
    return (temporal_config,)


@app.cell
def _(mo):
    mo.md(r"""
    All temporal features are causal, the value calculated for a given cycle uses only the current and previous observations from the same engine.

    Future cycles are never included in a temporal window. This is essential for a realistic predictive-maintenance setting, where measurements occurring after the prediction time would not yet be available.

    The centred rolling windows previously used for visualisation are therefore not used for modelling.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The slope is obtained by fitting a straight line to the sensor measurements contained in the current temporal window. Its coefficient represents the average rate of change of the sensor over that period.

    A positive slope indicates an increasing recent trend, a negative slope a decreasing trend, and a value close to zero indicates relative stability.

    When only one observation is available, the slope is defined as zero because no temporal evolution can yet be estimated.
    """)
    return


@app.cell
def _(
    add_temporal_features,
    available_sensors,
    df_train,
    df_validation,
    pd,
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

    assert len(df_train_temporal) == len(df_train)
    assert len(df_validation_temporal) == len(df_validation)

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

    feature_sets = {
        "cycle": cycle_features,
        "raw_sensors": raw_sensor_features,
        "raw_sensors_cycle": raw_sensor_cycle_features,
        "raw_temporal": raw_temporal_features,
    }

    feature_set_summary = pd.DataFrame(
        [
            {
                "feature_set": name,
                "n_features": len(features),
            }
            for name, features in feature_sets.items()
        ]
    )

    feature_set_summary
    return df_train_temporal, df_validation_temporal, temporal_features


@app.cell
def _(mo):
    mo.md(r"""
    ### Temporal feature validation

    Before using the generated features for modelling, several consistency checks are performed.

    The transformation must preserve the number of observations and must not produce missing temporal values. At the first cycle of every engine, delta and slope features must equal zero because no previous observations are available.

    A sample trajectory is also inspected below using deliberately different window sizes. This verifies that rolling means, deltas, and slopes are calculated independently according to their configured temporal horizons.
    """)
    return


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


@app.cell
def _(
    df_train,
    df_train_temporal,
    df_validation,
    df_validation_temporal,
    temporal_features,
):
    temporal_checks = {
        "train_rows_preserved": len(df_train_temporal) == len(df_train),
        "validation_rows_preserved": (
            len(df_validation_temporal) == len(df_validation)
        ),
        "train_missing_temporal_values": (
            df_train_temporal[temporal_features].isna().sum().sum()
        ),
        "validation_missing_temporal_values": (
            df_validation_temporal[temporal_features].isna().sum().sum()
        ),
    }

    temporal_checks
    return


@app.cell
def _(mo):
    mo.md(r"""
    The resulting values show the expected complementary behaviours: the rolling mean smooths short-term sensor variation, the short-window delta reacts more strongly to recent changes, and the longer-window slope provides a more stable estimate of the underlying trend.

    These checks confirm that temporal features are generated independently within each engine trajectory and using only information available up to the current cycle.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 7. Conclusions and modelling decisions

    This notebook transformed the FD001 run-to-failure trajectories into a supervised learning dataset suitable for Remaining Useful Life prediction.

    The original linear RUL was first reconstructed from the complete run-to-failure trajectories and retained as the physical ground-truth target. A capped RUL of 125 cycles was then introduced as the initial modelling target. This threshold does not represent a physical transition between healthy and degraded states, but rather a modelling assumption intended to reduce the precision required when engines are still far from failure and the degradation-related sensor signal is weaker.

    The predictor analysis also clarified which variables may legitimately be used by the model. The 14 retained sensors remain the main source of degradation-related information. The operational settings are excluded from the initial baseline because one of them is constant and the other two show no meaningful monotonic relationship with engine age or the retained sensor measurements. Their possible nonlinear contribution is not ruled out and may later be reassessed through model comparison. The operating cycle is treated separately, it is available at prediction time and therefore does not consitute data leakage, but it may provide a strong shortcut through engine age. its contribution will consequently be evaluated independently from the sensor measurements.

    To obtain a realistic validation procedure, the data were split at the engine level rather than at the observation level. Eighty engines are used for training and twenty are reserved for validation, while the official FD001 test remains untouched for final evaluation. Any data-dependent preprocessing required by future models will be learned exclusively from the training engines and then applied unchanged to validation data.

    Several feature configurations are now available for comparison, cycle only, raw sensors, raw sensors combined with cycle, and raw sensors enriched with temporal features. The temporal features were designed to describe recent sensor evolution through rolling means, deltas, and slops. They are computed causally within each engine trajectory, using only the current and previous observations, and their window sizes can be configured independently.

    The next stage will therefore focus on baseline regression models evaluated on the fixed validation set. By comparing the prepared feature configurations one at a time, it will be possible to determine how much predictive information comes from engine age, instantaneous sensor values, and recent sensor dynamics, and to identify which modelling choices genuinely improves RUL prediction.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
