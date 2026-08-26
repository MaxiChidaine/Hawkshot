import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    from hawkshot.data.cmapss import (
        SENSOR_COLUMNS,
        filter_constant_sensors,
        load_fd001,
    )

    from hawkshot.features.temporal import add_temporal_features

    df_raw = load_fd001("data/raw/cmapss")
    df_filtered, removed_sensors = filter_constant_sensors(df_raw)

    available_sensors = [
        sensor for sensor in SENSOR_COLUMNS if sensor in df_filtered.columns
    ]

    return (
        LinearRegression,
        Pipeline,
        StandardScaler,
        add_temporal_features,
        available_sensors,
        df_filtered,
        mean_absolute_error,
        pd,
        plt,
        root_mean_squared_error,
        train_test_split,
    )


@app.cell
def _(df_filtered):
    _max_cycle = df_filtered.groupby("engine_id")["cycle"].transform("max")

    df_rul = df_filtered.assign(
        max_cycle=_max_cycle, rul=_max_cycle - df_filtered["cycle"]
    )

    rul_cap = 125

    df_prepared = df_rul.assign(rul_capped=df_rul["rul"].clip(upper=rul_cap))
    return (df_prepared,)


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
def _(df_train, df_validation, mean_absolute_error, root_mean_squared_error):
    y_train = df_train["rul_capped"]
    y_validation = df_validation["rul_capped"]

    baseline_median = y_train.median()
    baseline_mean = y_train.mean()

    y_pred_baseline_median = []
    y_pred_baseline_mean = []

    for ruls in y_validation:
        y_pred_baseline_median.append(baseline_median)

    for ruls in y_validation:
        y_pred_baseline_mean.append(baseline_mean)

    mae_baseline_median = mean_absolute_error(y_validation, y_pred_baseline_median)
    rmse_baseline_mean = root_mean_squared_error(y_validation, y_pred_baseline_mean)
    f"MAE : {mae_baseline_median}, rmse: {rmse_baseline_mean}"
    return (
        mae_baseline_median,
        rmse_baseline_mean,
        y_pred_baseline_mean,
        y_pred_baseline_median,
        y_train,
        y_validation,
    )


@app.cell
def _(
    mean_absolute_error,
    root_mean_squared_error,
    y_pred_baseline_mean,
    y_pred_baseline_median,
    y_validation,
):
    mae_baseline_mean = mean_absolute_error(y_validation, y_pred_baseline_mean)
    rmse_baseline_median = root_mean_squared_error(y_validation, y_pred_baseline_median)
    f"MAE : {mae_baseline_mean}, rmse: {rmse_baseline_median}"
    return mae_baseline_mean, rmse_baseline_median


@app.cell
def _(
    LinearRegression,
    df_train,
    df_validation,
    mean_absolute_error,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    X_train_cycle = df_train[["cycle"]]
    X_validation_cycle = df_validation[["cycle"]]

    model_cycle_only = LinearRegression()
    model_cycle_only.fit(X_train_cycle, y_train)

    y_pred_cycle_only = model_cycle_only.predict(X_validation_cycle)

    mae_cycle_only = mean_absolute_error(y_validation, y_pred_cycle_only)
    rmse_cycle_only = root_mean_squared_error(y_validation, y_pred_cycle_only)
    f"MAE : {mae_cycle_only}, rmse: {rmse_cycle_only}"
    return (
        X_validation_cycle,
        mae_cycle_only,
        rmse_cycle_only,
        y_pred_cycle_only,
    )


@app.cell
def _(X_validation_cycle, plt, y_pred_cycle_only, y_validation):
    _x = X_validation_cycle["cycle"]
    _y_real = y_validation
    _y_predicted = y_pred_cycle_only

    plt.figure(figsize=(10, 6))
    plt.scatter(
        _x,
        _y_real,
        label="True RUL with cycles",
        color="blue",
        marker="o",
        linewidths=0.5,
        s=1,
    )
    plt.scatter(
        _x,
        _y_predicted,
        label="Predicted RUL with cycles",
        color="red",
        marker="s",
        linewidths=0.5,
        s=1,
    )

    plt.plot()

    plt.xlabel("Cycles")
    plt.ylabel("True RUL and Predicted RUL")
    plt.title("Cycle only prediction model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return


@app.cell
def _(
    LinearRegression,
    Pipeline,
    StandardScaler,
    available_sensors,
    df_train,
    df_validation,
    mean_absolute_error,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    sensor_features = available_sensors

    X_train_sensors = df_train[sensor_features]
    X_validation_sensors = df_validation[sensor_features]

    model_raw_sensors = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )

    model_raw_sensors.fit(X_train_sensors, y_train)
    y_pred_raw_sensors = model_raw_sensors.predict(X_validation_sensors)

    mae_raw_sensors = mean_absolute_error(y_validation, y_pred_raw_sensors)
    rmse_raw_sensors = root_mean_squared_error(y_validation, y_pred_raw_sensors)
    f"MAE : {mae_raw_sensors}, rmse: {rmse_raw_sensors}"
    return mae_raw_sensors, rmse_raw_sensors


@app.cell
def _(
    LinearRegression,
    Pipeline,
    StandardScaler,
    available_sensors,
    df_train,
    df_validation,
    mean_absolute_error,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    sensor_cycle_features = [
        "cycle",
        *available_sensors,
    ]

    X_train_sensors_cycle = df_train[sensor_cycle_features]
    X_validation_sensors_cycle = df_validation[sensor_cycle_features]

    model_sensors_cycle = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )

    model_sensors_cycle.fit(X_train_sensors_cycle, y_train)
    y_pred_sensors_cycle = model_sensors_cycle.predict(X_validation_sensors_cycle)

    mae_sensors_cycle = mean_absolute_error(y_validation, y_pred_sensors_cycle)
    rmse_sensors_cycle = root_mean_squared_error(y_validation, y_pred_sensors_cycle)
    f"MAE : {mae_sensors_cycle}, rmse: {rmse_sensors_cycle}"
    return mae_sensors_cycle, rmse_sensors_cycle


@app.cell
def _(
    mae_baseline_mean,
    mae_baseline_median,
    mae_cycle_only,
    mae_raw_sensors,
    mae_sensors_cycle,
    pd,
    rmse_baseline_mean,
    rmse_baseline_median,
    rmse_cycle_only,
    rmse_raw_sensors,
    rmse_sensors_cycle,
):
    models = [
        "constant_median",
        "constant_mean",
        "cycle_only",
        "raw_sensors",
        "sensors_cycle",
    ]
    maes = [
        mae_baseline_median,
        mae_baseline_mean,
        mae_cycle_only,
        mae_raw_sensors,
        mae_sensors_cycle,
    ]
    rmses = [
        rmse_baseline_median,
        rmse_baseline_mean,
        rmse_cycle_only,
        rmse_raw_sensors,
        rmse_sensors_cycle,
    ]

    pd.DataFrame({"model": models, "MAE": maes, "RMSE": rmses}).round(2)
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
    return df_train_temporal, df_validation_temporal


@app.cell
def _(
    LinearRegression,
    Pipeline,
    StandardScaler,
    available_sensors,
    df_train_temporal,
    df_validation_temporal,
    mean_absolute_error,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    temporal_features = [
        column
        for column in df_train_temporal.columns
        if any(suffix in column for suffix in ["_mean_", "_delta_", "_slope_"])
    ]

    raw_temporal_features = [
        *available_sensors,
        *temporal_features,
    ]

    X_train_raw_temporal = df_train_temporal[raw_temporal_features]
    X_validation_temporal = df_validation_temporal[raw_temporal_features]

    model_raw_temporal = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )

    model_raw_temporal.fit(X_train_raw_temporal, y_train)
    y_pred_raw_temporal = model_raw_temporal.predict(X_validation_temporal)

    mae_raw_temporal = mean_absolute_error(y_validation, y_pred_raw_temporal)
    rmse_raw_temporal = root_mean_squared_error(y_validation, y_pred_raw_temporal)
    f"MAE : {mae_raw_temporal}, RMSE: {rmse_raw_temporal}"
    return (temporal_features,)


@app.cell
def _(
    LinearRegression,
    Pipeline,
    StandardScaler,
    available_sensors,
    df_train_temporal,
    df_validation_temporal,
    mean_absolute_error,
    root_mean_squared_error,
    temporal_features,
    y_train,
    y_validation,
):
    raw_temporal_cycle_features = [
        "cycle",
        *available_sensors,
        *temporal_features,
    ]

    X_train_raw_temporal_cycle = df_train_temporal[raw_temporal_cycle_features]
    X_validation_temporal_cycle = df_validation_temporal[raw_temporal_cycle_features]

    model_raw_temporal_cycle = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )

    model_raw_temporal_cycle.fit(X_train_raw_temporal_cycle, y_train)
    y_pred_raw_temporal_cycle = model_raw_temporal_cycle.predict(
        X_validation_temporal_cycle
    )

    mae_raw_temporal_cycle = mean_absolute_error(
        y_validation, y_pred_raw_temporal_cycle
    )
    rmse_raw_temporal_cycle = root_mean_squared_error(
        y_validation, y_pred_raw_temporal_cycle
    )
    f"MAE : {mae_raw_temporal_cycle}, RMSE: {rmse_raw_temporal_cycle}"
    return


if __name__ == "__main__":
    app.run()
