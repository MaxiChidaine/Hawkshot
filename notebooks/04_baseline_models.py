import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    from sklearn.linear_model import LinearRegression

    from hawkshot.data.cmapss import (
        filter_constant_sensors,
        load_fd001,
    )

    df_raw = load_fd001("data/raw/cmapss")
    df_filtered, removed_sensors = filter_constant_sensors(df_raw)

    return (
        LinearRegression,
        df_filtered,
        mean_absolute_error,
        pd,
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
def _(
    baseline_mae_mae,
    df_train,
    df_validation,
    mean_absolute_error,
    root_mean_squared_error,
):
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

    baseline_mae = mean_absolute_error(y_validation, y_pred_baseline_median)
    baseline_rsme = root_mean_squared_error(y_validation, y_pred_baseline_mean)
    f"MAE : {baseline_mae}, RSME: {baseline_rsme}"
    return y_train, y_validation


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
    rsme_cycle_only = root_mean_squared_error(y_validation, y_pred_cycle_only)
    f"MAE : {mae_cycle_only}, RSME: {rsme_cycle_only}"
    return model_cycle_only, y_pred_cycle_only


@app.cell
def _(model_cycle_only):
    model_cycle_only.coef_
    return


@app.cell
def _(model_cycle_only):
    model_cycle_only.intercept_
    return


@app.cell
def _(y_pred_cycle_only):
    y_pred_cycle_only.min()
    return


@app.cell
def _(y_pred_cycle_only):
    y_pred_cycle_only.max()
    return


if __name__ == "__main__":
    app.run()
