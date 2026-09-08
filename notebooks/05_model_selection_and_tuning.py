import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    from sklearn.ensemble import (
        ExtraTreesRegressor,
    )

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
        ExtraTreesRegressor,
        GridSearchCV,
        GroupKFold,
        add_temporal_features,
        available_sensors,
        df_filtered,
        mean_absolute_error,
        mo,
        np,
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
    return engine_ids, train_engines, validation_engines


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
def _(engine_ids):
    engine_ids
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

    temporal_features = [
        column
        for column in df_train_temporal.columns
        if any(suffix in column for suffix in ["_mean_", "_delta_", "_slope_"])
    ]

    raw_temporal_cycle_features = [
        "cycle",
        *available_sensors,
        *temporal_features,
    ]

    return df_train_temporal, raw_temporal_cycle_features


@app.cell
def _(GroupKFold, df_train_temporal, np, pd, raw_temporal_cycle_features):
    gkf = GroupKFold(n_splits=5)

    groups_fold = df_train_temporal["engine_id"]
    X_fold = df_train_temporal[raw_temporal_cycle_features]
    y_fold = df_train_temporal["rul_capped"]

    folds = list(gkf.split(X_fold, y_fold, groups_fold))

    length_train_fold = []
    length_validation_fold = []
    list_intersection = []
    list_fold = []

    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        train_engines_fold = groups_fold.iloc[train_index].unique()
        validation_engines_fold = groups_fold.iloc[validation_index].unique()
        intersection = np.intersect1d(train_engines_fold, validation_engines_fold)

        length_train_fold.append(len(train_engines_fold))
        length_validation_fold.append(len(validation_engines_fold))
        list_intersection.append(intersection.size)
        list_fold.append(fold_number)

    pd.DataFrame(
        {
            "folds": list_fold,
            "train_engines": length_train_fold,
            "validation_engines": length_validation_fold,
            "common_engines": list_intersection,
        }
    )

    return X_fold, folds, gkf, groups_fold, list_fold, y_fold


@app.cell
def _(
    ExtraTreesRegressor,
    X_fold,
    folds,
    mean_absolute_error,
    np,
    root_mean_squared_error,
    y_fold,
):
    model_et = ExtraTreesRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )

    maes_et_fold = []
    rmses_et_fold = []

    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        X_train_fold = X_fold.iloc[train_index]
        X_validation_fold = X_fold.iloc[validation_index]

        y_train_fold = y_fold.iloc[train_index]
        y_validation_fold = y_fold.iloc[validation_index]

        model_et.fit(X_train_fold, y_train_fold)

        y_pred_et = model_et.predict(X_validation_fold)

        mae_validation_et = mean_absolute_error(y_validation_fold, y_pred_et)
        rmse_validation_et = root_mean_squared_error(y_validation_fold, y_pred_et)

        maes_et_fold.append(mae_validation_et)
        rmses_et_fold.append(rmse_validation_et)

    f"mean MAE: {np.mean(maes_et_fold)}, std MAE: {np.std(maes_et_fold)}, mean RMSE: {np.mean(rmses_et_fold)}, std RMSE: {np.std(rmses_et_fold)}"

    return maes_et_fold, rmses_et_fold


@app.cell
def _(list_fold, maes_et_fold, pd, rmses_et_fold):
    pd.DataFrame(
        {
            "fold": list_fold,
            "MAE": maes_et_fold,
            "RMSE": rmses_et_fold,
        }
    ).round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The Extra Tree baseline shows relatively stable performance across the five engine-level folds. Four folds produce very similar errors, while the third fold is moderately more difficult, leading to a mean validation MAE of approximately 10.62 cycles with a standard deviation of 0.27 cycles.
    """)
    return


@app.cell
def _(ExtraTreesRegressor, GridSearchCV, X_fold, gkf, groups_fold, y_fold):
    param_grid = {
        "max_depth": [8, 10, 12, 14],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.5, 0.75, 1.0],
    }

    model_et_GS = ExtraTreesRegressor(n_estimators=100, random_state=42)

    grid_search = GridSearchCV(
        estimator=model_et_GS,
        param_grid=param_grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
    )

    grid_search.fit(X_fold, y_fold, groups=groups_fold)
    result = grid_search.best_params_
    result
    return


if __name__ == "__main__":
    app.run()
