import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        RandomForestRegressor,
        GradientBoostingRegressor,
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
        GradientBoostingRegressor,
        GridSearchCV,
        GroupKFold,
        Pipeline,
        RandomForestRegressor,
        SVR,
        StandardScaler,
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
def _(add_temporal_features, available_sensors, df_train, temporal_config):
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

    for _fold_number, (_train_index, __validation_index) in enumerate(folds, start=1):
        train_engines_fold = groups_fold.iloc[_train_index].unique()
        validation_engines_fold = groups_fold.iloc[__validation_index].unique()
        intersection = np.intersect1d(train_engines_fold, validation_engines_fold)

        length_train_fold.append(len(train_engines_fold))
        length_validation_fold.append(len(validation_engines_fold))
        list_intersection.append(intersection.size)
        list_fold.append(_fold_number)

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
    X_train_fold,
    X_validation_fold,
    folds,
    mean_absolute_error,
    np,
    root_mean_squared_error,
    y_fold,
    y_train_fold,
    y_validation_fold,
):
    model_et = ExtraTreesRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )

    maes_et_fold = []
    rmses_et_fold = []

    for _fold_number, (_train_index, _validation_index) in enumerate(folds, start=1):
        X_train_fold_et = X_fold.iloc[_train_index]
        X_validation_fold_et = X_fold.iloc[_validation_index]

        y_train_fold_et = y_fold.iloc[_train_index]
        y_validation_fold_et = y_fold.iloc[_validation_index]

        model_et.fit(X_train_fold_et, y_train_fold_et)

        y_pred_et = model_et.predict(X_validation_fold_et)

        mae_validation_et = mean_absolute_error(y_validation_fold_et, y_pred_et)
        rmse_validation_et = root_mean_squared_error(y_validation_fold_et, y_pred_et)

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
def _(ExtraTreesRegressor, GridSearchCV, X_fold, gkf, groups_fold, pd, y_fold):
    param_grid = {
        "max_depth": [14, 16, 18],
        "min_samples_leaf": [2, 4, 6],
    }

    model_et_GS = ExtraTreesRegressor(
        n_estimators=100, random_state=42, min_samples_leaf=1.0
    )

    grid_search = GridSearchCV(
        estimator=model_et_GS,
        param_grid=param_grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
    )

    grid_search.fit(X_fold, y_fold, groups=groups_fold)
    results = pd.DataFrame(grid_search.cv_results_)
    top_4 = results.nlargest(4, "mean_test_score")

    print(top_4[["params", "mean_test_score", "std_test_score", "rank_test_score"]])
    return (results,)


@app.cell
def _(results):
    results["mean_MAE"] = -results["mean_test_score"]

    top_4_display = results.sort_values("rank_test_score").head(4).round(3)

    top_4_display[
        [
            "param_max_depth",
            "param_min_samples_leaf",
            "mean_MAE",
            "std_test_score",
            "rank_test_score",
        ]
    ]

    return


@app.cell
def _(
    RandomForestRegressor,
    X_fold,
    folds,
    mean_absolute_error,
    np,
    root_mean_squared_error,
    y_fold,
):
    model_rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )

    maes_rf_fold = []
    rmses_rf_fold = []

    for _fold_number, (_train_index, _validation_index) in enumerate(folds, start=1):
        X_train_fold_rf = X_fold.iloc[_train_index]
        X_validation_fold_rf = X_fold.iloc[_validation_index]

        y_train_fold_rf = y_fold.iloc[_train_index]
        y_validation_fold_rf = y_fold.iloc[_validation_index]

        model_rf.fit(X_train_fold_rf, y_train_fold_rf)

        y_pred_rf = model_rf.predict(X_validation_fold_rf)

        mae_validation_rf = mean_absolute_error(y_validation_fold_rf, y_pred_rf)
        rmse_validation_rf = root_mean_squared_error(y_validation_fold_rf, y_pred_rf)

        maes_rf_fold.append(mae_validation_rf)
        rmses_rf_fold.append(rmse_validation_rf)

    f"mean MAE: {np.mean(maes_rf_fold)}, std MAE: {np.std(maes_rf_fold)}, mean RMSE: {np.mean(rmses_rf_fold)}, std RMSE: {np.std(rmses_rf_fold)}"
    return maes_rf_fold, rmses_rf_fold


@app.cell
def _(
    GradientBoostingRegressor,
    X_fold,
    folds,
    mean_absolute_error,
    np,
    root_mean_squared_error,
    y_fold,
):
    model_gb = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        random_state=42,
    )

    maes_gb_fold = []
    rmses_gb_fold = []

    for _fold_number, (_train_index, _validation_index) in enumerate(folds, start=1):
        X_train_fold_gb = X_fold.iloc[_train_index]
        X_validation_fold_gb = X_fold.iloc[_validation_index]

        y_train_fold_gb = y_fold.iloc[_train_index]
        y_validation_fold_gb = y_fold.iloc[_validation_index]

        model_gb.fit(X_train_fold_gb, y_train_fold_gb)

        y_pred_gb = model_gb.predict(X_validation_fold_gb)

        mae_validation_gb = mean_absolute_error(y_validation_fold_gb, y_pred_gb)
        rmse_validation_gb = root_mean_squared_error(y_validation_fold_gb, y_pred_gb)

        maes_gb_fold.append(mae_validation_gb)
        rmses_gb_fold.append(rmse_validation_gb)

    f"mean MAE: {np.mean(maes_gb_fold)}, std MAE: {np.std(maes_gb_fold)}, mean RMSE: {np.mean(rmses_gb_fold)}, std RMSE: {np.std(rmses_gb_fold)}"
    return maes_gb_fold, rmses_gb_fold


@app.cell
def _(
    Pipeline,
    SVR,
    StandardScaler,
    X_fold,
    folds,
    mean_absolute_error,
    np,
    root_mean_squared_error,
    y_fold,
):
    model_SVR = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale")),
        ]
    )

    maes_SVR_fold = []
    rmses_SVR_fold = []

    for _fold_number, (_train_index, _validation_index) in enumerate(folds, start=1):
        X_train_fold_SVR = X_fold.iloc[_train_index]
        X_validation_fold_SVR = X_fold.iloc[_validation_index]

        y_train_fold_SVR = y_fold.iloc[_train_index]
        y_validation_fold_SVR = y_fold.iloc[_validation_index]

        model_SVR.fit(X_train_fold_SVR, y_train_fold_SVR)

        y_pred_SVR = model_SVR.predict(X_validation_fold_SVR)

        mae_validation_SVR = mean_absolute_error(y_validation_fold_SVR, y_pred_SVR)
        rmse_validation_SVR = root_mean_squared_error(y_validation_fold_SVR, y_pred_SVR)

        maes_SVR_fold.append(mae_validation_SVR)
        rmses_SVR_fold.append(rmse_validation_SVR)

    f"mean MAE: {np.mean(maes_SVR_fold)}, std MAE: {np.std(maes_SVR_fold)}, mean RMSE: {np.mean(rmses_SVR_fold)}, std RMSE: {np.std(rmses_SVR_fold)}"
    return maes_SVR_fold, rmses_SVR_fold


@app.cell
def _(
    maes_SVR_fold,
    maes_et_fold,
    maes_gb_fold,
    maes_rf_fold,
    np,
    pd,
    rmses_SVR_fold,
    rmses_et_fold,
    rmses_gb_fold,
    rmses_rf_fold,
):
    CV_results_comparison = [
        {
            "model": "Random Forest",
            "mean_MAE_CV": np.mean(maes_rf_fold),
            "std_mae": np.std(maes_rf_fold),
            "mean_RMSE_CV": np.mean(rmses_rf_fold),
            "std_RMSE": np.std(rmses_rf_fold),
        },
        {
            "model": "Extra Trees",
            "mean_MAE_CV": np.mean(maes_et_fold),
            "std_mae": np.std(maes_et_fold),
            "mean_RMSE_CV": np.mean(rmses_et_fold),
            "std_RMSE": np.std(rmses_et_fold),
        },
        {
            "model": "Gradient Boosting",
            "mean_MAE_CV": np.mean(maes_gb_fold),
            "std_mae": np.std(maes_gb_fold),
            "mean_RMSE_CV": np.mean(rmses_gb_fold),
            "std_RMSE": np.std(rmses_gb_fold),
        },
        {
            "model": "SVR",
            "mean_MAE_CV": np.mean(maes_SVR_fold),
            "std_mae": np.std(maes_SVR_fold),
            "mean_RMSE_CV": np.mean(rmses_SVR_fold),
            "std_RMSE": np.std(rmses_SVR_fold),
        },
    ]

    pd.DataFrame(CV_results_comparison).round(3)
    return


if __name__ == "__main__":
    app.run()
