import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import winsound

    from sklearn.base import clone
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

    from time import perf_counter

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
        clone,
        df_filtered,
        mean_absolute_error,
        mo,
        np,
        pd,
        perf_counter,
        root_mean_squared_error,
        train_test_split,
        winsound,
    )


@app.cell
def _(winsound):
    def notify_done():
        winsound.MessageBeep()

    return (notify_done,)


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

    df_validation_temporal = add_temporal_features(
        df_validation,
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
    return (
        df_train_temporal,
        df_validation_temporal,
        raw_temporal_cycle_features,
    )


@app.cell
def _(clone, mean_absolute_error, np, perf_counter, root_mean_squared_error):
    def evaluate_grouped_cv(model, X, y, groups, cv):
        maes = []
        rmses = []
        fit_times = []
        predict_times = []

        for train_index, validation_index in cv.split(X, y, groups=groups):
            X_train_fold = X.iloc[train_index]
            X_validation_fold = X.iloc[validation_index]

            y_train_fold = y.iloc[train_index]
            y_validation_fold = y.iloc[validation_index]

            fold_model = clone(model)

            fit_start = perf_counter()
            fold_model.fit(X_train_fold, y_train_fold)
            fit_times.append(perf_counter() - fit_start)

            predict_start = perf_counter()
            y_pred = fold_model.predict(X_validation_fold)
            predict_times.append(perf_counter() - predict_start)

            maes.append(mean_absolute_error(y_validation_fold, y_pred))
            rmses.append(root_mean_squared_error(y_validation_fold, y_pred))

        return {
            "mean_MAE": np.mean(maes),
            "std_MAE": np.std(maes),
            "mean_RMSE": np.mean(rmses),
            "std_RMSE": np.std(rmses),
            "mean_fit_time": np.mean(fit_times),
            "mean_predict_time": np.mean(predict_times),
        }

    return (evaluate_grouped_cv,)


@app.cell
def _(GridSearchCV):
    def run_grouped_grid_search(model, param_grid, X, y, groups, cv):
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring={
                "MAE": "neg_mean_absolute_error",
                "RMSE": "neg_root_mean_squared_error",
            },
            refit="MAE",
            cv=cv,
            n_jobs=-1,
        )

        grid_search.fit(X, y, groups=groups)

        return grid_search

    return (run_grouped_grid_search,)


@app.cell
def _(GroupKFold, df_train_temporal, np, pd, raw_temporal_cycle_features):
    cv = GroupKFold(n_splits=5)

    groups_fold = df_train_temporal["engine_id"]
    X_fold = df_train_temporal[raw_temporal_cycle_features]
    y_fold = df_train_temporal["rul_capped"]

    folds = list(cv.split(X_fold, y_fold, groups_fold))

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
    return X_fold, cv, groups_fold, y_fold


@app.cell
def _(
    ExtraTreesRegressor,
    X_fold,
    cv,
    evaluate_grouped_cv,
    groups_fold,
    y_fold,
):
    model_et = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    results_et = evaluate_grouped_cv(
        model=model_et,
        X=X_fold,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    results_et
    return model_et, results_et


@app.cell
def _(mo):
    mo.md(r"""
    The Extra Tree baseline shows relatively stable performance across the five engine-level folds. Four folds produce very similar errors, while the third fold is moderately more difficult, leading to a mean validation MAE of approximately 10.62 cycles with a standard deviation of 0.27 cycles.
    """)
    return


@app.cell
def _(
    ExtraTreesRegressor,
    X_fold,
    cv,
    groups_fold,
    notify_done,
    run_grouped_grid_search,
    y_fold,
):
    param_grid_et = {
        "max_depth": [14, 16, 18],
        "min_samples_leaf": [2, 4, 6],
    }

    model_et_GS = ExtraTreesRegressor(
        n_estimators=100, random_state=42, min_samples_leaf=1.0
    )

    grid_et = run_grouped_grid_search(
        model=model_et_GS,
        param_grid=param_grid_et,
        X=X_fold,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    grid_et.best_params_
    return grid_et


@app.cell
def _(grid_et):
    -grid_et.best_score_
    return


@app.cell
def _(grid_et):
    grid_et.best_estimator_
    return


@app.cell
def _(
    RandomForestRegressor,
    X_fold,
    cv,
    evaluate_grouped_cv,
    groups_fold,
    model_et,
    notify_done,
    y_fold,
):
    model_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    results_rf = evaluate_grouped_cv(
        model=model_rf,
        X=X_fold,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    results_rf
    return (results_rf,)


@app.cell
def _(
    GradientBoostingRegressor,
    X_fold,
    cv,
    evaluate_grouped_cv,
    groups_fold,
    notify_done,
    y_fold,
):
    model_gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        min_samples_leaf=5,
        max_depth=5,
        random_state=42,
    )

    results_gb = evaluate_grouped_cv(
        model=model_gb,
        X=X_fold,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    results_gb
    return (results_gb,)


@app.cell
def _(
    Pipeline,
    SVR,
    StandardScaler,
    X_fold,
    cv,
    evaluate_grouped_cv,
    groups_fold,
    notify_done,
    y_fold,
):
    model_SVR = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale")),
        ]
    )

    results_SVR = evaluate_grouped_cv(
        model=model_SVR,
        X=X_fold,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    results_SVR
    return (results_SVR,)


@app.cell
def _(pd, results_SVR, results_et, results_gb, results_rf):
    CV_results_comparison = [
        {
            "Model": ["Gradient Boosting", "Random Forest", "Extra Trees", "SVR"],
            "mean_MAE": [
                results_gb["mean_MAE"],
                results_rf["mean_MAE"],
                results_et["mean_MAE"],
                results_SVR["mean_MAE"],
            ],
            "std_MAE": [
                results_gb["std_MAE"],
                results_rf["std_MAE"],
                results_et["std_MAE"],
                results_SVR["std_MAE"],
            ],
            "mean_RMSE": [
                results_gb["mean_RMSE"],
                results_rf["mean_RMSE"],
                results_et["mean_RMSE"],
                results_SVR["mean_RMSE"],
            ],
            "std_RMSE": [
                results_gb["std_RMSE"],
                results_rf["std_RMSE"],
                results_et["std_RMSE"],
                results_SVR["std_RMSE"],
            ],
        }
    ]

    pd.DataFrame(CV_results_comparison).round(3)
    return


@app.cell
def _(
    GradientBoostingRegressor,
    X_fold,
    cv,
    groups_fold,
    notify_done,
    run_grouped_grid_search,
    y_fold,
):
    param_grid_gb = [
        {
            "learning_rate": [0.04],
            "n_estimators": [300],
        },
        {
            "learning_rate": [0.03],
            "n_estimators": [400],
        },
        {
            "learning_rate": [0.02],
            "n_estimators": [600],
        },
    ]

    model_gb_GS = GradientBoostingRegressor(
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )

    grid_gb = run_grouped_grid_search(
        model=model_gb_GS,
        param_grid=param_grid_gb,
        X=X_fold,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    grid_gb.best_params_
    return grid_gb, param_grid_gb


@app.cell
def _(grid_gb):
    -grid_gb.best_score_
    return


@app.cell
def _(grid_gb):
    grid_gb.best_estimator_
    return


@app.cell
def _(temporal_config):
    temporal_config_short = {
        "mean": [5],
        "delta": [5],
        "slope": [5],
    }

    temporal_config_medium = {
        "mean": [10],
        "delta": [10],
        "slope": [10],
    }

    temporal_config_long = {
        "mean": [20],
        "delta": [20],
        "slope": [20],
    }

    configs = [
        temporal_config_short,
        temporal_config_medium,
        temporal_config_long,
        temporal_config,
    ]
    return (configs,)


@app.cell
def _(
    GradientBoostingRegressor,
    add_temporal_features,
    available_sensors,
    configs,
    cv,
    df_train,
    evaluate_grouped_cv,
    notify_done,
):
    results_gb_conftuning = []

    for config in configs:
        df_train_temporal_conftuning = add_temporal_features(
            df_train,
            available_sensors,
            config,
        )

        temporal_features_conftuning = [
            column
            for column in df_train_temporal_conftuning.columns
            if any(suffix in column for suffix in ["_mean_", "_delta_", "_slope_"])
        ]

        raw_temporal_cycle_features_conftuning = [
            "cycle",
            *available_sensors,
            *temporal_features_conftuning,
        ]

        groups_fold_conftuning = df_train_temporal_conftuning["engine_id"]

        X_fold_conftuning = df_train_temporal_conftuning[
            raw_temporal_cycle_features_conftuning
        ]

        y_fold_conftuning = df_train_temporal_conftuning["rul_capped"]

        model_gb_tuned = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
        )

        results_gb_tuned = evaluate_grouped_cv(
            model=model_gb_tuned,
            X=X_fold_conftuning,
            y=y_fold_conftuning,
            groups=groups_fold_conftuning,
            cv=cv,
        )

        results_gb_conftuning.append(results_gb_tuned)

    notify_done()

    return (results_gb_conftuning,)


@app.cell
def _(results_gb_conftuning):
    results_gb_conftuning
    return


@app.cell
def _(
    GradientBoostingRegressor,
    available_sensors,
    cv,
    df_train_temporal,
    evaluate_grouped_cv,
    groups_fold,
    notify_done,
    raw_temporal_cycle_features,
    y_fold,
):
    results_gb_sensor_ablation_summary = []

    for _sensor in available_sensors:
        ablation_features = [
            c
            for c in raw_temporal_cycle_features
            if c == _sensor or c.startswith(f"{_sensor}_")
        ]

        ablation_feature_columns = [
            c for c in raw_temporal_cycle_features if c not in ablation_features
        ]

        X_fold_sensor_ablation = df_train_temporal[ablation_feature_columns]

        model_gb_sensor_ablation = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
        )

        results_gb_sensor_ablation = evaluate_grouped_cv(
            model=model_gb_sensor_ablation,
            X=X_fold_sensor_ablation,
            y=y_fold,
            groups=groups_fold,
            cv=cv,
        )

        results_gb_sensor_ablation_summary.append(results_gb_sensor_ablation)

    notify_done()

    results_gb_sensor_ablation_summary.round(3)
    return


@app.cell
def _(
    GradientBoostingRegressor,
    available_sensors,
    cv,
    df_train_temporal,
    evaluate_grouped_cv,
    groups_fold,
    notify_done,
    raw_temporal_cycle_features,
    y_fold,
):
    # no raw sensors

    ablation_features_no_raw = available_sensors

    ablation_feature_columns_no_raw = [
        c for c in raw_temporal_cycle_features if c not in ablation_features_no_raw
    ]

    X_fold_ablation_no_raw = df_train_temporal[ablation_feature_columns_no_raw]

    model_gb_ablation_no_raw = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )

    results_gb_ablation_no_raw = evaluate_grouped_cv(
        model=model_gb_ablation_no_raw,
        X=X_fold_ablation_no_raw,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    results_gb_ablation_no_raw
    return (ablation_features_no_raw,)


@app.cell
def _(
    GradientBoostingRegressor,
    cv,
    df_train_temporal,
    evaluate_grouped_cv,
    groups_fold,
    notify_done,
    raw_temporal_cycle_features,
    y_fold,
):
    # no means
    ablation_features_no_means = [
        c for c in raw_temporal_cycle_features if "_mean_" in c
    ]

    ablation_feature_columns_no_means = [
        c for c in raw_temporal_cycle_features if c not in ablation_features_no_means
    ]

    X_fold_ablation_no_means = df_train_temporal[ablation_feature_columns_no_means]

    model_gb_ablation_no_means = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )

    results_gb_ablation_no_means = evaluate_grouped_cv(
        model=model_gb_ablation_no_means,
        X=X_fold_ablation_no_means,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    results_gb_ablation_no_means
    return (ablation_features_no_means,)


@app.cell
def _(
    GradientBoostingRegressor,
    cv,
    df_train_temporal,
    evaluate_grouped_cv,
    groups_fold,
    notify_done,
    raw_temporal_cycle_features,
    y_fold,
):
    # no deltas
    ablation_features_no_deltas = [
        c for c in raw_temporal_cycle_features if "_delta_" in c
    ]

    ablation_feature_columns_no_deltas = [
        c for c in raw_temporal_cycle_features if c not in ablation_features_no_deltas
    ]

    X_fold_ablation_no_deltas = df_train_temporal[ablation_feature_columns_no_deltas]

    model_gb_ablation_no_deltas = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )

    results_gb_ablation_no_deltas = evaluate_grouped_cv(
        model=model_gb_ablation_no_deltas,
        X=X_fold_ablation_no_deltas,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    results_gb_ablation_no_deltas
    return


@app.cell
def _(
    GradientBoostingRegressor,
    cv,
    df_train_temporal,
    evaluate_grouped_cv,
    groups_fold,
    notify_done,
    raw_temporal_cycle_features,
    y_fold,
):
    # no slope
    ablation_features_no_slope = [
        c for c in raw_temporal_cycle_features if "_slope_" in c
    ]

    ablation_feature_columns_no_slope = [
        c for c in raw_temporal_cycle_features if c not in ablation_features_no_slope
    ]

    X_fold_ablation_no_slope = df_train_temporal[ablation_feature_columns_no_slope]

    model_gb_ablation_no_slope = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )

    results_gb_ablation_no_slope = evaluate_grouped_cv(
        model=model_gb_ablation_no_slope,
        X=X_fold_ablation_no_slope,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    results_gb_ablation_no_slope
    return (results_gb_ablation_no_slope,)


@app.cell
def _(
    GradientBoostingRegressor,
    cv,
    df_train_temporal,
    evaluate_grouped_cv,
    groups_fold,
    notify_done,
    raw_temporal_cycle_features,
    results_gb_ablation_no_slope,
    y_fold,
):
    # no means nor deltas
    ablation_features_no_means_nor_deltas = [
        c for c in raw_temporal_cycle_features if "_mean_" in c or "_delta_" in c
    ]

    ablation_feature_columns_no_means_nor_deltas = [
        c
        for c in raw_temporal_cycle_features
        if c not in ablation_features_no_means_nor_deltas
    ]

    X_fold_ablation_no_means_nor_deltas = df_train_temporal[
        ablation_feature_columns_no_means_nor_deltas
    ]

    model_gb_ablation_no_means_nor_deltas = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )

    results_gb_ablation_no_means_nor_deltas = evaluate_grouped_cv(
        model=model_gb_ablation_no_means_nor_deltas,
        X=X_fold_ablation_no_means_nor_deltas,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    results_gb_ablation_no_means_nor_deltas
    return


@app.cell
def _(
    GradientBoostingRegressor,
    ablation_features_no_means,
    ablation_features_no_raw,
    cv,
    df_train_temporal,
    evaluate_grouped_cv,
    groups_fold,
    notify_done,
    raw_temporal_cycle_features,
    y_fold,
):
    # no means nor raw
    ablation_feature_columns_no_means_nor_raw = [
        c
        for c in raw_temporal_cycle_features
        if c not in ablation_features_no_raw and c not in ablation_features_no_means
    ]

    X_fold_ablation_no_means_nor_raw = df_train_temporal[
        ablation_feature_columns_no_means_nor_raw
    ]

    model_gb_ablation_no_means_nor_raw = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )

    results_gb_ablation_no_means_nor_raw = evaluate_grouped_cv(
        model=model_gb_ablation_no_means_nor_raw,
        X=X_fold_ablation_no_means_nor_raw,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    results_gb_ablation_no_means_nor_raw
    return


@app.cell
def _(df_train_temporal, raw_temporal_cycle_features):
    final_feature_set = [
        column for column in raw_temporal_cycle_features if "_mean_" not in column
    ]

    X_fold_final = df_train_temporal[final_feature_set]
    return (X_fold_final,)


@app.cell
def _(
    GradientBoostingRegressor,
    X_fold_final,
    cv,
    groups_fold,
    notify_done,
    param_grid_gb,
    run_grouped_grid_search,
    y_fold,
):
    grid_search_gb_final = run_grouped_grid_search(
        model=GradientBoostingRegressor(random_state=42),
        param_grid=param_grid_gb,
        X=X_fold_final,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()
    return (grid_search_gb_final,)


@app.cell
def _(grid_search_gb_final, pd):
    results_gb_after_ablation = pd.DataFrame(grid_search_gb_final.cv_results_)

    results_gb_after_ablation["mean_MAE"] = -results_gb_after_ablation["mean_test_MAE"]

    top_4_gb_after_ablation = results_gb_after_ablation.nsmallest(
        4,
        "mean_MAE",
    )

    top_4_gb_after_ablation[
        [
            "params",
            "mean_MAE",
            "std_test_MAE",
            "mean_test_RMSE",
        ]
    ]
    return


@app.cell
def _(
    GradientBoostingRegressor,
    after_ablation_feature_set,
    cv,
    df_train_temporal,
    evaluate_grouped_cv,
    groups_fold,
    notify_done,
    y_fold,
):
    X_fold_final_comparison = df_train_temporal[after_ablation_feature_set]

    model_gb_final_comparison = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )

    results_gb_final_comparison = evaluate_grouped_cv(
        model=model_gb_final_comparison,
        X=X_fold_final_comparison,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    results_gb_final_comparison
    return X_fold_final_comparison, results_gb_final_comparison


@app.cell
def _(
    ExtraTreesRegressor,
    X_fold_final_comparison,
    cv,
    evaluate_grouped_cv,
    groups_fold,
    notify_done,
    y_fold,
):
    model_et_final_comparison = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=14,
        max_features=1.0,
        min_samples_leaf=4,
        random_state=42,
    )

    results_et_final_comparison = evaluate_grouped_cv(
        model=model_et_final_comparison,
        X=X_fold_final_comparison,
        y=y_fold,
        groups=groups_fold,
        cv=cv,
    )

    notify_done()

    results_et_final_comparison
    return (results_et_final_comparison,)


@app.cell
def _(pd, results_et_final_comparison, results_gb_final_comparison):
    pd.DataFrame(
        {
            "Model": ["Gradient Boosting", "Extra Trees"],
            "mean_MAE": [
                results_gb_final_comparison["mean_MAE"],
                results_et_final_comparison["mean_MAE"],
            ],
            "stds_MAE": [
                results_gb_final_comparison["std_MAE"],
                results_et_final_comparison["std_MAE"],
            ],
            "means_RMSE": [
                results_gb_final_comparison["mean_RMSE"],
                results_et_final_comparison["mean_RMSE"],
            ],
            "stds_RMSE": [
                results_gb_final_comparison["std_RMSE"],
                results_et_final_comparison["std_RMSE"],
            ],
        }
    ).round(3)
    return


@app.cell
def _(
    GradientBoostingRegressor,
    after_ablation_feature_set,
    df_train_temporal,
    df_validation_temporal,
    mean_absolute_error,
    notify_done,
    root_mean_squared_error,
):
    X_train_after_ablation = df_train_temporal[after_ablation_feature_set]
    X_validation_after_ablation = df_validation_temporal[after_ablation_feature_set]

    y_train = df_train_temporal["rul_capped"]
    y_validation = df_validation_temporal["rul_capped"]

    model_gb_evaluation = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        loss="absolute_error",
    )

    model_gb_evaluation.fit(X_train_after_ablation, y_train)

    y_pred_train_gb_evaluation = model_gb_evaluation.predict(X_train_after_ablation)
    y_pred_validation_gb_evaluation = model_gb_evaluation.predict(
        X_validation_after_ablation
    )

    mae_train_gb_evaluation = mean_absolute_error(y_train, y_pred_train_gb_evaluation)
    mae_validation_gb_evaluation = mean_absolute_error(
        y_validation, y_pred_validation_gb_evaluation
    )

    rmse_train_gb_evaluation = root_mean_squared_error(
        y_train, y_pred_train_gb_evaluation
    )
    rmse_validation_gb_evaluation = root_mean_squared_error(
        y_validation, y_pred_validation_gb_evaluation
    )

    notify_done()
    return (
        X_train_after_ablation,
        X_validation_after_ablation,
        mae_train_gb_evaluation,
        mae_validation_gb_evaluation,
        rmse_train_gb_evaluation,
        rmse_validation_gb_evaluation,
        y_train,
        y_validation,
    )


@app.cell
def _(
    ExtraTreesRegressor,
    X_train_after_ablation,
    X_validation_after_ablation,
    mean_absolute_error,
    notify_done,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    model_et_evaluation = ExtraTreesRegressor(
        n_estimators=100,
        min_samples_leaf=4,
        max_features=1.0,
        random_state=42,
    )

    model_et_evaluation.fit(X_train_after_ablation, y_train)

    y_pred_train_et_evaluation = model_et_evaluation.predict(X_train_after_ablation)
    y_pred_validation_et_evaluation = model_et_evaluation.predict(
        X_validation_after_ablation
    )

    mae_train_et_evaluation = mean_absolute_error(y_train, y_pred_train_et_evaluation)
    mae_validation_et_evaluation = mean_absolute_error(
        y_validation, y_pred_validation_et_evaluation
    )

    rmse_train_et_evaluation = root_mean_squared_error(
        y_train, y_pred_train_et_evaluation
    )
    rmse_validation_et_evaluation = root_mean_squared_error(
        y_validation, y_pred_validation_et_evaluation
    )

    notify_done()
    return (
        mae_train_et_evaluation,
        mae_validation_et_evaluation,
        rmse_train_et_evaluation,
        rmse_validation_et_evaluation,
    )


@app.cell
def _(
    mae_train_et_evaluation,
    mae_train_gb_evaluation,
    mae_validation_et_evaluation,
    mae_validation_gb_evaluation,
    pd,
    rmse_train_et_evaluation,
    rmse_train_gb_evaluation,
    rmse_validation_et_evaluation,
    rmse_validation_gb_evaluation,
):
    pd.DataFrame(
        {
            "Model": ["Gradient Boosting", "Extra Tree"],
            "MAE_train": [mae_train_gb_evaluation, mae_train_et_evaluation],
            "RMSE_train": [rmse_train_gb_evaluation, rmse_train_et_evaluation],
            "MAE_validation": [
                mae_validation_gb_evaluation,
                mae_validation_et_evaluation,
            ],
            "RMSE_validation": [
                rmse_validation_gb_evaluation,
                rmse_validation_et_evaluation,
            ],
        }
    ).round(3)
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
    return


@app.cell
def _(GradientBoostingRegressor):
    gb_loss_models = {
        "squared_error": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            min_samples_leaf=5,
            loss="squared_error",
            random_state=42,
        ),
        "absolute_error": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            min_samples_leaf=5,
            loss="absolute_error",
            random_state=42,
        ),
        "huber_0.9": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            min_samples_leaf=5,
            loss="huber",
            alpha=0.9,
            random_state=42,
        ),
    }
    return (gb_loss_models,)


@app.cell
def _(
    X_fold_final,
    cv,
    evaluate_grouped_cv,
    gb_loss_models,
    groups_fold,
    pd,
    y_fold,
):
    gb_loss_results = []

    for name, model in gb_loss_models.items():
        metrics = evaluate_grouped_cv(
            model,
            X_fold_final,
            y_fold,
            groups_fold,
            cv=cv,
        )

        gb_loss_results.append(
            {
                "configuration": name,
                **metrics,
            }
        )

    gb_loss_results_df = pd.DataFrame(gb_loss_results).sort_values("mean_MAE")
    gb_loss_results_df.round(3)
    return


@app.cell
def _(GradientBoostingRegressor):
    gb_subsample_models = {
        "subsample_1.0": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            min_samples_leaf=5,
            loss="absolute_error",
            subsample=1.0,
            random_state=42,
        ),
        "subsample_0.8": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            min_samples_leaf=5,
            loss="absolute_error",
            subsample=0.8,
            random_state=42,
        ),
        "subsample_0.6": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            min_samples_leaf=5,
            loss="absolute_error",
            subsample=0.6,
            random_state=42,
        ),
    }
    return (gb_subsample_models,)


@app.cell
def _(
    X_fold_final,
    cv,
    evaluate_grouped_cv,
    gb_subsample_models,
    groups_fold,
    pd,
    y_fold,
):
    gb_subsample_results = []

    for name, model in gb_subsample_models.items():
        metrics = evaluate_grouped_cv(
            model,
            X_fold_final,
            y_fold,
            groups_fold,
            cv=cv,
        )

        gb_subsample_results.append(
            {
                "configuration": name,
                **metrics,
            }
        )

    pd.DataFrame(gb_subsample_results).sort_values("mean_MAE").round(3)
    return


if __name__ == "__main__":
    app.run()
