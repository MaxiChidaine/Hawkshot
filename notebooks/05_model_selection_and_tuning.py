import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import winsound

    from pathlib import Path
    from sklearn.model_selection import ParameterGrid
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

    from itertools import product

    df_raw = load_fd001("data/raw/cmapss")
    df_filtered, removed_sensors = filter_constant_sensors(df_raw)

    available_sensors = [
        sensor for sensor in SENSOR_COLUMNS if sensor in df_filtered.columns
    ]

    checkpoint_dir = Path("results/gb_tuning_b2")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        GridSearchCV,
        GroupKFold,
        ParameterGrid,
        Path,
        Pipeline,
        RandomForestRegressor,
        SVR,
        StandardScaler,
        add_temporal_features,
        available_sensors,
        checkpoint_dir,
        clone,
        df_filtered,
        mean_absolute_error,
        mo,
        np,
        pd,
        perf_counter,
        product,
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
def _(clone, mean_absolute_error, np, perf_counter, root_mean_squared_error):
    def evaluate_grouped_cv(model, X, y, groups, cv):
        maes = []
        rmses = []
        fit_times = []
        predict_times = []
        mean_signed_errors = []
        overestimation_rates = []
        mean_overestimationss = []

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

            error = y_pred - y_validation_fold

            positive_errors = error[error > 0]

            overestimation_rates.append((error > 0).mean())
            mean_signed_errors.append(error.mean())
            mean_overestimationss.append(
                positive_errors.mean() if len(positive_errors) > 0 else 0.0
            )

        return {
            "mean_MAE": np.mean(maes),
            "std_MAE": np.std(maes),
            "mean_RMSE": np.mean(rmses),
            "std_RMSE": np.std(rmses),
            "mean_fit_time": np.mean(fit_times),
            "mean_predict_time": np.mean(predict_times),
            "overestimation_rate": np.mean(overestimation_rates),
            "mean_signed_error": np.mean(mean_signed_errors),
            "mean_overestimation": np.mean(mean_overestimationss),
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
def _(mean_absolute_error, perf_counter, root_mean_squared_error):
    def evaluate_model(model, X_train, X_validation, y_train, y_validation):
        fit_start = perf_counter()
        model.fit(X_train, y_train)
        fit_time = perf_counter() - fit_start

        predict_train_start = perf_counter()
        y_pred_train = model.predict(X_train)
        predict_train_time = perf_counter() - predict_train_start

        predict_validation_start = perf_counter()
        y_pred_validation = model.predict(X_validation)
        predict_validation_time = perf_counter() - predict_validation_start

        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_validation = mean_absolute_error(y_validation, y_pred_validation)

        rmse_train = root_mean_squared_error(y_train, y_pred_train)
        rmse_validation = root_mean_squared_error(y_validation, y_pred_validation)

        return {
            "mae_train": mae_train,
            "mae_validation": mae_validation,
            "rmse_train": rmse_train,
            "rmse_validation": rmse_validation,
            "fit_time": fit_time,
            "predict_train_time": predict_train_time,
            "predict_validation_time": predict_validation_time,
            "y_pred_validation": y_pred_validation,
        }

    return (evaluate_model,)


@app.cell
def _(np):
    from sklearn.metrics import make_scorer

    def overestimation_rate(y_true, y_pred):
        return (y_pred > y_true).mean()

    def mean_signed_error_scorer(estimator, X, y_true):
        y_pred = estimator.predict(X)

        return float(np.mean(y_pred - np.asarray(y_true)))

    def mean_overestimation(y_true, y_pred):
        error = y_pred - y_true
        positive_errors = error[error > 0]

        if len(positive_errors) == 0:
            return 0.0

        return positive_errors.mean()

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "overestimation_rate": make_scorer(
            overestimation_rate,
            greater_is_better=False,
        ),
        "mean_signed_error": mean_signed_error_scorer,
        "mean_overestimation": make_scorer(
            mean_overestimation,
            greater_is_better=False,
        ),
    }
    return (scoring,)


@app.cell
def _(GroupKFold, df_train_temporal, np, pd, raw_temporal_cycle_features):
    cv = GroupKFold(n_splits=5)

    groups_cv = df_train_temporal["engine_id"]
    X_cv_full = df_train_temporal[raw_temporal_cycle_features]
    y_cv = df_train_temporal["rul_capped"]

    folds = list(cv.split(X_cv_full, y_cv, groups_cv))

    length_train_fold = []
    length_validation_fold = []
    list_intersection = []
    list_fold = []

    for _fold_number, (_train_index, __validation_index) in enumerate(folds, start=1):
        train_engines_fold = groups_cv.iloc[_train_index].unique()
        validation_engines_fold = groups_cv.iloc[__validation_index].unique()
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
    return X_cv_full, cv, groups_cv, y_cv


@app.cell
def _():
    CV_results_comparison = []

    columns_to_display = [
        "model",
        "mean_MAE",
        "std_MAE",
        "mean_RMSE",
        "std_RMSE",
        "overestimation_rate",
        "mean_signed_error",
        "mean_overestimation",
    ]
    return CV_results_comparison, columns_to_display


@app.cell
def _(
    CV_results_comparison,
    ExtraTreesRegressor,
    X_cv_full,
    cv,
    evaluate_grouped_cv,
    groups_cv,
    y_cv,
):
    model_et = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    results_et = evaluate_grouped_cv(
        model=model_et,
        X=X_cv_full,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    CV_results_comparison.append(
        {
            "model": "model_et",
            **results_et,
        }
    )
    return


@app.cell
def _(
    CV_results_comparison,
    RandomForestRegressor,
    X_cv_full,
    cv,
    evaluate_grouped_cv,
    groups_cv,
    notify_done,
    y_cv,
):
    model_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    results_rf = evaluate_grouped_cv(
        model=model_rf,
        X=X_cv_full,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    notify_done()

    CV_results_comparison.append(
        {
            "model": "model_rf",
            **results_rf,
        }
    )
    return


@app.cell
def _(
    CV_results_comparison,
    GradientBoostingRegressor,
    X_cv_full,
    cv,
    evaluate_grouped_cv,
    groups_cv,
    notify_done,
    y_cv,
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
        X=X_cv_full,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    notify_done()

    CV_results_comparison.append(
        {
            "model": "model_gb",
            **results_gb,
        }
    )
    return


@app.cell
def _(
    CV_results_comparison,
    Pipeline,
    SVR,
    StandardScaler,
    X_cv_full,
    cv,
    evaluate_grouped_cv,
    groups_cv,
    notify_done,
    y_cv,
):
    model_SVR = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale")),
        ]
    )

    results_SVR = evaluate_grouped_cv(
        model=model_SVR,
        X=X_cv_full,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    notify_done()

    CV_results_comparison.append(
        {
            "model": "model_SVR",
            **results_SVR,
        }
    )
    return


@app.cell
def _(CV_results_comparison, columns_to_display, pd):
    pd.DataFrame(CV_results_comparison).sort_values("overestimation_rate")[
        columns_to_display
    ].round(3)
    return


@app.cell
def _(available_sensors, raw_temporal_cycle_features):
    feature_sets = {
        "full": raw_temporal_cycle_features,
        "no_means": [c for c in raw_temporal_cycle_features if "_mean_" not in c],
        "no_deltas": [c for c in raw_temporal_cycle_features if "_delta_" not in c],
        "no_slopes": [c for c in raw_temporal_cycle_features if "_slope_" not in c],
        "no_means_no_deltas": [
            c
            for c in raw_temporal_cycle_features
            if "_mean_" not in c and "_delta_" not in c
        ],
        "no_means_no_raw": [
            c
            for c in raw_temporal_cycle_features
            if "_mean_" not in c and c not in available_sensors
        ],
    }
    return (feature_sets,)


@app.cell
def _(mo):
    mo.md(r"""
    # GRADIENT BOOSTING
    """)
    return


@app.cell
def _(
    GradientBoostingRegressor,
    X_cv_full,
    cv,
    groups_cv,
    notify_done,
    run_grouped_grid_search,
    y_cv,
):
    param_grid_gb_coarse = [
        {
            "learning_rate": [0.015],
            "n_estimators": [800],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [2, 5, 10],
        },
        {
            "learning_rate": [0.02],
            "n_estimators": [600],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [2, 5, 10],
        },
        {
            "learning_rate": [0.03],
            "n_estimators": [400],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [2, 5, 10],
        },
        {
            "learning_rate": [0.04],
            "n_estimators": [300],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [2, 5, 10],
        },
        {
            "learning_rate": [0.06],
            "n_estimators": [200],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [2, 5, 10],
        },
    ]

    model_gb_coarse = GradientBoostingRegressor(
        random_state=42,
    )

    grid_gb_coarse = run_grouped_grid_search(
        model=model_gb_coarse,
        param_grid=param_grid_gb_coarse,
        X=X_cv_full,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    notify_done()
    return (grid_gb_coarse,)


@app.cell
def _(GradientBoostingRegressor, grid_gb_coarse):
    gb_coarse_params = grid_gb_coarse.best_params_

    gb_coarse_tuned = GradientBoostingRegressor(
        **gb_coarse_params,
        random_state=42,
    )
    return gb_coarse_params, gb_coarse_tuned


@app.cell
def _(
    cv,
    df_train_temporal,
    evaluate_grouped_cv,
    feature_sets,
    gb_coarse_tuned,
    groups_cv,
    notify_done,
    y_cv,
):
    feature_set_results_gb = []

    for _name, _features in feature_sets.items():
        _metrics = evaluate_grouped_cv(
            model=gb_coarse_tuned,
            X=df_train_temporal[_features],
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        )

        feature_set_results_gb.append(
            {
                "feature_set": _name,
                "n_features": len(_features),
                **_metrics,
            }
        )

    notify_done()
    return (feature_set_results_gb,)


@app.cell
def _(feature_set_results_gb, pd):
    feature_set_results_gb_df = pd.DataFrame(feature_set_results_gb).sort_values(
        "mean_MAE"
    )
    feature_set_results_gb_df
    return


@app.cell
def _(df_train_temporal, feature_sets):
    final_feature_set_name_gb = "no_means"
    final_feature_set_gb = feature_sets[final_feature_set_name_gb]
    X_cv_gb = df_train_temporal[final_feature_set_gb]
    return (X_cv_gb,)


@app.cell(hide_code=True)
def _():
    tuning_A_param_grid_gb = {
        "squared_error": {
            "loss": "squared_error",
        },
        "absolute_error": {
            "loss": "absolute_error",
        },
        "huber_0.80": {
            "loss": "huber",
            "alpha": 0.80,
        },
        "huber_0.90": {
            "loss": "huber",
            "alpha": 0.90,
        },
        "huber_0.95": {
            "loss": "huber",
            "alpha": 0.95,
        },
        "quantile_0.50": {
            "loss": "quantile",
            "alpha": 0.50,
        },
        "quantile_0.40": {
            "loss": "quantile",
            "alpha": 0.40,
        },
        "quantile_0.30": {
            "loss": "quantile",
            "alpha": 0.30,
        },
    }

    subsample_values = [0.6, 0.8, 1.0]
    max_features_values = [0.75, 1.0]
    return max_features_values, subsample_values, tuning_A_param_grid_gb


@app.cell
def _(
    GradientBoostingRegressor,
    X_cv_gb,
    cv,
    evaluate_grouped_cv,
    gb_coarse_params,
    groups_cv,
    max_features_values,
    notify_done,
    product,
    subsample_values,
    tuning_A_param_grid_gb,
    y_cv,
):
    gb_tuning_A_results = []

    for _config_name, _loss_params in tuning_A_param_grid_gb.items():
        for _subsample, _max_features in product(
            subsample_values,
            max_features_values,
        ):
            _model = GradientBoostingRegressor(
                **gb_coarse_params,
                **_loss_params,
                subsample=_subsample,
                max_features=_max_features,
                random_state=42,
            )

            _metrics = evaluate_grouped_cv(
                model=_model,
                X=X_cv_gb,
                y=y_cv,
                groups=groups_cv,
                cv=cv,
            )

            gb_tuning_A_results.append(
                {
                    "configuration": _config_name,
                    "loss": _loss_params["loss"],
                    "alpha": _loss_params.get("alpha"),
                    "subsample": _subsample,
                    "max_features": _max_features,
                    **_metrics,
                }
            )

    notify_done()
    return (gb_tuning_A_results,)


@app.cell
def _(gb_tuning_A_results, pd):
    gb_tuning_a_columns = [
        "configuration",
        "loss",
        "alpha",
        "subsample",
        "max_features",
        "mean_MAE",
        "std_MAE",
        "mean_RMSE",
        "overestimation_rate",
        "mean_signed_error",
        "mean_overestimation",
    ]

    gb_tuning_a_results_df = (
        pd.DataFrame(gb_tuning_A_results)[gb_tuning_a_columns]
        .sort_values("mean_MAE")
        .round(3)
    )
    gb_tuning_a_results_df
    return (gb_tuning_a_results_df,)


@app.cell
def _(gb_tuning_a_results_df):
    best_mae = gb_tuning_a_results_df["mean_MAE"].min()

    gb_tuning_a_candidates = gb_tuning_a_results_df[
        gb_tuning_a_results_df["mean_MAE"] <= best_mae * 1.05
    ].sort_values(["overestimation_rate", "mean_MAE"])

    gb_tuning_a_candidates
    return


@app.cell
def _():
    gb_tuning_b_profiles = {
        "absolute_error_accuracy": {
            "loss": "absolute_error",
            "subsample": 0.6,
            "max_features": 1.0,
        },
        "huber_0.80": {
            "loss": "huber",
            "alpha": 0.80,
            "subsample": 0.6,
            "max_features": 1.0,
        },
        "squared_error_balanced": {
            "loss": "squared_error",
            "subsample": 0.6,
            "max_features": 0.75,
        },
        "quantile_0.40_conservative": {
            "loss": "quantile",
            "alpha": 0.40,
            "subsample": 0.6,
            "max_features": 0.75,
        },
        "quantile_0.30_conservative": {
            "loss": "quantile",
            "alpha": 0.30,
            "subsample": 1.0,
            "max_features": 0.75,
        },
    }
    return (gb_tuning_b_profiles,)


@app.cell
def _():
    boosting_schedules = [
        (0.02, 600),
        (0.03, 400),
        (0.04, 300),
        (0.05, 240),
        (0.06, 200),
    ]

    max_depth_values = [3, 5, 7]
    min_samples_leaf_values = [2, 5, 10]
    return boosting_schedules, max_depth_values, min_samples_leaf_values


@app.cell
def _(
    boosting_schedules,
    gb_tuning_b_profiles,
    max_depth_values,
    min_samples_leaf_values,
):
    tuning_B_param_grid_gb = []

    for _profiles in gb_tuning_b_profiles.values():
        for _learning_rate, _n_estimators in boosting_schedules:
            _params = {
                "learning_rate": [_learning_rate],
                "n_estimators": [_n_estimators],
                "max_depth": max_depth_values,
                "min_samples_leaf": min_samples_leaf_values,
                "loss": [_profiles["loss"]],
                "subsample": [_profiles["subsample"]],
                "max_features": [_profiles["max_features"]],
            }

            if "alpha" in _profiles:
                _params["alpha"] = [_profiles["alpha"]]

            tuning_B_param_grid_gb.append(_params)
    return (tuning_B_param_grid_gb,)


@app.cell
def _(
    GradientBoostingRegressor,
    GridSearchCV,
    X_cv_gb,
    cv,
    groups_cv,
    notify_done,
    scoring,
    tuning_B_param_grid_gb,
    y_cv,
):
    grid_gb_tuning_b = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid=tuning_B_param_grid_gb,
        scoring=scoring,
        refit=False,
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    grid_gb_tuning_b.fit(X_cv_gb, y_cv, groups=groups_cv)

    notify_done()
    return (grid_gb_tuning_b,)


@app.cell
def _(grid_gb_tuning_b, pd):
    gb_tuning_b_results = pd.DataFrame(grid_gb_tuning_b.cv_results_).round(3)

    gb_tuning_b_results["mean_MAE"] = -gb_tuning_b_results["mean_test_MAE"]

    gb_tuning_b_results["mean_RMSE"] = -gb_tuning_b_results["mean_test_RMSE"]

    gb_tuning_b_results["overestimation_rate"] = -gb_tuning_b_results[
        "mean_test_overestimation_rate"
    ]

    gb_tuning_b_results["mean_overestimation"] = -gb_tuning_b_results[
        "mean_test_mean_overestimation"
    ]

    gb_tuning_b_results["mean_signed_error"] = gb_tuning_b_results[
        "mean_test_mean_signed_error"
    ]

    gb_tuning_b_results
    return


@app.cell
def _():
    gb_tuning_b2_profiles = {
        "absolute_error": {
            "loss": "absolute_error",
            "subsample": 0.6,
            "max_features": 1.0,
        },
        "huber_0.80": {
            "loss": "huber",
            "alpha": 0.80,
            "subsample": 0.6,
            "max_features": 1.0,
        },
        "squared_error": {
            "loss": "squared_error",
            "subsample": 0.6,
            "max_features": 0.75,
        },
        "quantile_0.40": {
            "loss": "quantile",
            "alpha": 0.40,
            "subsample": 0.6,
            "max_features": 0.75,
        },
        "quantile_0.30": {
            "loss": "quantile",
            "alpha": 0.30,
            "subsample": 1.0,
            "max_features": 0.75,
        },
    }
    return (gb_tuning_b2_profiles,)


@app.cell
def _():
    boosting_schedules_b2 = [
        (0.010, 1200),
        (0.015, 800),
        (0.020, 600),
        (0.025, 480),
        (0.030, 400),
    ]

    max_depth_values_b2 = [6, 7, 8, 9]
    min_samples_leaf_high = [5, 10, 15, 20]
    min_samples_leaf_middle = [2, 5, 10, 15]
    return (
        boosting_schedules_b2,
        max_depth_values_b2,
        min_samples_leaf_high,
        min_samples_leaf_middle,
    )


@app.cell
def _(
    ParameterGrid,
    boosting_schedules_b2,
    gb_tuning_b2_profiles,
    max_depth_values_b2,
    min_samples_leaf_high,
    min_samples_leaf_middle,
):
    param_grids_gb_tuning_b2 = {}

    for _profile_name, _profile in gb_tuning_b2_profiles.items():
        if _profile_name in {
            "absolute_error",
            "huber_0.80",
            "quantile_0.40",
        }:
            _leaf_values = min_samples_leaf_high
        else:
            _leaf_values = min_samples_leaf_middle

        _profile_grid = []

        for _learning_rate, _n_estimators in boosting_schedules_b2:
            _grid = {
                "learning_rate": [_learning_rate],
                "n_estimators": [_n_estimators],
                "max_depth": max_depth_values_b2,
                "min_samples_leaf": _leaf_values,
                "loss": [_profile["loss"]],
                "subsample": [_profile["subsample"]],
                "max_features": [_profile["max_features"]],
            }

            if "alpha" in _profile:
                _grid["alpha"] = [_profile["alpha"]]

            _profile_grid.append(_grid)

        param_grids_gb_tuning_b2[_profile_name] = _profile_grid

    for _profile_name, _param_grid in param_grids_gb_tuning_b2.items():
        _n_candidates = len(list(ParameterGrid(_param_grid)))

        print(_profile_name, "->", _n_candidates, "candidates")

        assert _n_candidates == 80
    return (param_grids_gb_tuning_b2,)


@app.cell
def _(
    GradientBoostingRegressor,
    GridSearchCV,
    ParameterGrid,
    X_cv_gb,
    checkpoint_dir,
    cv,
    groups_cv,
    notify_done,
    param_grids_gb_tuning_b2,
    pd,
    scoring,
    y_cv,
):
    gb_tuning_b2_results = []

    for _profile_name, _param_grid in param_grids_gb_tuning_b2.items():
        _checkpoint_path = checkpoint_dir / f"{_profile_name}.csv"

        if _checkpoint_path.exists():
            print(f"[SKIP] {_profile_name}: checkpoint already exists.")

            _results = pd.read_csv(_checkpoint_path)

            gb_tuning_b2_results.append(_results)

            continue

        _n_candidates = len(list(ParameterGrid(_param_grid)))

        assert _n_candidates == 80

        print(
            f"\n[START] {_profile_name}: "
            f"{_n_candidates} candidates / "
            f"{_n_candidates * 5} fits"
        )

        _grid_search = GridSearchCV(
            estimator=GradientBoostingRegressor(
                random_state=42,
            ),
            param_grid=_param_grid,
            scoring=scoring,
            refit=False,
            cv=cv,
            n_jobs=-1,
            pre_dispatch="n_jobs",
            error_score="raise",
            verbose=2,
        )

        _grid_search.fit(
            X_cv_gb,
            y_cv,
            groups=groups_cv,
        )

        _results = pd.DataFrame(_grid_search.cv_results_)

        _results["profile"] = _profile_name

        _results["mean_MAE"] = -_results["mean_test_MAE"]

        _results["mean_RMSE"] = -_results["mean_test_RMSE"]

        _results["overestimation_rate"] = -_results["mean_test_overestimation_rate"]

        _results["mean_overestimation"] = -_results["mean_test_mean_overestimation"]

        _results["mean_signed_error"] = _results["mean_test_mean_signed_error"]

        _results.to_csv(
            _checkpoint_path,
            index=False,
        )

        gb_tuning_b2_results.append(_results)

        print(f"[DONE] {_profile_name}")

        notify_done()
    return (gb_tuning_b2_results,)


@app.cell
def _(gb_tuning_b2_results):
    gb_tuning_b2_results
    return


@app.cell
def _(Path, gb_tuning_b2_profiles):
    checkpoint_dir_b3 = Path("results/gb_tuning_b3")
    checkpoint_dir_b3.mkdir(parents=True, exist_ok=True)

    gb_tuning_b3_specs = {
        "absolute_error": {
            "schedules": [
                (0.015, 800),
            ],
            "max_depth": [8, 9, 10, 11],
            "min_samples_leaf": [15, 20, 25, 30],
        },
        "huber_0.80": {
            "schedules": [
                (0.0075, 1600),
                (0.0100, 1200),
                (0.0150, 800),
            ],
            "max_depth": [8, 9, 10],
            "min_samples_leaf": [15, 20, 25],
        },
        "squared_error": {
            "schedules": [
                (0.0075, 1600),
                (0.0100, 1200),
                (0.0150, 800),
            ],
            "max_depth": [8, 9, 10],
            "min_samples_leaf": [10, 15, 20],
        },
        "quantile_0.40": {
            "schedules": [
                (0.0075, 1600),
                (0.0100, 1200),
                (0.0150, 800),
            ],
            "max_depth": [8, 9, 10],
            "min_samples_leaf": [15, 20, 25, 30],
        },
        "quantile_0.30": {
            "schedules": [
                (0.020, 600),
            ],
            "max_depth": [8, 9, 10, 11],
            "min_samples_leaf": [10, 15, 20, 25],
        },
    }

    param_grids_gb_tuning_b3 = {}

    for _profile_name, _spec in gb_tuning_b3_specs.items():
        _profile = gb_tuning_b2_profiles[_profile_name]
        _profile_grid = []

        for _learning_rate, _n_estimators in _spec["schedules"]:
            _grid = {
                "learning_rate": [_learning_rate],
                "n_estimators": [_n_estimators],
                "max_depth": _spec["max_depth"],
                "min_samples_leaf": _spec["min_samples_leaf"],
                "loss": [_profile["loss"]],
                "subsample": [_profile["subsample"]],
                "max_features": [_profile["max_features"]],
            }

            if "alpha" in _profile:
                _grid["alpha"] = [_profile["alpha"]]

            _profile_grid.append(_grid)

        param_grids_gb_tuning_b3[_profile_name] = _profile_grid
    return checkpoint_dir_b3, param_grids_gb_tuning_b3


@app.cell
def _(ParameterGrid, param_grids_gb_tuning_b3):
    expected_candidates_b3 = {
        "absolute_error": 16,
        "huber_0.80": 27,
        "squared_error": 27,
        "quantile_0.40": 36,
        "quantile_0.30": 16,
    }

    _total_candidates = 0

    for _profile_name, _param_grid in param_grids_gb_tuning_b3.items():
        _n_candidates = len(list(ParameterGrid(_param_grid)))

        print(f"{_profile_name}: {_n_candidates} candidates / {_n_candidates * 5} fits")

        assert _n_candidates == expected_candidates_b3[_profile_name]

        _total_candidates += _n_candidates

    assert _total_candidates == 122

    print(f"\nTOTAL: {_total_candidates} candidates / {_total_candidates * 5} fits")
    return (expected_candidates_b3,)


@app.cell
def _(
    GradientBoostingRegressor,
    GridSearchCV,
    ParameterGrid,
    X_cv_gb,
    checkpoint_dir_b3,
    cv,
    expected_candidates_b3,
    groups_cv,
    notify_done,
    param_grids_gb_tuning_b3,
    pd,
    scoring,
    y_cv,
):
    gb_tuning_b3_results = []

    for _profile_name, _param_grid in param_grids_gb_tuning_b3.items():
        _checkpoint_path = checkpoint_dir_b3 / f"{_profile_name}.csv"

        if _checkpoint_path.exists():
            print(f"[SKIP] {_profile_name}: checkpoint already exists")

            _results = pd.read_csv(_checkpoint_path)

            gb_tuning_b3_results.append(_results)

            continue

        _n_candidates = len(list(ParameterGrid(_param_grid)))

        assert _n_candidates == expected_candidates_b3[_profile_name]

        print(
            f"\n[START] {_profile_name}: "
            f"{_n_candidates} candidates / "
            f"{_n_candidates * 5} fits"
        )

        _grid_search = GridSearchCV(
            estimator=GradientBoostingRegressor(
                random_state=42,
            ),
            param_grid=_param_grid,
            scoring=scoring,
            refit=False,
            cv=cv,
            n_jobs=-1,
            pre_dispatch="n_jobs",
            error_score="raise",
            verbose=2,
        )

        _grid_search.fit(
            X_cv_gb,
            y_cv,
            groups=groups_cv,
        )

        _results = pd.DataFrame(_grid_search.cv_results_)

        _results["profile"] = _profile_name

        _results["mean_MAE"] = -_results["mean_test_MAE"]

        _results["mean_RMSE"] = -_results["mean_test_RMSE"]

        _results["overestimation_rate"] = -_results["mean_test_overestimation_rate"]

        _results["mean_overestimation"] = -_results["mean_test_mean_overestimation"]

        _results["mean_signed_error"] = _results["mean_test_mean_signed_error"]

        _results.to_csv(
            _checkpoint_path,
            index=False,
        )

        gb_tuning_b3_results.append(_results)

        print(f"[DONE] {_profile_name}-> {_checkpoint_path}")

        notify_done()
    return (gb_tuning_b3_results,)


@app.cell
def _(checkpoint_dir_b3, gb_tuning_b3_results, pd):
    gb_tuning_b3_results_df = pd.concat(
        gb_tuning_b3_results,
        ignore_index=True,
    )

    assert len(gb_tuning_b3_results_df) == 122

    gb_tuning_b3_results_df.to_csv(
        checkpoint_dir_b3 / "gb_tuning_b3_results.csv",
        index=False,
    )

    gb_tuning_b3_results_df.shape
    return


@app.cell
def _(pd):
    gb_finalists = {
        "absolute_error": {
            "loss": "absolute_error",
            "learning_rate": 0.015,
            "n_estimators": 800,
            "max_depth": 10,
            "min_samples_leaf": 30,
            "subsample": 0.6,
            "max_features": 1.0,
        },
        "huber_0.80": {
            "loss": "huber",
            "alpha": 0.80,
            "learning_rate": 0.0075,
            "n_estimators": 1600,
            "max_depth": 10,
            "min_samples_leaf": 25,
            "subsample": 0.6,
            "max_features": 1.0,
        },
        "squared_error": {
            "loss": "squared_error",
            "learning_rate": 0.01,
            "n_estimators": 1200,
            "max_depth": 10,
            "min_samples_leaf": 20,
            "subsample": 0.6,
            "max_features": 0.75,
        },
        "quantile_0.40": {
            "loss": "quantile",
            "alpha": 0.40,
            "learning_rate": 0.0075,
            "n_estimators": 1600,
            "max_depth": 10,
            "min_samples_leaf": 30,
            "subsample": 0.6,
            "max_features": 0.75,
        },
        "quantile_0.30": {
            "loss": "quantile",
            "alpha": 0.30,
            "learning_rate": 0.02,
            "n_estimators": 600,
            "max_depth": 11,
            "min_samples_leaf": 20,
            "subsample": 1.0,
            "max_features": 0.75,
        },
    }

    gb_finalists_df = pd.DataFrame(gb_finalists)
    gb_finalists_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    # EXTRA TREES
    """)
    return


@app.cell
def _(
    ExtraTreesRegressor,
    X_cv_full,
    cv,
    groups_cv,
    notify_done,
    pd,
    run_grouped_grid_search,
    y_cv,
):
    param_grid_et_coarse = {
        "max_depth": [14, 16, 18],
        "min_samples_leaf": [2, 4, 6],
    }

    model_et_coarse = ExtraTreesRegressor(
        n_estimators=100, random_state=42, min_samples_leaf=1.0
    )

    grid_et_coarse = run_grouped_grid_search(
        model=model_et_coarse,
        param_grid=param_grid_et_coarse,
        X=X_cv_full,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    notify_done()

    grid_et_coarse_results_df = pd.DataFrame(grid_et_coarse.cv_results_).round(3)
    grid_et_coarse_results_df
    return


@app.cell
def _(ExtraTreesRegressor, grid_et_reference, pd):
    et_reference_params = grid_et_reference.best_params_

    et_reference = ExtraTreesRegressor(
        **et_reference_params,
        random_state=42,
    )

    pd.DataFrame(grid_et_reference.cv_results_).to_csv(
        "results/et_pre_tuning_results.csv",
        index=False,
    )
    return (et_reference,)


@app.cell
def _(
    cv,
    df_train_temporal,
    et_reference,
    evaluate_grouped_cv,
    feature_sets,
    groups_cv,
    notify_done,
    y_cv,
):
    feature_set_results_et = []

    for _name, _features in feature_sets.items():
        _metrics = evaluate_grouped_cv(
            model=et_reference,
            X=df_train_temporal[_features],
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        )

        feature_set_results_et.append(
            {
                "feature_set": _name,
                "n_features": len(_features),
                **_metrics,
            }
        )

    notify_done()
    return (feature_set_results_et,)


@app.cell
def _(feature_set_results_et, pd):
    feature_set_results_et_df = pd.DataFrame(feature_set_results_et).sort_values(
        "mean_MAE"
    )

    feature_set_results_et_df.round(3)
    return


@app.cell
def _(df_train_temporal, feature_sets):
    final_feature_set_et = feature_sets["no_means_no_deltas"]

    X_cv_et = df_train_temporal[final_feature_set_et]
    return (X_cv_et,)


@app.cell
def _(
    ExtraTreesRegressor,
    GridSearchCV,
    X_cv_et,
    cv,
    groups_cv,
    notify_done,
    scoring,
    y_cv,
):
    param_grid_et_a = {
        "max_depth": [14, 16, 18, 20, 22, 26, None],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "max_features": [0.5, 0.75, 1.0],
    }

    grid_et_a = GridSearchCV(
        estimator=ExtraTreesRegressor(
            random_state=42,
            n_estimators=100,
        ),
        param_grid=param_grid_et_a,
        scoring=scoring,
        refit=False,
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    grid_et_a.fit(X_cv_et, y_cv, groups=groups_cv)

    notify_done()
    return (grid_et_a,)


@app.cell
def _(grid_et_a, pd):
    et_a_results = pd.DataFrame(grid_et_a.cv_results_).round(3)

    et_a_results["mean_MAE"] = -et_a_results["mean_test_MAE"]

    et_a_results["mean_RMSE"] = -et_a_results["mean_test_RMSE"]

    et_a_results["overestimation_rate"] = -et_a_results["mean_test_overestimation_rate"]

    et_a_results["mean_overestimation"] = -et_a_results["mean_test_mean_overestimation"]

    et_a_results["mean_signed_error"] = et_a_results["mean_test_mean_signed_error"]

    et_a_results
    return


@app.cell
def _(grid_et_a, pd):
    pd.DataFrame(grid_et_a.cv_results_).to_csv(
        "results/et_tuning_a_results.csv",
        index=False,
    )
    return


@app.cell
def _(Path):
    from datetime import datetime
    import time

    checkpoint_dir_et_b = Path("results/et_tuning_b")
    checkpoint_dir_et_b.mkdir(parents=True, exist_ok=True)

    et_b_chunks = []

    for _n_estimators in [200, 500, 1000]:
        for _criterion in [
            "squared_error",
            "poisson",
            "absolute_error",
        ]:
            et_b_chunks.append(
                {
                    "chunk_id": (f"n{_n_estimators}__{_criterion}__boostrap_false"),
                    "param_grid": {
                        "n_estimators": [_n_estimators],
                        "max_depth": [20, 22, 24, 26],
                        "min_samples_leaf": [1, 2, 3, 4],
                        "min_samples_split": [2, 4, 8],
                        "max_features": [0.875, 1.0],
                        "criterion": [_criterion],
                        "bootstrap": [False],
                    },
                }
            )

            for _max_samples in [0.7, 0.85, 1.0]:
                et_b_chunks.append(
                    {
                        "chunk_id": (
                            f"n{_n_estimators}"
                            f"__{_criterion}"
                            "__boostrap_true"
                            f"__samples_{_max_samples}"
                        ),
                        "param_grid": {
                            "n_estimators": [_n_estimators],
                            "max_depth": [20, 22, 24, 26],
                            "min_samples_leaf": [1, 2, 3, 4],
                            "min_samples_split": [2, 4, 8],
                            "max_features": [0.875, 1.0],
                            "criterion": [_criterion],
                            "bootstrap": [True],
                            "max_samples": [_max_samples],
                        },
                    }
                )
    return checkpoint_dir_et_b, datetime, et_b_chunks, time


@app.cell
def _(ParameterGrid, et_b_chunks):
    assert len(et_b_chunks) == 36

    _total_candidates = 0

    for _chunks in et_b_chunks:
        _n_candidates = len(list(ParameterGrid(_chunks["param_grid"])))

        assert _n_candidates == 96
        _total_candidates += _n_candidates

    print(
        f"{len(et_b_chunks)} chunks\n"
        f"{_total_candidates} candidates\n"
        f"{_total_candidates * 5} fits"
    )
    return


@app.cell
def _(
    ExtraTreesRegressor,
    GridSearchCV,
    ParameterGrid,
    X_cv_et,
    checkpoint_dir_et_b,
    cv,
    datetime,
    et_b_chunks,
    groups_cv,
    pd,
    scoring,
    time,
    y_cv,
):
    et_b_results = []

    for _chunk in et_b_chunks:
        _chunk_id = _chunk["chunk_id"]
        _param_grid = _chunk["param_grid"]

        _checkpoint_path = checkpoint_dir_et_b / f"{_chunk_id}.csv"

        _running_path = checkpoint_dir_et_b / f"{_chunk_id}.running"

        if _checkpoint_path.exists():
            _existing = pd.read_csv(_checkpoint_path)

            if len(_existing) == 96:
                print(f"[SKIP] {_chunk_id}")

                et_b_results.append(_existing)

                continue
            print(f"[INVALID CHECKPOINT] {_chunk_id} -> recompute")

        _n_candidates = len(list(ParameterGrid(_param_grid)))

        assert _n_candidates == 96

        print(
            f"\n{'=' * 70}\n"
            f"[START] {_chunk_id}\n"
            f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"{_n_candidates} candidates / "
            f"{_n_candidates * 5} fits\n"
            f"{'=' * 70}"
        )

        _running_path.write_text(f"Started : {datetime.now().isoformat()}")

        _start_time = time.perf_counter()

        _grid_search = GridSearchCV(
            estimator=ExtraTreesRegressor(
                random_state=42,
                n_jobs=1,
            ),
            param_grid=_param_grid,
            scoring=scoring,
            refit=False,
            cv=cv,
            n_jobs=10,
            pre_dispatch="n_jobs",
            error_score="raise",
            verbose=1,
        )

        _grid_search.fit(
            X_cv_et,
            y_cv,
            groups=groups_cv,
        )

        _results = pd.DataFrame(_grid_search.cv_results_)

        _results = pd.DataFrame(_grid_search.cv_results_)

        assert len(_results) == 96

        _results["mean_MAE"] = -_results["mean_test_MAE"]

        _results["mean_RMSE"] = -_results["mean_test_RMSE"]

        _results["overestimation_rate"] = -_results["mean_test_overestimation_rate"]

        _results["mean_overstimation"] = -_results["mean_test_mean_overestimation"]

        _results["mean_signed_error"] = _results["mean_test_mean_signed_error"]

        _results["chunk_id"] = _chunk_id

        _tmp_path = checkpoint_dir_et_b / f"{_chunk_id}.tmp.csv"

        _results.to_csv(
            _tmp_path,
            index=False,
        )

        _tmp_path.replace(_checkpoint_path)

        if _running_path.exists():
            _running_path.unlink()

        et_b_results.append(_results)

        _elapsed = time.perf_counter() - _start_time

        print(
            f"[DONE] {_chunk_id}\n"
            f"Duration: {_elapsed / 60:.1f} min\n"
            f"Checkpoint: {_checkpoint_path}"
        )
    return (et_b_results,)


@app.cell
def _(checkpoint_dir_et_b, et_b_results, et_tuing_b_results_df, pd):
    et_tuning_b_results_df = pd.concat(
        et_b_results,
        ignore_index=True,
    )

    assert len(et_tuning_b_results_df) == 3456

    et_tuing_b_results_df.to_csv(
        checkpoint_dir_et_b / "et_tuning_b_results.csv",
        index=False,
    )

    et_tuning_b_results_df.shape
    return


@app.cell
def _(
    ExtraTreesRegressor,
    X_cv_et,
    cv,
    groups_cv,
    pd,
    run_grouped_grid_search,
    y_cv,
):
    param_grid_et_c = {"n_estimators": [200, 400, 600, 1000]}

    grid_et_c = run_grouped_grid_search(
        model=ExtraTreesRegressor(
            max_depth=23,
            min_samples_leaf=3,
            min_samples_split=2,
            max_features=1.0,
        ),
        param_grid=param_grid_et_c,
        X=X_cv_et,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    grid_et_c_results_df = pd.DataFrame(grid_et_c.cv_results_).round(3)
    grid_et_c_results_df
    return


@app.cell
def _(
    X_train_after_ablation,
    X_validation_after_ablation,
    evaluate_model,
    final_evaluation_models,
    notify_done,
    pd,
    y_train,
    y_validation,
):
    final_evaluation_results = []

    for _name, _model in final_evaluation_models.items():
        _metrics = evaluate_model(
            _model,
            X_train_after_ablation,
            X_validation_after_ablation,
            y_train,
            y_validation,
        )

        final_evaluation_results.append(
            {
                "model": _name,
                **_metrics,
            }
        )

    notify_done()

    pd.DataFrame(final_evaluation_results).sort_values("mae_validation").round(3)
    return (final_evaluation_results,)


@app.cell
def _(final_evaluation_results, y_validation):
    error_gb = final_evaluation_results[0]["y_pred_validation"] - y_validation
    error_et = final_evaluation_results[1]["y_pred_validation"] - y_validation
    return error_et, error_gb


@app.cell
def _(error_et, error_gb):
    overestimation_rate_gb = (error_gb > 0).mean()
    overestimation_rate_et = (error_et > 0).mean()

    mean_signed_error_gb = error_gb.mean()
    mean_signed_error_et = error_et.mean()

    mean_overestimations_gb = error_gb[error_gb > 0].mean()
    mean_overestimations_et = error_et[error_et > 0].mean()
    return (
        mean_overestimations_et,
        mean_overestimations_gb,
        mean_signed_error_et,
        mean_signed_error_gb,
        overestimation_rate_et,
        overestimation_rate_gb,
    )


@app.cell
def _(
    mean_overestimations_et,
    mean_overestimations_gb,
    mean_signed_error_et,
    mean_signed_error_gb,
    overestimation_rate_et,
    overestimation_rate_gb,
    pd,
):
    pd.DataFrame(
        {
            "model": ["Gradient Boosting", "Extra Trees"],
            "overestimation_rate": [overestimation_rate_gb, overestimation_rate_et],
            "mean_signed_error": [mean_signed_error_gb, mean_signed_error_et],
            "mean_overestimation": [mean_overestimations_gb, mean_overestimations_et],
        }
    ).round(3)
    return


app._unparsable_cell(
    r"""
    loss_quantile_models = {
        "quantile_0.5" : GradientBoostingRegressor(
            n_estimators=300
            learning_rate=0.04
            max_depth=5
            min_samples_leaf=5
            subsample=0.6
            max_features=1.0
            random_state=42
            loss="quantile",
            alpha=0.5,
        ),
        "quantile_0.4" : GradientBoostingRegressor(
            n_estimators=300
            learning_rate=0.04
            max_depth=5
            min_samples_leaf=5
            subsample=0.6
            max_features=1.0
            random_state=42
            loss="quantile",
            alpha=0.4,
        ),
        "quantile_0.3" : GradientBoostingRegressor(
            n_estimators=300
            learning_rate=0.04
            max_depth=5
            min_samples_leaf=5
            subsample=0.6
            max_features=1.0
            random_state=42
            loss="quantile",
            alpha=0.3,
        ),
    }
    """,
    name="_",
)


@app.cell
def _(
    X_fold_final,
    columns_to_display,
    cv,
    et_n_estimators_models,
    evaluate_grouped_cv,
    groups_cv,
    notify_done,
    pd,
    y_cv,
):
    loss_quantile_results = []

    for _name, _model in et_n_estimators_models.items():
        _metrics = evaluate_grouped_cv(
            _model,
            X_fold_final,
            y_cv,
            groups_cv,
            cv=cv,
        )

        loss_quantile_results.append(
            {
                "configuration": _name,
                **_metrics,
            }
        )

    notify_done()

    pd.DataFrame(loss_quantile_results).sort_values("overestimation_rate")[
        columns_to_display
    ].round(3)
    return


if __name__ == "__main__":
    app.run()
