import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split, GroupKFold
    from sklearn.base import clone
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from pathlib import Path

    from hawkshot.features.temporal import add_temporal_features
    from hawkshot.modeling.oof import generate_oof_predictions
    from hawkshot.experiments.runner import run_or_reuse_oof
    from hawkshot.modeling.evaluation import summarize_oof_predictions

    from hawkshot.data.cmapss import (
        SENSOR_COLUMNS,
        filter_constant_sensors,
        load_fd001,
    )

    df_raw = load_fd001("data/raw/cmapss")
    df_filtered, removed_sensors = filter_constant_sensors(df_raw)

    available_sensors = [
        sensor for sensor in SENSOR_COLUMNS if sensor in df_filtered.columns
    ]
    return (
        DummyRegressor,
        GroupKFold,
        LinearRegression,
        Path,
        Pipeline,
        StandardScaler,
        add_temporal_features,
        available_sensors,
        df_filtered,
        mo,
        np,
        pd,
        run_or_reuse_oof,
        summarize_oof_predictions,
        train_test_split,
        clone,
        root_mean_squared_error,
        mean_absolute_error,
        generate_oof_predictions,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Experimental Setup and Model Candidates
    """)
    return


@app.cell
def _(df_filtered, pd, train_test_split):
    _max_cycle = df_filtered.groupby("engine_id")["cycle"].transform("max")

    df_rul = df_filtered.assign(
        max_cycle=_max_cycle, rul=_max_cycle - df_filtered["cycle"]
    )

    rul_cap = 125

    df_prepared = df_rul.assign(rul_capped=df_rul["rul"].clip(upper=rul_cap))

    engine_ids = df_prepared["engine_id"].unique()

    train_engines, validation_engines = train_test_split(
        engine_ids, test_size=0.2, random_state=42
    )

    df_train = df_prepared[df_prepared["engine_id"].isin(train_engines)].copy()

    df_validation = df_prepared[
        df_prepared["engine_id"].isin(validation_engines)
    ].copy()

    train_lifetime = df_train.groupby("engine_id")["max_cycle"].first()

    validation_lifetime = df_validation.groupby("engine_id")["max_cycle"].first()

    pd.DataFrame(
        {
            "train": train_lifetime.describe(),
            "validation": validation_lifetime.describe(),
        }
    ).round(1)
    return (df_train,)


@app.cell
def _(add_temporal_features, available_sensors, df_train):
    temporal_config = {
        "mean": [5, 10, 20],
        "delta": [5, 10],
        "slope": [10, 20],
    }

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
    return X_cv_full, folds, groups_cv, y_cv


@app.cell
def _(raw_temporal_cycle_features):
    feature_sets = {
        "full": raw_temporal_cycle_features,
        "no_means": [c for c in raw_temporal_cycle_features if "_mean_" not in c],
        "no_means_no_deltas": [
            c
            for c in raw_temporal_cycle_features
            if "_mean_" not in c and "_delta_" not in c
        ],
    }
    return feature_sets


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Out-of-fold Predictions
    """)
    return


@app.cell
def _(
    DummyRegressor,
    Path,
    X_cv_full,
    df_train_temporal,
    folds,
    groups_cv,
    run_or_reuse_oof,
    y_cv,
):
    oof_dummy = run_or_reuse_oof(
        model=DummyRegressor(strategy="median"),
        X=X_cv_full,
        y=y_cv,
        groups=groups_cv,
        cycles=df_train_temporal["cycle"],
        cv_splits=folds,
        checkpoint_path=Path("results/oof/dummy_median.csv"),
    )

    oof_dummy
    return (oof_dummy,)


@app.cell
def _(oof_dummy, summarize_oof_predictions):
    dummy_scores = summarize_oof_predictions(
        oof_dummy,
        model_id="dummy_median",
    )
    return (dummy_scores,)


@app.cell
def _(
    LinearRegression,
    Path,
    Pipeline,
    StandardScaler,
    X_cv_full,
    df_train_temporal,
    dummy_scores,
    folds,
    groups_cv,
    pd,
    run_or_reuse_oof,
    summarize_oof_predictions,
    y_cv,
):
    linear_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    oof_linear = run_or_reuse_oof(
        model=linear_model,
        X=X_cv_full,
        y=y_cv,
        groups=groups_cv,
        cycles=df_train_temporal["cycle"],
        cv_splits=folds,
        checkpoint_path=Path("results/oof/linear_regression.csv"),
    )

    linear_scores = summarize_oof_predictions(
        oof_linear,
        model_id="linear_regression",
    )

    comparison_df = pd.DataFrame(
        [
            dummy_scores,
            linear_scores,
        ]
    )

    comparison_df.round(3)
    return


if __name__ == "__main__":
    app.run()
