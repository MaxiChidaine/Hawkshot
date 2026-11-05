import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import json

    from sklearn.model_selection import train_test_split, GroupKFold
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from pathlib import Path

    from hawkshot.features.temporal import add_temporal_features
    from hawkshot.experiments.runner import run_or_reuse_oof
    from hawkshot.modeling.evaluation import (
        summarize_oof_predictions,
        analyze_oof_errors_by_rul,
        analyze_oof_errors_by_engine,
    )
    from hawkshot.experiments.finalists import build_model_from_config

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
        analyze_oof_errors_by_engine,
        analyze_oof_errors_by_rul,
        available_sensors,
        build_model_from_config,
        df_filtered,
        json,
        mo,
        np,
        pd,
        run_or_reuse_oof,
        summarize_oof_predictions,
        train_test_split,
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
    folds,
    groups_cv,
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
    return (linear_scores,)


@app.cell
def _(Path, json):
    config_path = Path("configs/oof_finalists.json")

    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    finalists = payload["finalists"]

    assert payload["schema_version"] == 1
    assert len(finalists) == 12
    assert len({config["id"] for config in finalists}) == 12

    print(f"{len(finalists)} configurations chargées.")

    for config in finalists:
        print(config["id"], "→", config["feature_set"])
    return (finalists,)


@app.cell
def _(finalists):
    import hawkshot.experiments.finalists as finalists_module

    models = {}

    for finalist in finalists:
        model = finalists_module.build_model_from_config(finalist)
        models[finalist["id"]] = model

        print(finalist["id"], "→", type(model).__name__)
    return


@app.cell
def _(X_cv_full):
    full_columns = X_cv_full.columns.tolist()

    mean_columns = [col for col in full_columns if "_mean_" in col]

    delta_columns = [col for col in full_columns if "_delta_" in col]

    print("Total features :", len(full_columns))
    print("Mean features :", len(mean_columns))
    print("Delta features :", len(delta_columns))
    return delta_columns, mean_columns


@app.cell
def _(X_cv_full, delta_columns, mean_columns, y_cv):
    oof_feature_sets = {
        "full": X_cv_full,
        "no_means": X_cv_full.drop(columns=mean_columns),
        "no_means_no_deltas": X_cv_full.drop(columns=mean_columns + delta_columns),
    }

    assert mean_columns
    assert delta_columns

    for name, X_features in oof_feature_sets.items():
        assert len(X_features) == len(y_cv)
        assert X_features.index.equals(X_cv_full.index)

        print(
            f"{name}: "
            f"{X_features.shape[0]} observations, "
            f"{X_features.shape[1]} features"
        )
    return (oof_feature_sets,)


@app.cell
def _(
    Path,
    build_model_from_config,
    df_train_temporal,
    dummy_scores,
    finalists,
    folds,
    groups_cv,
    linear_scores,
    oof_feature_sets,
    pd,
    run_or_reuse_oof,
    summarize_oof_predictions,
    y_cv,
):
    _scores = [dummy_scores, linear_scores]

    for _finalist in finalists:
        _model_id = _finalist["id"]

        _model = build_model_from_config(_finalist)

        _X_model = oof_feature_sets[_finalist["feature_set"]]

        print(f"Processing: {_model_id}", flush=True)

        _oof = run_or_reuse_oof(
            model=_model,
            X=_X_model,
            y=y_cv,
            groups=groups_cv,
            cycles=df_train_temporal["cycle"],
            cv_splits=folds,
            checkpoint_path=Path("results/oof") / f"{_model_id}.csv",
        )

        _model_scores = summarize_oof_predictions(
            _oof,
            model_id=_model_id,
        )

        _scores.append(_model_scores)

        print(f"  MAE: {_model_scores['MAE']:.3f}", flush=True)

    comparison_df = pd.DataFrame(_scores)

    assert len(comparison_df) == 14
    assert comparison_df["Model"].is_unique

    comparison_df.sort_values("MAE").reset_index(drop=True).round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Temporal Analysis
    """)
    return


@app.cell
def _(analyze_oof_errors_by_rul, pd):
    _linear_oof = pd.read_csv("results/oof/linear_regression.csv")

    linear_errors_by_rul = analyze_oof_errors_by_rul(
        _linear_oof,
        model_id="linear_regression",
    )

    linear_errors_by_rul[
        [
            "rul_band",
            "n_observations",
            "MAE",
            "RMSE",
            "overestimation_rate",
            "p90_abs_error",
        ]
    ].round(3)
    return


@app.cell
def _(Path, analyze_oof_errors_by_rul, finalists, pd):
    _oof_paths = sorted(Path("results/oof").glob("*.csv"))

    _expected_ids = {_config["id"] for _config in finalists}
    _found_ids = {_path.stem for _path in _oof_paths}

    if _missing := _expected_ids - _found_ids:
        raise FileNotFoundError(f"Missing OOF checkpoints: {_missing}")

    if len(_oof_paths) != 14:
        raise ValueError(f"Expected 14 OOF checkpoints, found {len(_oof_paths)}.")

    _rul_results = []

    for _path in _oof_paths:
        _oof_df = pd.read_csv(_path)

        _rul_results.append(
            analyze_oof_errors_by_rul(
                _oof_df,
                model_id=_path.stem,
            )
        )

    rul_comparison_df = pd.concat(
        _rul_results,
        ignore_index=True,
    )

    print("Models:", rul_comparison_df["Model"].nunique())
    print("RUL bands:", rul_comparison_df["rul_band"].nunique())
    print("Rows:", len(rul_comparison_df))
    return (rul_comparison_df,)


@app.cell
def _(rul_comparison_df):
    rul_mae_table = rul_comparison_df.pivot(
        index="Model",
        columns="rul_band",
        values="MAE",
    )

    rul_mae_table.round(2)
    return


@app.cell
def _(rul_comparison_df):
    rul_overestimation_table = (
        rul_comparison_df.pivot(
            index="Model",
            columns="rul_band",
            values="overestimation_rate",
        )
        * 100
    )

    rul_overestimation_table.round(1)
    return


@app.cell
def _(rul_comparison_df):
    import matplotlib.pyplot as _plt

    _model_labels = {
        "linear_regression": "Linear Regression",
        "xgboost_pseudohuber": "XGBoost Pseudo-Huber",
        "gradientboosting_quantile_0_30": "GB Quantile 0.30",
        "extratrees_absolute_error": "Extra Trees Absolute",
    }

    _band_order = [f"{_start}-{_start + 9}" for _start in range(0, 120, 10)] + [
        "120-124"
    ]

    _fig, _ax = _plt.subplots(figsize=(12, 5.5))

    for _model_id, _label in _model_labels.items():
        _model_data = (
            rul_comparison_df.loc[rul_comparison_df["Model"] == _model_id]
            .set_index("rul_band")
            .reindex(_band_order)
        )

        if _model_data["MAE"].isna().any():
            raise ValueError(f"Missing RUL data for {_model_id}")

        _ax.plot(
            _band_order,
            _model_data["MAE"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=_label,
        )

    _ax.set(
        title="Prediction Error Across Remaining Useful Life",
        xlabel="True RUL (cycles)",
        ylabel="Mean Absolute Error (cycles)",
    )

    _ax.grid(alpha=0.25)
    _ax.legend()
    _ax.tick_params(axis="x", rotation=45)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(rul_comparison_df):
    rul_large_overestimation_table = rul_comparison_df.pivot(
        index="Model",
        columns="rul_band",
        values="overestimation_gt_20_rate",
    ).mul(100)

    rul_large_overestimation_table[
        ["0-9", "10-19", "20-29", "30-39", "60-69", "70-79"]
    ].round(1)
    return


@app.cell
def _(Path, analyze_oof_errors_by_engine, pd, rul_comparison_df):
    _engine_tables = []

    for _model_id in rul_comparison_df["Model"].unique():
        _checkpoint = Path("results/oof") / f"{_model_id}.csv"

        _oof = pd.read_csv(_checkpoint)

        _engine_tables.append(
            analyze_oof_errors_by_engine(
                _oof,
                model_id=_model_id,
                rul_range=(0, 30),
            )
        )

    engine_errors_near_failure_df = pd.concat(
        _engine_tables,
        ignore_index=True,
    )

    assert engine_errors_near_failure_df["Model"].nunique() == 14
    assert len(engine_errors_near_failure_df) == 14 * 80

    engine_errors_near_failure_df.sort_values(
        "max_overestimation",
        ascending=False,
    ).head(15).round(3)
    return (engine_errors_near_failure_df,)


@app.cell
def _(engine_errors_near_failure_df, finalists):
    _finalist_ids = {_config["id"] for _config in finalists}

    _finalist_engine_errors = engine_errors_near_failure_df.loc[
        engine_errors_near_failure_df["Model"].isin(_finalist_ids)
    ]

    engine_risk_summary_df = (
        _finalist_engine_errors.groupby("Model")
        .agg(
            n_engines=("engine_id", "nunique"),
            affected_engines=(
                "n_overestimations_gt_20",
                lambda _values: (_values > 0).sum(),
            ),
            worst_overestimation=(
                "max_overestimation",
                "max",
            ),
            p90_max_overestimation=(
                "max_overestimation",
                lambda _values: _values.quantile(0.90),
            ),
            longest_streak=(
                "longest_overestimation_gt_20_streak",
                "max",
            ),
        )
        .sort_values("affected_engines", ascending=False)
        .reset_index()
    )

    engine_risk_summary_df.round(2)
    return


@app.cell
def _(Path, engine_errors_near_failure_df, pd):
    import matplotlib.pyplot as _plt

    _pseudo_engines = set(
        engine_errors_near_failure_df.loc[
            (engine_errors_near_failure_df["Model"] == "xgboost_pseudohuber")
            & (engine_errors_near_failure_df["n_overestimations_gt_20"] > 0),
            "engine_id",
        ]
    )

    _quantile_engines = set(
        engine_errors_near_failure_df.loc[
            (engine_errors_near_failure_df["Model"] == "gradientboosting_quantile_0_30")
            & (engine_errors_near_failure_df["n_overestimations_gt_20"] > 0),
            "engine_id",
        ]
    )

    print("XGB Pseudo-Huber :", len(_pseudo_engines))
    print("GB Quantile 0.30 :", len(_quantile_engines))
    print("Moteurs communs :", len(_pseudo_engines & _quantile_engines))
    print("Seulement XGB :", sorted(_pseudo_engines - _quantile_engines))
    print("Seulement GB Quantile :", sorted(_quantile_engines - _pseudo_engines))

    _model_labels = {
        "xgboost_pseudohuber": "XGB Pseudo-Huber",
        "gradientboosting_quantile_0_30": "GB Quantile 0.30",
    }

    _engine_ids = [
        2,
        12,
        min(_pseudo_engines & _quantile_engines),
    ]

    _oof_by_model = {
        _model_id: pd.read_csv(Path("results/oof") / f"{_model_id}.csv")
        for _model_id in _model_labels
    }

    _fig, _axes = _plt.subplots(
        len(_engine_ids),
        1,
        figsize=(12, 10),
    )

    for _ax, _engine_id in zip(_axes, _engine_ids, strict=True):
        _reference = _oof_by_model["xgboost_pseudohuber"]

        _reference = _reference.loc[
            (_reference["engine_id"] == _engine_id)
            & (_reference["y_true"].between(0, 30))
        ].sort_values("cycle")

        _ax.plot(
            _reference["cycle"],
            _reference["y_true"],
            color="black",
            linewidth=2,
            label="True RUL",
        )

        _ax.plot(
            _reference["cycle"],
            _reference["y_true"] + 20,
            color="gray",
            linestyle=":",
            label="True RUL + 20",
        )

        for _model_id, _label in _model_labels.items():
            _data = _oof_by_model[_model_id]

            _data = _data.loc[
                (_data["engine_id"] == _engine_id) & (_data["y_true"].between(0, 30))
            ].sort_values("cycle")

            _ax.plot(
                _data["cycle"],
                _data["y_pred"],
                label=_label,
                linewidth=1.8,
            )

        _ax.set_title(f"Engine {_engine_id}")
        _ax.set_ylabel("RUL (cycles)")
        _ax.grid(alpha=0.25)
        _ax.legend()

    _axes[-1].set_xlabel("Engine cycle")

    _fig.suptitle(
        "Near-Failure Prediction Trajectories",
        fontsize=14,
    )

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(Path, analyze_oof_errors_by_engine, finalists, pd):
    _engine_tables = []

    for _finalist in finalists:
        _model_id = _finalist["id"]

        _oof = pd.read_csv(Path("results/oof") / f"{_model_id}.csv")

        _engine_tables.append(
            analyze_oof_errors_by_engine(
                _oof,
                model_id=_model_id,
                rul_range=(60, 79),
            )
        )

    engine_errors_60_79_df = pd.concat(
        _engine_tables,
        ignore_index=True,
    )

    assert len(engine_errors_60_79_df) == 12 * 80
    assert (engine_errors_60_79_df["n_observations"] == 20).all()

    engine_errors_60_79_df.head()
    return (engine_errors_60_79_df,)


@app.cell
def _(engine_errors_60_79_df, engine_errors_near_failure_df, finalists, pd):
    _phase_results = []

    for _finalist in finalists:
        _model_id = _finalist["id"]

        _near = engine_errors_near_failure_df.loc[
            engine_errors_near_failure_df["Model"] == _model_id
        ].set_index("engine_id")

        _mid = engine_errors_60_79_df.loc[
            engine_errors_60_79_df["Model"] == _model_id
        ].set_index("engine_id")

        assert _near.index.is_unique
        assert _mid.index.is_unique
        assert set(_near.index) == set(_mid.index)

        _affected_near = set(_near.index[_near["n_overestimations_gt_20"] > 0])

        _affected_mid = set(_mid.index[_mid["n_overestimations_gt_20"] > 0])

        _phase_results.append(
            {
                "Model": _model_id,
                "affected_0_30": len(_affected_near),
                "affected_60_79": len(_affected_mid),
                "common_engines": len(_affected_near & _affected_mid),
                "only_0_30": len(_affected_near - _affected_mid),
                "only_60_79": len(_affected_mid - _affected_near),
                "MAE_spearman": _near["MAE"].corr(
                    _mid["MAE"],
                    method="spearman",
                ),
            }
        )

    engine_phase_comparison_df = pd.DataFrame(_phase_results)

    engine_phase_comparison_df.sort_values(
        "MAE_spearman",
        ascending=False,
    ).round(3)
    return


@app.cell
def _(rul_comparison_df):
    _focus_models = [
        "xgboost_pseudohuber",
        "gradientboosting_quantile_0_30",
    ]

    _focus_bands = [
        "0-9",
        "10-19",
        "20-29",
        "60-69",
        "70-79",
    ]

    asymmetry_comparison_df = rul_comparison_df.loc[
        rul_comparison_df["Model"].isin(_focus_models)
        & rul_comparison_df["rul_band"].isin(_focus_bands)
    ].assign(
        over_gt20_pct=lambda _df: (_df["overestimation_gt_20_rate"] * 100),
        under_gt20_pct=lambda _df: (_df["underestimation_gt_20_rate"] * 100),
    )[
        [
            "Model",
            "rul_band",
            "MAE",
            "mean_signed_error",
            "mean_overestimation",
            "mean_underestimation",
            "over_gt20_pct",
            "under_gt20_pct",
        ]
    ]

    asymmetry_comparison_df.round(2)
    return


if __name__ == "__main__":
    app.run()
