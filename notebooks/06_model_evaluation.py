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
    # 06 — Model Evaluation and Error Analysis

    ## Objective

    The previous notebook identified twelve promising machine-learning
    configurations for Remaining Useful Life (RUL) prediction on the
    NASA C-MAPSS FD001 dataset.

    While aggregate metrics provide a first indication of predictive
    performance, they do not fully describe how models behave throughout
    an engine's degradation trajectory.

    This notebook therefore moves beyond global accuracy to investigate
    the conditions under which prediction errors occur, their magnitude,
    and their potential implications for predictive maintenance.

    The analysis addresses four main questions:

    1. How do the twelve finalists compare in overall predictive accuracy?
    2. At which stages of engine degradation do prediction errors increase?
    3. Are large RUL overestimations isolated or persistent across cycles
       and engines?
    4. How do different models balance overestimation and underestimation?

    The study concludes with a justified selection and configuration
    freeze of the Hawkshot V1 reference model.

    ## Evaluation Protocol

    All experiments use out-of-fold (OOF) predictions generated through
    five-fold cross-validation grouped by engine.

    The original FD001 training dataset is divided into 80 development
    engines and 20 held-out internal validation engines. The analyses
    presented here focus exclusively on the development subset.

    To ensure reproducibility and avoid redundant computation, OOF
    predictions are stored in validated checkpoints and reused across
    the different analyses.

    **Methodological limitation:** The finalist configurations were
    previously tuned using the development data. Consequently, these
    OOF results support model comparison and diagnostic analysis but
    should not be interpreted as an independent estimate of final
    generalization performance.

    The official FD001 test set remains untouched.

    ---

    ## 1. Experimental Setup and Model Candidates
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Experimental Setup

    The data preparation, temporal feature engineering, and grouped
    cross-validation protocol established in previous notebooks are
    reconstructed here to ensure reproducibility.

    The evaluation uses 80 development engines, five-fold GroupKFold
    cross-validation, and the previously selected feature configurations.

    No additional feature optimization or hyperparameter tuning is
    performed at this stage.
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
    return df_train_temporal, raw_temporal_cycle_features, temporal_config


@app.cell
def _(mo):
    mo.md(r"""
    ### Temporal Feature Reconstruction

    The temporal feature configuration selected during model
    development is reconstructed without further optimization.

    For each retained sensor, the feature engineering pipeline
    generates rolling means, temporal differences, and slopes
    using the previously selected window sizes.

    These features provide information about recent sensor
    behavior and degradation trends, beyond instantaneous
    measurements.

    The same feature definitions are preserved across the
    evaluation experiments to ensure consistency with
    Notebook 05.
    """)
    return


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
def _(mo):
    mo.md(r"""
    ## 2. Out-of-Fold Predictions and Global Performance

    The twelve finalist configurations selected in Notebook 05 are
    evaluated alongside two reference models: a median DummyRegressor
    and a linear regression baseline.

    Each model is evaluated using out-of-fold (OOF) predictions,
    ensuring that every development observation is predicted by
    a model that was not trained on its engine.

    Predictions are saved in reproducible checkpoints, allowing
    subsequent analyses to reuse them without additional training.

    Global MAE and RMSE provide the initial performance comparison,
    while signed-error metrics help identify potential differences
    in model behavior.
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
    return (comparison_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Global Performance — Key Findings

    **XGBoost Pseudo-Huber achieves the lowest overall MAE
    (8.745 cycles)**, followed by Gradient Boosting Absolute Error
    (9.044 cycles) and XGBoost Quantile (9.100 cycles).

    However, the ranking is not identical across evaluation metrics.
    Gradient Boosting Squared Error achieves a slightly lower RMSE
    (13.946 cycles) than XGBoost Pseudo-Huber (14.003 cycles).

    This illustrates that the configuration minimizing average
    absolute error does not necessarily minimize larger prediction
    errors.

    Although XGBoost Pseudo-Huber emerges as the strongest candidate
    in terms of global MAE, aggregate metrics alone are insufficient
    to characterize reliability throughout engine degradation.

    The following analyses therefore investigate **when errors
    occur, how severe they become, and whether they are concentrated
    on specific engines**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Prediction Errors Across Remaining Useful Life

    Global metrics summarize predictive performance but may conceal
    significant differences across engine degradation stages.

    To investigate these variations, OOF predictions are grouped
    into 10-cycle intervals according to their target RUL.

    For each interval, we examine prediction accuracy and the
    frequency of RUL overestimation across all fourteen models.

    Observations with a capped RUL of 125 are analyzed separately,
    since their actual remaining lifetime may exceed this value.
    They are excluded from the temporal MAE visualization.

    This analysis aims to identify whether prediction difficulties
    are concentrated in particular degradation stages or remain
    consistent throughout the engine lifetime.
    """)
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
def _(mo):
    mo.md(r"""
    ### Key Findings — RUL-Dependent Performance

    Prediction accuracy varies substantially across the remaining
    lifetime of an engine.

    The nonlinear models achieve particularly low MAE values near
    failure, but their errors generally increase at intermediate
    RUL values, with several configurations reaching their highest
    errors around 60–80 remaining cycles.

    Linear regression follows a different pattern, with considerably
    larger errors near failure and a less pronounced dependence on
    the same degradation stages.

    These results suggest that the difficulty of RUL estimation is
    not uniform throughout an engine's lifetime. However, they do
    not establish whether these variations originate from sensor
    behavior, feature representation, or model limitations.

    **An important distinction remains:** a larger prediction error
    does not necessarily imply a more optimistic RUL estimate.
    The direction and magnitude of these errors must therefore be
    examined separately.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Large RUL Overestimations

    In predictive maintenance, overestimating an engine's remaining
    lifetime may delay maintenance decisions, while underestimation
    may lead to unnecessary early interventions.

    To investigate potentially consequential prediction errors,
    we measure the proportion of observations where predicted RUL
    exceeds the target by more than 20 cycles.

    This threshold is used as an exploratory diagnostic criterion,
    not as a validated operational safety limit.

    Unlike the overall overestimation rate, this metric distinguishes
    minor optimistic predictions from substantially larger errors.

    The analysis focuses on near-failure conditions and the
    intermediate RUL intervals previously identified as challenging.
    """)
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
def _(mo):
    mo.md(r"""
    ### Key Findings — Large Overestimations

    Large RUL overestimations are relatively rare near failure
    for the nonlinear finalists, despite differences in their
    overall overestimation rates.

    However, their frequency increases considerably at
    intermediate RUL values, particularly around 60–79 cycles.

    The choice of training objective has a substantial influence
    on this behavior.

    For example, between 60 and 69 remaining cycles:

    - **XGBoost Pseudo-Huber** overestimates RUL by more than
      20 cycles in approximately 31.9% of observations.
    - **Gradient Boosting Quantile 0.30** reduces this proportion
      to approximately 13.9%.

    This suggests that quantile-based objectives can reduce
    large optimistic predictions in specific degradation stages.

    However, a lower overestimation frequency does not
    necessarily indicate better overall performance.
    Such improvements may come at the cost of increased
    underestimation elsewhere in the engine's lifetime.

    Furthermore, aggregate rates cannot reveal whether
    these errors are isolated or repeatedly affect the
    same engines.

    The next analysis therefore investigates prediction
    errors at the individual engine level.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Engine-Level Error Analysis

    RUL-band statistics aggregate predictions across the entire fleet,
    but they cannot reveal whether large errors are isolated or
    concentrated on specific engines.

    We therefore examine prediction errors at the individual engine
    level, focusing initially on the final 30 cycles before failure.

    In addition to overall accuracy, we investigate:

    - The maximum RUL overestimation observed for each engine.
    - The number of predictions exceeding the true RUL by 20 cycles.
    - The longest sequence of consecutive large overestimations.

    This analysis helps distinguish occasional prediction errors
    from persistent optimistic behavior along an engine's trajectory.
    """)
    return


@app.cell
def _(Path, analyze_oof_errors_by_engine, finalists, pd, rul_comparison_df):
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

    _finalist_ids = {_config["id"] for _config in finalists}

    (
        engine_errors_near_failure_df.loc[
            engine_errors_near_failure_df["Model"].isin(_finalist_ids)
        ]
        .sort_values("max_overestimation", ascending=False)
        .head(10)
        .round(3)
    )
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
def _(mo):
    mo.md(r"""
    ### Key Findings — Near-Failure Behavior

    Large overestimations are not distributed uniformly across
    the fleet.

    Among the 80 development engines, XGBoost Pseudo-Huber produces
    at least one overestimation exceeding 20 cycles on 9 engines,
    compared with 6 for Gradient Boosting Quantile 0.30.

    Only four engines are affected by both models, indicating that
    their most difficult cases do not completely overlap.

    The selected trajectories illustrate how models can behave
    differently on the same engine, even when their aggregate
    performance is relatively similar.

    However, counting affected engines alone does not fully
    characterize prediction reliability. The magnitude and
    persistence of these errors must also be considered.

    These findings reinforce the importance of examining
    individual trajectories alongside aggregate evaluation metrics.
    """)
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
def _(mo):
    mo.md(r"""
    ### Are the Same Engines Difficult Throughout Degradation?

    To investigate whether prediction difficulties persist across
    degradation stages, we compare engine-level errors near failure
    and at intermediate RUL values (60–79 cycles).

    The Spearman correlation between engine-level MAE rankings is
    positive but moderate across the finalist models.

    This suggests that engines which are difficult to predict
    in one degradation stage are not necessarily the most
    difficult in another.

    The overlap of engines affected by large overestimations
    provides an additional perspective on this behavior.

    Overall, prediction difficulty appears to depend not only
    on the engine itself, but also on the stage of its
    degradation trajectory.

    **Implication:** Global accuracy and engine-level reliability
    cannot be fully characterized by a single RUL interval.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Overestimation–Underestimation Trade-off

    The previous analyses revealed different error patterns
    between XGBoost Pseudo-Huber and Gradient Boosting Quantile 0.30.

    To better understand this trade-off, we compare the direction
    and magnitude of their prediction errors across selected RUL
    intervals.

    Prediction error is defined as:

    \[
    e = \hat{y} - y
    \]

    Positive errors indicate RUL overestimation, while negative
    errors indicate underestimation.

    Particular attention is given to errors exceeding 20 cycles
    in either direction, both near failure and at intermediate
    remaining lifetimes.
    """)
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


@app.cell
def _(rul_comparison_df):
    _high_rul_bands = [
        "80-89",
        "90-99",
        "100-109",
        "110-119",
        "120-124",
        "125_capped",
    ]

    _models = [
        "xgboost_pseudohuber",
        "gradientboosting_quantile_0_30",
    ]

    asymmetry_high_rul_df = rul_comparison_df.loc[
        rul_comparison_df["Model"].isin(_models)
        & rul_comparison_df["rul_band"].isin(_high_rul_bands)
    ].assign(
        under_gt20_pct=lambda _df: (_df["underestimation_gt_20_rate"] * 100),
        over_gt20_pct=lambda _df: (_df["overestimation_gt_20_rate"] * 100),
    )[
        [
            "Model",
            "rul_band",
            "n_observations",
            "MAE",
            "mean_signed_error",
            "under_gt20_pct",
            "over_gt20_pct",
        ]
    ]

    asymmetry_high_rul_df.round(2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Key Findings — Asymmetric Prediction Behavior

    Gradient Boosting Quantile 0.30 substantially reduces large
    RUL overestimations at intermediate degradation stages.

    Between 60 and 79 remaining cycles, overestimations exceeding
    20 cycles occur in approximately 15% of its predictions,
    compared with 32% for XGBoost Pseudo-Huber.

    However, this improvement comes with a significant trade-off
    at higher RUL values.

    Between 110 and 119 remaining cycles, GB Quantile 0.30
    underestimates the target by more than 20 cycles in
    approximately 34% of observations, compared with 17%
    for XGBoost Pseudo-Huber.

    The quantile model therefore produces more conservative
    predictions in certain regions, but this behavior does
    not consistently translate into better accuracy.

    Observations with a target RUL capped at 125 cycles must
    be interpreted separately: their target does not represent
    their exact physical remaining lifetime.

    **Overall, XGBoost Pseudo-Huber offers a more balanced
    compromise across the evaluated degradation stages.**
    While GB Quantile 0.30 reduces some potentially concerning
    overestimations, its increased underestimation at higher
    RUL values and weaker global MAE limit its suitability
    as the general-purpose V1 reference model.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Final Model Selection and Configuration Freeze

    The previous analyses highlighted substantial differences
    between finalist models in terms of global accuracy, error
    severity, and behavior across degradation stages.

    The final comparison brings together global performance
    and error metrics from two critical RUL intervals:
    near failure (0–29 cycles) and intermediate degradation
    (60–79 cycles).

    Model selection considers these complementary indicators
    rather than relying exclusively on global MAE.
    """)
    return


@app.cell
def _(comparison_df, finalists, rul_comparison_df):
    _finalist_ids = {_finalist["id"] for _finalist in finalists}

    _global = comparison_df.loc[
        comparison_df["Model"].isin(_finalist_ids),
        [
            "Model",
            "MAE",
            "RMSE",
            "overestimation_rate",
            "mean_signed_error",
        ],
    ].copy()

    _near = (
        rul_comparison_df.loc[
            rul_comparison_df["rul_band"].isin(["0-9", "10-19", "20-29"])
        ]
        .groupby("Model")
        .agg(
            near_MAE=("MAE", "mean"),
            near_over_gt20=("overestimation_gt_20_rate", "mean"),
            near_under_gt20=("underestimation_gt_20_rate", "mean"),
        )
    )

    _mid = (
        rul_comparison_df.loc[rul_comparison_df["rul_band"].isin(["60-69", "70-79"])]
        .groupby("Model")
        .agg(
            mid_MAE=("MAE", "mean"),
            mid_over_gt20=("overestimation_gt_20_rate", "mean"),
            mid_under_gt20=("underestimation_gt_20_rate", "mean"),
        )
    )

    finalist_decision_df = (
        _global.merge(_near, on="Model")
        .merge(_mid, on="Model")
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    assert len(finalist_decision_df) == 12
    assert finalist_decision_df["Model"].is_unique

    finalist_decision_df.round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Final Decision — XGBoost Pseudo-Huber

    **XGBoost Pseudo-Huber is selected as the Hawkshot V1
    reference model.**

    It achieves the lowest global OOF MAE (8.745 cycles)
    while maintaining competitive performance near failure
    and a more balanced error profile across degradation stages.

    Although Gradient Boosting Quantile 0.30 reduces large
    overestimations in certain RUL intervals, this advantage
    comes with increased underestimation at higher RUL values
    and weaker overall accuracy.

    The selected configuration therefore represents a
    practical compromise for a general-purpose RUL predictor,
    rather than an optimal solution for every operating condition.

    ### Limitations and Next Steps

    Several limitations must be considered:

    - OOF results may reflect model-selection bias, as the
      configurations were previously tuned on development data.
    - The RUL target is capped at 125 cycles, limiting the
      interpretation of early-life predictions.
    - FD001 is a simulated dataset and does not establish
      performance under real industrial operating conditions.
    - No application-specific maintenance cost or safety
      threshold has been validated.

    The selected model configuration is frozen before any
    independent evaluation.

    **The next step is to train the frozen configuration
    and evaluate its generalization performance on the
    official FD001 test set, without further tuning based
    on test results.**
    """)
    return


@app.cell
def _(Path, finalists, json, oof_feature_sets, temporal_config):
    _selected = next(
        _config for _config in finalists if _config["id"] == "xgboost_pseudohuber"
    )

    assert _selected["feature_set"] == "no_means"

    _feature_columns = oof_feature_sets[_selected["feature_set"]].columns.tolist()

    assert len(_feature_columns) == 71

    _frozen_config = {
        "schema_version": 1,
        "dataset": "NASA C-MAPSS FD001",
        "id": _selected["id"],
        "model": _selected["model"],
        "profile": _selected["profile"],
        "params": _selected["params"],
        "feature_set": _selected["feature_set"],
        "temporal_config": temporal_config,
        "rul_cap": 125,
        "feature_columns": _feature_columns,
    }

    _config_path = Path("configs/final_model_v1.json")
    _config_path.parent.mkdir(parents=True, exist_ok=True)

    if _config_path.exists():
        _existing = json.loads(_config_path.read_text(encoding="utf-8"))

        if _existing != _frozen_config:
            raise ValueError(
                "Frozen V1 configuration differs. "
                "Do not overwrite it without an explicit decision."
            )

        print("V1 configuration already frozen.")

    else:
        _config_path.write_text(
            json.dumps(_frozen_config, indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"V1 configuration frozen: {_config_path}")

    print("Selected model:", _selected["id"])
    print("Features:", len(_feature_columns))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Summary and Next Steps

    This notebook evaluated twelve finalist configurations through
    out-of-fold predictions, combining global performance metrics
    with RUL-dependent and engine-level error analysis.

    XGBoost Pseudo-Huber was selected as the Hawkshot V1 reference
    model, achieving the lowest global OOF MAE (8.745 cycles)
    while offering a reasonable compromise between prediction
    accuracy and asymmetric errors.

    The analysis also highlighted important limitations, including
    increased errors at intermediate RUL values and occasional
    large overestimations near failure.

    ### V1 Scope and Future Improvements

    The selected model should not be considered fully optimized.

    Additional hyperparameter tuning, feature engineering, and
    RUL-specific modeling strategies could potentially improve
    predictive performance.

    However, further optimization is deliberately deferred.
    The priority is now to complete a reproducible end-to-end
    ML system rather than pursue marginal improvements to
    cross-validation metrics.

    These potential improvements remain candidates for a
    future V2 iteration, subject to time and project priorities.

    ### Next Steps

    With the V1 configuration frozen, the project moves toward:

    1. **Independent evaluation:** Train the selected configuration
       on the official FD001 training data and assess its
       generalization on the untouched official test set.

    2. **Reproducible ML pipeline:** Implement repeatable training,
       model serialization, and batch inference.

    3. **ML Engineering:** Develop an inference interface,
       automated tests, CI, containerization, and basic monitoring.

    The objective is to transform the experimental model into
    a reproducible and demonstrable predictive maintenance system.

    **Hawkshot V1 prioritizes completing the ML lifecycle;
    model refinement remains an opportunity for V2.**
    """)
    return


if __name__ == "__main__":
    app.run()
