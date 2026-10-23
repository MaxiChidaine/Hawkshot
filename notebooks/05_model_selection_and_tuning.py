import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Model selection and hyperparameter tuning

    This notebook develops and compares tabular machine-learning models for Remaining Useful Life (RUL) prediction on the NASA C-MAPSS FD001 dataset.

    The objective is not limited to minimizing average prediction error. In a predictive-maintenance context, overestimating the remaining lifetime of an engine may be more critical than underestimating it, since an optimistic prediction can delay maintenance beyond a safe operating point.

    Model selection therefore considers two complementary aspects:

    - **predictive accuracy**, measured using MAE and RMSE;
    - **operational risk**, characterized through the frequency, direction and magnitude of RUL overestimation.

    The modelling process follows a staged strategy. Candidate model families are first screened using grouped cross-validation. Feature ablation is then performed separately for each retained family before progressively refining structural parameters, regularization, loss functions, and convergence settings.

    The searches in this notebook use the 80 model-development engines and grouped cross-validation. The 20-engine development-validation subset was already consulted during model and hyperparameter comparisons in notebook 04, so it is not an independent final holdout. The official FD001 test is reserved for evaluation after model selection is frozen.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import winsound
    import time

    from pathlib import Path
    from sklearn.model_selection import ParameterGrid
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.metrics import (
        mean_absolute_error,
        root_mean_squared_error,
        make_scorer,
    )
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
    from hawkshot.modeling.evaluation import evaluate_grouped_cv
    from hawkshot.modeling.metrics import (
        mean_overestimation,
        mean_signed_error_scorer,
        overestimation_rate,
    )
    from hawkshot.modeling.search_results import format_grid_search_results
    from hawkshot.experiments.runner import run_or_reuse_grid_search
    from hawkshot.experiments.provenance import materialize_cv_splits
    from time import perf_counter
    from datetime import datetime
    from itertools import product
    from xgboost import XGBRegressor

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
        ParameterGrid,
        Path,
        Pipeline,
        RandomForestRegressor,
        SVR,
        StandardScaler,
        XGBRegressor,
        add_temporal_features,
        available_sensors,
        datetime,
        df_filtered,
        evaluate_grouped_cv,
        format_grid_search_results,
        make_scorer,
        mean_overestimation,
        mean_signed_error_scorer,
        mo,
        np,
        overestimation_rate,
        pd,
        product,
        time,
        train_test_split,
        winsound,
        materialize_cv_splits,
        run_or_reuse_grid_search,
        perf_counter,
        mean_absolute_error,
        root_mean_squared_error,
        clone,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Experimental protocol

    The preprocessing and temporal feature engineering established in the previous notebooks are reproduced here so that the modelling workflow remains executable from a consistent dataset representation.

    The prediction target is the capped Remaining Useful Life(`rul_capped`), with a maximum value of 125 cycles. Capping the early-life portion of the degradation trajectory prevents the model from being required to distinguish very large RUL values when little degradation information is yet observable.

    Model development uses an engine-level split: 80 engines for grouped cross-validation and 20 engines for complementary development validation. No engine appears in both subsets. The 20-engine subset is not an independent final test because its results already informed earlier decisions.
    """)
    return


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
def _(mo):
    mo.md(r"""
    Twenty percent of the engines form a development-validation subset. Although the searches in this notebook use the remaining 80 engines, the 20-engine subset was already consulted for model and hyperparameter comparisons in notebook 04. It must therefore not be described as an untouched independent holdout.

    The remaining engines constitute the model-development dataset. Internal comparisons are performed using grouped cross-validation, which provides repeated estimates of generalization performance while preserving the independence of complete engine-trajectories.

    The 20-engine subset provides additional development diagnostics, not an unbiased final performance estimate. The official FD001 test is reserved for evaluation only after model selection and reporting choices are fixed.
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
def _(mo):
    mo.md(r"""
    ### Grouped cross-validation

    A standard row-level cross-validation strategy would introduce leakage. Different cycles from the same engine could be assigned to both the training and validation folds.

    Five-fold `GroupKFold` cross-validation is therefore used with `engine_id` as the grouping variable. each engine belongs exclusively to one side of a fold, so performance is always measured on complete trajectories from engines unseen during the corresponding model fit.

    The check below verifies that the intersection between training and validation engine identifiers is empty for every fold.
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
    return X_cv_full, cv, groups_cv, y_cv


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Evaluation metrics

    Model quality is evaluated from both statistical and an operational perspective.

    **Mean Absolute Error (MAE)** is used as the primary accuracy metric because it directly represents the average absolute RUL prediction error in cycles.

    **Root Mean Squared Error (RMSE)** complements MAE by assigning greater importance to large individual errors.

    Prediction error is defined as :

    \[
    e = \hat{y} - y
    \]

    where positive errors correspond to **RUL overestimation** and negative errors correspond to underestimation.

    Three additional metrics characterize the asymmetric operational risk:

    - **overestimation rate**: fraction of predictions for which the predicted RUL exceeds the true RUL.
    - **mean signed error**: average prediction error, used to identify an overall optimistic or conservative bias.
    - **mean overestimation**: average magnitude of the error when an overestimation actually occurs.

    Lower MAE and RMSE remain desirable, but models are not selected from a single accuracy metric alone. Particular attention is given to configurations that achieve useful reduction in optimistic RUL predictions without excessive degradation of overall predictive performance.
    """)
    return


@app.cell
def _(
    make_scorer,
    mean_overestimation,
    mean_signed_error_scorer,
    overestimation_rate,
):
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
def _(winsound):
    def notify_done():
        winsound.MessageBeep()

    return (notify_done,)


@app.cell
def _(GridSearchCV, scoring):
    def run_grouped_grid_search(model, param_grid, X, y, groups, cv):
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            refit="MAE",
            cv=cv,
            n_jobs=-1,
        )

        grid_search.fit(X, y, groups=groups)

        return grid_search

    return (run_grouped_grid_search,)


@app.cell
def _(Path, format_grid_search_results):
    def process_grid_search_results(grid_search, output_path):
        """Format and save grid search results (legacy workflow)."""

        results_df = format_grid_search_results(grid_search)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(output_path, index=False)

        return results_df

    return (process_grid_search_results,)


@app.cell
def _(Path, pd):
    def concat_csv_folder(
        folder_path: str | Path,
        output_path: str | Path,
    ) -> pd.DataFrame:
        folder_path = Path(folder_path)
        output_path = Path(output_path)

        print("folder_path reçu :", folder_path)
        print("folder_path absolu :", folder_path.resolve())
        print("exists :", folder_path.exists())

        csv_files = sorted(folder_path.glob("*.csv"))

        print("Nombre de CSV trouvés :", len(csv_files))
        print(csv_files[:3])

        if not csv_files:
            raise FileNotFoundError(
                f"Aucun fichier CSV trouvé dans {folder_path.resolve()}"
            )

        dataframes = [
            pd.read_csv(csv_file)
            for csv_file in csv_files
            if csv_file.resolve() != output_path.resolve()
        ]

        combined_df = pd.concat(
            dataframes,
            ignore_index=True,
        )

        combined_df.to_csv(
            output_path,
            index=False,
        )

        return combined_df

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Initial model screening

    Before performing model-specific tuning, several regression families are evaluated using the complete temporal feature set and the common grouped cross-validation protocol.

    The first comparison is deliberately coarse. Its purpose is not to select final models or hyperparameters, but to establish reference performance and identify model families worth investigating further.

    The screening includes three tree-based ensemble methods : Extra Trees, Random Forest and Gradient Boosting, together with RBF Support Vector Regressor as a non-tree baseline. The configurations used at this stage are therefore considered provisional.
    """)
    return


@app.cell
def _(
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

    et_screening_results = {
        "model": "Extra Trees",
        **evaluate_grouped_cv(
            model=model_et,
            X=X_cv_full,
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        ),
    }

    et_screening_results
    return (et_screening_results,)


@app.cell
def _(
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

    rf_screening_results = {
        "model": "Random Forest",
        **evaluate_grouped_cv(
            model=model_rf,
            X=X_cv_full,
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        ),
    }

    notify_done()

    rf_screening_results
    return (rf_screening_results,)


@app.cell
def _(
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

    gb_screening_results = {
        "model": "Gradient Boosting",
        **evaluate_grouped_cv(
            model=model_gb,
            X=X_cv_full,
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        ),
    }

    notify_done()

    gb_screening_results
    return (gb_screening_results,)


@app.cell
def _(
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
    model_svr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svr",
                SVR(
                    kernel="rbf",
                    C=10,
                    epsilon=0.5,
                    gamma="scale",
                ),
            ),
        ]
    )

    svr_screening_results = {
        "model": "SVR",
        **evaluate_grouped_cv(
            model=model_svr,
            X=X_cv_full,
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        ),
    }

    notify_done()

    svr_screening_results
    return (svr_screening_results,)


@app.cell
def _(
    et_screening_results,
    gb_screening_results,
    pd,
    rf_screening_results,
    svr_screening_results,
):
    screening_results_df = pd.DataFrame(
        [
            et_screening_results,
            rf_screening_results,
            gb_screening_results,
            svr_screening_results,
        ]
    )

    screening_columns = [
        "model",
        "mean_MAE",
        "std_MAE",
        "mean_RMSE",
        "std_RMSE",
        "overestimation_rate",
        "mean_signed_error",
        "mean_overestimation",
        "mean_fit_time",
        "mean_predict_time",
    ]

    screening_results_df.sort_values("mean_MAE")[screening_columns].round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The screening confirms that tree-based ensembles provide sufficiently competitive performance to justify a more detailed investigation.

    The following sections therefore focus on Gradient Boosting, Extra Trees, Random Forest and later XGBoost. SVR is retained as a reference baseline but is not subjected to the same extensive tuning procedure.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Feature ablation strategy

    Temporal feature engineering introduced several complementary descriptions of sensor evolution. Rolling means, recent deltas, and local slopes. Although each transformation may contain useful information, retaining all of them is not necessarily optimal for every model family.

    A common set of feature ablations is therefore defined below. Starting from the complete representation, individual temporal feature groups or combination of them are removed to measure their contribution.

    Feature selection is performed separately for each model family rather than imposing a single feature set on all models. This allows each estimator to retain the representation that works best with its own learning behaviour.
    """)
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
    ## 5. Gradient Boosting

    Gradient Boosting is investigated first among the retained ensemble methods. Unlike Random Forest and Extra Trees, it builds trees sequentially, with each new estimators correcting errors made by the previous ones.

    The search is performed progressively, a coarse configuration is first used to identify a suitable model scale, then the feature representation and loss function are examined before refining the most promising configurations.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Coarse parameter search

    The first search explores the interaction between model complexity and boosting convergence.

    Learning rate and number of estimators are evaluated as paired schedules. Smaller learning rates are associated with more boosting stages, while tree depth and minimum leaf size control the complexity of each individual tree.

    This search is intentionally broad and provides a reference configuration for subsequent feature ablation rather than a final model.
    """)
    return


@app.cell(disabled=True)
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


@app.cell(disabled=True)
def _(GradientBoostingRegressor, grid_gb_coarse):
    gb_coarse_params = grid_gb_coarse.best_params_

    gb_coarse_tuned = GradientBoostingRegressor(
        **gb_coarse_params,
        random_state=42,
    )
    return gb_coarse_params, gb_coarse_tuned


@app.cell
def _(mo):
    mo.md(r"""
    ### Feature ablation

    The coarse configuration is now held fixed while the feature subsets defined earlier are compared.

    Isolating feature selection from the main hyperparameter search reduces the search space and helps determine which temporal transformations are actually useful to Gradient Boosting.
    """)
    return


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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


@app.cell
def _(mo):
    mo.md(r"""
    Removing the rolling-mean features gives the most useful representation for Gradient Boosting. The remaining experiments therefore use the `no_means`feature set, retaining raw sensor values, cycle information, deltas, and slopes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Loss and model refinement

    With the feature representation fixed, the next stage investigates how the training objective and stochastic sampling affect both prediction accuracy and overestimation risk.

    Several loss functions are considered because they respond differently to prediction errors. Subsampling and feature sampling are evaluated at the same time to determine whether additional stochastic regularization improves generalization.

    Candidate configurations are compared using the complete set of accuracy and risk metrics rather than MAE alone.
    """)
    return


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


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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


@app.cell(disabled=True)
def _(gb_tuning_a_results_df):
    best_mae = gb_tuning_a_results_df["mean_MAE"].min()

    gb_tuning_a_candidates = gb_tuning_a_results_df[
        gb_tuning_a_results_df["mean_MAE"] <= best_mae * 1.05
    ].sort_values(["overestimation_rate", "mean_MAE"])

    gb_tuning_a_candidates
    return


@app.cell
def _(mo):
    mo.md(r"""
    The initial search reveals several distinct trade-offs rather than a single dominant configuration. Representative profiles are therefore retained for more focused tuning. An accuracy-oriented absolute-error model, Huber and squared-error alternatives, and two increasingly conservative quantile models.

    The following searches refine these profiles around their most promising regions while preserving their different accuracy-risk behaviours.
    """)
    return


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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
def _(mo):
    mo.md(r"""
    The search is then extended toward lower learning rates, deeper trees, and larger leaf sizes where the strongest configurations were considered.
    """)
    return


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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


@app.cell(disabled=True)
def _(
    GradientBoostingRegressor,
    GridSearchCV,
    ParameterGrid,
    Path,
    X_cv_gb,
    cv,
    groups_cv,
    notify_done,
    param_grids_gb_tuning_b2,
    pd,
    scoring,
    y_cv,
):
    gb_tuning_b2_results = []

    checkpoint_dir_gb_b2 = Path("results/gb_tuning_b2")
    checkpoint_dir_gb_b2.mkdir(
        parents=True,
        exist_ok=True,
    )

    for _profile_name, _param_grid in param_grids_gb_tuning_b2.items():
        _checkpoint_path = checkpoint_dir_gb_b2 / f"{_profile_name}.csv"

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


@app.cell(disabled=True)
def _(gb_tuning_b2_results):
    gb_tuning_b2_results
    return


@app.cell
def _(mo):
    mo.md(r"""
    A final local refinement is performed around the best region of each retained profile. At this stage the objective is no longer broad exploration, but confirming the local optimum before freezing the Gradient Boosting candidates
    """)
    return


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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
def _(mo):
    mo.md(r"""
    ### Gradient Boosting finalists

    The final search confirms that no single Gradient Boosting configuration dominates every evaluation criterion.

    The five profiles are therefore retained as distinct candidates. They span a progression from accuracy-oriented predictions to increasingly conservative RUL estimates, allowing this trade-off to remain explicit until the final model comparison.
    """)
    return


@app.cell
def _(pd):
    gb_finalists = [
        {
            "Model": "GradientBoosting",
            "Profile": "absolute_error",
            "MAE": 9.043725,
            "RMSE": 13.975010,
            "overestimation_rate": 0.448684,
            "mean_signed_error": 0.830330,
            "mean_overestimation": 10.990880,
            "params": {
                "loss": "absolute_error",
                "learning_rate": 0.015,
                "n_estimators": 800,
                "max_depth": 10,
                "min_samples_leaf": 30,
                "subsample": 0.6,
                "max_features": 1.0,
                "random_state": 42,
            },
        },
        {
            "Model": "GradientBoosting",
            "Profile": "huber_0.80",
            "MAE": 9.173041,
            "RMSE": 14.004847,
            "overestimation_rate": 0.448336,
            "mean_signed_error": 0.472971,
            "mean_overestimation": 10.732692,
            "params": {
                "loss": "huber",
                "alpha": 0.80,
                "learning_rate": 0.0075,
                "n_estimators": 1600,
                "max_depth": 10,
                "min_samples_leaf": 25,
                "subsample": 0.6,
                "max_features": 1.0,
                "random_state": 42,
            },
        },
        {
            "Model": "GradientBoosting",
            "Profile": "squared_error",
            "MAE": 9.322893,
            "RMSE": 13.936398,
            "overestimation_rate": 0.441580,
            "mean_signed_error": 0.102021,
            "mean_overestimation": 10.649951,
            "params": {
                "loss": "squared_error",
                "learning_rate": 0.010,
                "n_estimators": 1200,
                "max_depth": 10,
                "min_samples_leaf": 20,
                "subsample": 0.6,
                "max_features": 0.75,
                "random_state": 42,
            },
        },
        {
            "Model": "GradientBoosting",
            "Profile": "quantile_0.40",
            "MAE": 9.462445,
            "RMSE": 14.400706,
            "overestimation_rate": 0.399240,
            "mean_signed_error": -1.860276,
            "mean_overestimation": 9.503805,
            "params": {
                "loss": "quantile",
                "alpha": 0.40,
                "learning_rate": 0.0075,
                "n_estimators": 1600,
                "max_depth": 10,
                "min_samples_leaf": 30,
                "subsample": 0.6,
                "max_features": 0.75,
                "random_state": 42,
            },
        },
        {
            "Model": "GradientBoosting",
            "Profile": "quantile_0.30",
            "MAE": 10.447701,
            "RMSE": 15.647978,
            "overestimation_rate": 0.349840,
            "mean_signed_error": -4.378928,
            "mean_overestimation": 8.669946,
            "params": {
                "loss": "quantile",
                "alpha": 0.30,
                "learning_rate": 0.020,
                "n_estimators": 600,
                "max_depth": 11,
                "min_samples_leaf": 20,
                "subsample": 1.0,
                "max_features": 0.75,
                "random_state": 42,
            },
        },
    ]

    gb_finalists_df = pd.DataFrame(gb_finalists)

    gb_finalists_df
    return (gb_finalists,)


@app.cell
def _(mo):
    mo.md(r"""
    The absolute-error profile achieves the lowest MAE, while squared error provides the lowest RMSE and an almost neutral signed bias.

    The quantile profiles progressively reduce optimistic RUL predictions. In particular, in the 0.40 quantile lowers the overestimation rate below 40%, while the 0.30 profile pushes this behaviour further at a substantial cost in overall prediction error.

    These configurations are therefore kept as complementary finalists rather than reduced to a single Gradient Boosting model.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Extra Trees

    Extra Trees is investigated as a highly randomized tree ensemble. Compared with Gradient Boosting, its trees are trained independently and introduce additional randomness when selecting split thresholds.

    The tuning strategy therefore focuses on controlling tree complexity, feature sampling, and the choice of splitting criterion before examining whether additional trees provide meaningful gains.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    A preliminary search is first used to obtain a reasonable structural configuration for the feature-ablation stage. The objective at this point is only to avoid comparing feature sets with an obviously unsuitable model.
    """)
    return


@app.cell(disabled=True)
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
        n_estimators=100,
        random_state=42,
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
    return (grid_et_coarse,)


@app.cell(disabled=True)
def _(ExtraTreesRegressor, grid_et_coarse, pd):
    et_reference_params = grid_et_coarse.best_params_

    et_reference = ExtraTreesRegressor(
        **et_reference_params,
        random_state=42,
    )

    pd.DataFrame(grid_et_coarse.cv_results_).to_csv(
        "results/et_pre_tuning_results.csv",
        index=False,
    )
    return (et_reference,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Feature ablation

    The common feature subsets defined earlier are evaluated using the preliminary Extra Trees configuration.

    This isolates the effect of the input representation before the main hyperparameter search.
    """)
    return


@app.cell(disabled=True)
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
def _(mo):
    mo.md(r"""
    Extra Trees performs best without rolling means and delta features. The retained representation therefore contains the current sensor values, cycle information, and temporal slopes.

    All subsequent Extra Trees searches use this reduced feature set.
    """)
    return


@app.cell
def _(df_train_temporal, feature_sets):
    final_feature_set_et = feature_sets["no_means_no_deltas"]

    X_cv_et = df_train_temporal[final_feature_set_et]
    return (X_cv_et,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Structural tuning

    With the feature representation fixed, the first detailed search explores the main complexity parameters of the ensemble: tree depth, minimum leaf size, and the fraction of features considered at each split.

    The number of trees is kept deliberately modest during this stage so that structural effects can be studied without unnecessary computational cost.
    """)
    return


@app.cell(disabled=True)
def _(
    ExtraTreesRegressor,
    GridSearchCV,
    X_cv_et,
    cv,
    groups_cv,
    notify_done,
    process_grid_search_results,
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

    grid_et_a_results_df = process_grid_search_results(
        grid_et_a, "results/ExtraTrees/et_tuning_a"
    )
    return grid_et_a_results_df


@app.cell
def _(mo):
    mo.md(r"""
    ### Criterion and sampling strategy

    The second stage examines whether the splitting criterion, bootstrap sampling, and ensemble size change the accuracy-risk trade-off.

    Three criteria are compared together with bootstrap and non-bootstrap variants. The search is divided into independent chunks so that the large experiment can be checkpointed and resumed safely.
    """)
    return


@app.cell(disabled=True)
def _(Path):
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
                    "chunk_id": (f"n{_n_estimators}__{_criterion}__bootstrap_false"),
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
                            "__bootstrap_true"
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
    return checkpoint_dir_et_b, et_b_chunks


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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

        assert len(_results) == 96

        _results["mean_MAE"] = -_results["mean_test_MAE"]

        _results["mean_RMSE"] = -_results["mean_test_RMSE"]

        _results["overestimation_rate"] = -_results["mean_test_overestimation_rate"]

        _results["mean_overestimation"] = -_results["mean_test_mean_overestimation"]

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


@app.cell(disabled=True)
def _(checkpoint_dir_et_b, et_b_results, pd):
    et_tuning_b_results_df = pd.concat(
        et_b_results,
        ignore_index=True,
    )

    et_tuning_b_results_df.to_csv(
        checkpoint_dir_et_b / "et_tuning_b_results.csv",
        index=False,
    )

    et_tuning_b_results_df.shape
    return


@app.cell
def _(mo):
    mo.md(r"""
    The results reveal two useful behaviours. Absolute error provides the strongest MAE-oriented configurations, while the Poisson criterion offers a more balanced alternative with lower optimistic bias.

    bootstrap sampling does not provide sufficient benefit to justify its use for the retained Extra trees candidates, so the final refinement focuses on non-bootstrap configurations.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Boundary refinement

    The strongest configurations lie close to the edges of the previous search ranges. A focused boundary search is therefore performed for the absolute-error and Poisson profiles before fixing the final candidates.

    This final step checks that the selected depths and leaf sizes represent stable local regions rather than arbitrary grid boundaries.
    """)
    return


@app.cell(disabled=True)
def _(
    ExtraTreesRegressor,
    X_cv_et,
    cv,
    groups_cv,
    notify_done,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    param_grid_et_boundary = [
        {
            "criterion": ["absolute_error"],
            "n_estimators": [500],
            "bootstrap": [False],
            "max_features": [1.0],
            "min_samples_split": [2],
            "max_depth": [16, 18, 20, 22],
            "min_samples_leaf": [4, 6, 8],
        },
        {
            "criterion": ["poisson"],
            "n_estimators": [500],
            "bootstrap": [False],
            "max_features": [1.0],
            "min_samples_split": [2],
            "max_depth": [24, 26, 30, None],
            "min_samples_leaf": [1, 2, 3],
        },
    ]

    grid_et_boundary = run_grouped_grid_search(
        model=ExtraTreesRegressor(
            random_state=42,
            n_jobs=1,
        ),
        param_grid=param_grid_et_boundary,
        X=X_cv_et,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    grid_et_boundary_results_df = process_grid_search_results(
        grid_et_boundary, "results/et_boundary_results.csv"
    )
    grid_et_boundary_results_df

    notify_done()
    return


@app.cell(disabled=True)
def _(
    ExtraTreesRegressor,
    X_cv_et,
    cv,
    groups_cv,
    notify_done,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    param_grid_et_absolute_boundary = {
        "criterion": ["absolute_error"],
        "n_estimators": [500],
        "bootstrap": [False],
        "max_features": [1.0],
        "min_samples_split": [2],
        "max_depth": [10, 12, 14, 16],
        "min_samples_leaf": [4, 6, 8],
    }

    grid_et_absolute_boundary = run_grouped_grid_search(
        model=ExtraTreesRegressor(
            random_state=42,
            n_jobs=1,
        ),
        param_grid=param_grid_et_absolute_boundary,
        X=X_cv_et,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    grid_et_absolute_boundary_results_df = process_grid_search_results(
        grid_et_absolute_boundary, "results/et_absolute_boundary_results.csv"
    )
    grid_et_absolute_boundary_results_df

    notify_done()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Extra Trees finalists

    Two complementary Extra Trees configurations are retained.

    The absolute-error profile prioritizes predictive accuracy, while the Poisson profile sacrifices some MAE in exchange for a lower RMSE and a substantially more neutral signed bias.

    Both are therefore preserved for the final comparison rather than forcing a single winner at this stage.
    """)
    return


@app.cell
def _(pd):
    et_finalists = [
        {
            "Model": "ExtraTrees",
            "Profile": "absolute_error",
            "MAE": 9.371,
            "RMSE": 14.495,
            "overestimation_rate": 0.413,
            "mean_signed_error": 1.524,
            "mean_overestimation": 13.159,
            "params": {
                "n_estimators": 500,
                "criterion": "absolute_error",
                "max_depth": 16,
                "min_samples_leaf": 6,
                "min_samples_split": 2,
                "max_features": 1.0,
                "bootstrap": False,
                "random_state": 42,
            },
        },
        {
            "Model": "ExtraTrees",
            "Profile": "poisson",
            "MAE": 9.880,
            "RMSE": 14.217,
            "overestimation_rate": 0.397,
            "mean_signed_error": -0.017,
            "mean_overestimation": 12.404,
            "params": {
                "n_estimators": 1000,
                "criterion": "poisson",
                "max_depth": 26,
                "min_samples_leaf": 2,
                "min_samples_split": 2,
                "max_features": 1.0,
                "bootstrap": False,
                "random_state": 42,
            },
        },
    ]

    et_finalists_df = pd.DataFrame(et_finalists)

    et_finalists_df
    return (et_finalists,)


@app.cell
def _(mo):
    mo.md(r"""
    The absolute-error configuration reaches the lowest Extra Trees MAE, whereas the Poisson alternative produces a lower RMSE and an almost unbiased mean signed error.

    These profiles therefore represent different positions on the same accuracy-risk trade-off and remain candidates for later finalist comparisons.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Random Forest

    Random Forest is evaluated as a second bagging-based ensemble method. Unlike Extra Trees, split thresholds are optimized within each tree, while diversity is mainly introduced through bootstrap sampling and feature subsampling.

    The analysis follows the same progressive strategy used for the other model families, a coarse search first identifies a reasonable reference configuration, which is then used for feature ablation before the main hyperparameter refinement.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Coarse parameter search

    A preliminary search explores the main structural parameters of the forest tree depth, minimum leaf size and feature sampling.
    """)
    return


@app.cell
def _(
    RandomForestRegressor,
    X_cv_full,
    cv,
    groups_cv,
    notify_done,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    param_grid_rf_coarse = {
        "max_depth": [12, 16, 20, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.5, 0.75, 1.0],
    }

    grid_rf_coarse = run_grouped_grid_search(
        model=RandomForestRegressor(
            random_state=42,
            n_jobs=1,
        ),
        param_grid=param_grid_rf_coarse,
        X=X_cv_full,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    grid_rf_coarse_result_df = process_grid_search_results(
        grid_rf_coarse, "results/rf_coarse_results.csv"
    )
    grid_rf_coarse_result_df

    notify_done()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Feature ablation

    The retained coarse configuration is used to compare the common feature subsets defined earlier.
    """)
    return


@app.cell(disabled=True)
def _(RandomForestRegressor):
    rf_ablation_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        max_features=0.5,
        random_state=42,
        n_jobs=-1,
    )
    return (rf_ablation_model,)


@app.cell(disabled=True)
def _(
    cv,
    df_train_temporal,
    evaluate_grouped_cv,
    feature_sets,
    groups_cv,
    notify_done,
    rf_ablation_model,
    y_cv,
):
    feature_set_results_rf = []

    for _name, _features in feature_sets.items():
        _metrics = evaluate_grouped_cv(
            model=rf_ablation_model,
            X=df_train_temporal[_features],
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        )

        feature_set_results_rf.append(
            {
                "feature_set": _name,
                "n_features": len(_features),
                **_metrics,
            }
        )

    notify_done()
    return (feature_set_results_rf,)


@app.cell(disabled=True)
def _(feature_set_results_rf, pd):
    feature_set_results_rf_df = pd.DataFrame(feature_set_results_rf)

    feature_set_results_rf_df.round(3)
    return


@app.cell
def _(df_train_temporal, feature_sets):
    final_feature_set_rf = feature_sets["no_means_no_deltas"]

    X_cv_rf = df_train_temporal[final_feature_set_rf]
    return (X_cv_rf,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Structural tuning

    With the feature set fixed, the main structural parameters are refined such as tree depth, minimum leaf size, minimum split size, and feature sampling.
    """)
    return


@app.cell(disabled=True)
def _(
    RandomForestRegressor,
    X_cv_rf,
    cv,
    groups_cv,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    param_grid_rf_a = {
        "max_depth": [10, 14, 18, 22, None],
        "min_samples_leaf": [1, 2, 4, 6],
        "min_samples_split": [2, 4, 8],
        "max_features": [0.25, 0.4, 0.55, 0.7],
    }

    grid_rf_a = run_grouped_grid_search(
        model=RandomForestRegressor(
            random_state=42,
            n_jobs=1,
            n_estimators=300,
        ),
        param_grid=param_grid_rf_a,
        X=X_cv_rf,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    grid_rf_a_results_df = process_grid_search_results(
        grid_rf_a, "results/RandomForest/rf_tuning_a_results"
    )
    grid_rf_a_results_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    The strongest configurations favour relatively high feature sampling. A focused search is therefore performed around this region before fixing the forest structure.
    """)
    return


@app.cell(disabled=True)
def _(
    RandomForestRegressor,
    X_cv_rf,
    cv,
    groups_cv,
    notify_done,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    param_grid_rf_a_boundary = {
        "max_depth": [
            14,
            18,
            22,
            None,
        ],
        "min_samples_leaf": [
            2,
            4,
            6,
        ],
        "max_features": [
            0.70,
            0.85,
            1.0,
        ],
    }

    grid_rf_a_boundary = run_grouped_grid_search(
        model=RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=1,
            min_samples_leaf=2,
        ),
        param_grid=param_grid_rf_a_boundary,
        X=X_cv_rf,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    grid_rf_a_boundary_results_df = process_grid_search_results(
        grid_rf_a_boundary, "results/RandomForest/rf_tuning_a_boundary_result"
    )
    grid_rf_a_boundary_results_df
    notify_done()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Criterion and bootstrap strategy

    The structural region is now combined with alternative splitting criteria and sampling strategies.

    Squared error, absolute error, and Poisson criteria are compared with and without bootstrap sampling. When bootstrap is enabled, different sampling fractions are also evaluated.
    """)
    return


@app.cell(disabled=True)
def _(
    RandomForestRegressor,
    X_cv_rf,
    cv,
    groups_cv,
    notify_done,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    param_grid_rf_b = [
        {
            "criterion": [
                "squared_error",
                "absolute_error",
                "poisson",
            ],
            "max_depth": [14, 18, None],
            "min_samples_leaf": [2, 4],
            "max_features": [0.70, 0.85, 1.0],
            "bootstrap": [False],
        },
        {
            "criterion": [
                "squared_error",
                "absolute_error",
                "poisson",
            ],
            "max_depth": [14, 18, None],
            "min_samples_leaf": [2, 4],
            "max_features": [0.70, 0.85, 1.0],
            "bootstrap": [True],
            "max_samples": [0.70, 0.85, 1.0],
        },
    ]

    grid_rf_b = run_grouped_grid_search(
        model=RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        param_grid=param_grid_rf_b,
        X=X_cv_rf,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    grid_rf_b_results_df = process_grid_search_results(
        grid_rf_b, "results/RandomForest/rf_tuning_b_results"
    )
    grid_rf_b_results_df

    notify_done()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The absolute-error and Poisson profiles are retained for a final
    ensemble-size comparison.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Ensemble-size refinement

    The final Random Forest experiment varies only the number of trees for the two retained profiles, with all other parameters fixed.
    """)
    return


@app.cell(disabled=True)
def _(
    RandomForestRegressor,
    X_cv_rf,
    cv,
    groups_cv,
    notify_done,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    rf_finalist_params = [
        {
            "criterion": ["absolute_error"],
            "max_depth": [14],
            "min_samples_leaf": [4],
            "min_samples_split": [2],
            "max_features": [0.70],
            "bootstrap": [True],
            "max_samples": [1.0],
            "n_estimators": [300, 500],
        },
        {
            "criterion": ["poisson"],
            "max_depth": [18],
            "min_samples_leaf": [2],
            "min_samples_split": [2],
            "max_features": [0.70],
            "bootstrap": [True],
            "max_samples": [1.0],
            "n_estimators": [300, 500, 1000],
        },
    ]

    grid_rf_estimators_tuning = run_grouped_grid_search(
        model=RandomForestRegressor(
            random_state=42,
            n_jobs=1,
        ),
        param_grid=rf_finalist_params,
        X=X_cv_rf,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    grid_rf_estimators_tuning_results_df = process_grid_search_results(
        grid_rf_estimators_tuning, "results/RandomForest/rf_estimators_tuning"
    )
    grid_rf_estimators_tuning_results_df

    notify_done()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Additional trees provide only marginal improvements once the ensemble is sufficiently large. The retained sizes therefore favour the point where performance has effectively reached a plateau. Adding more trees would barely improve the models but significantly increase the computational costs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Random Forest finalists

    Two complementary configurations are retained.

    The absolute-error profile prioritizes MAE, while the Poisson profile
    sacrifices some absolute accuracy in exchange for lower RMSE, fewer
    overestimations, and an almost neutral signed bias.
    """)
    return


@app.cell
def _(pd):
    rf_finalists = [
        {
            "Model": "RandomForest",
            "Profile": "absolute_error",
            "MAE": 9.528693,
            "RMSE": 14.845954,
            "overestimation_rate": 0.408963,
            "mean_signed_error": 1.433922,
            "mean_overestimation": 13.382428,
            "params": {
                "n_estimators": 300,
                "criterion": "absolute_error",
                "max_depth": 14,
                "min_samples_leaf": 4,
                "min_samples_split": 2,
                "max_features": 0.70,
                "bootstrap": True,
                "max_samples": 1.0,
                "random_state": 42,
            },
        },
        {
            "Model": "RandomForest",
            "Profile": "poisson",
            "MAE": 9.842720,
            "RMSE": 14.325350,
            "overestimation_rate": 0.391151,
            "mean_signed_error": -0.055939,
            "mean_overestimation": 12.482731,
            "params": {
                "n_estimators": 500,
                "criterion": "poisson",
                "max_depth": 18,
                "min_samples_leaf": 2,
                "min_samples_split": 2,
                "max_features": 0.70,
                "bootstrap": True,
                "max_samples": 1.0,
                "random_state": 42,
            },
        },
    ]

    rf_finalists_df = pd.DataFrame(rf_finalists)
    rf_finalists_df
    return (rf_finalists,)


@app.cell
def _(mo):
    mo.md(r"""
    Neither configuration dominates across all evaluation criteria, so both are
    kept for later finalist comparisons.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. XGBoost

    XGBoost extends the boosting analysis with greater control over tree structure, regularization, and the training objective.

    The flexibility is particularly relevant for the RUL prediction because the objective function can be adapted to different accuracy-risk behaviours, including objectives that explicitly discourage optimistic predictions.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Coarse structural screening

    The initial search explores several boosting schedules together with the main structural and sampling parameters of the trees.
    """)
    return


@app.cell(disabled=True)
def _():
    param_grids_xgb_coarse = [
        {
            "learning_rate": [0.10],
            "n_estimators": [300],
            "max_depth": [3, 5, 7],
            "min_child_weight": [1, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.7, 1.0],
        },
        {
            "learning_rate": [0.05],
            "n_estimators": [500],
            "max_depth": [3, 5, 7],
            "min_child_weight": [1, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.7, 1.0],
        },
        {
            "learning_rate": [0.03],
            "n_estimators": [800],
            "max_depth": [3, 5, 7],
            "min_child_weight": [1, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.7, 1.0],
        },
    ]
    return (param_grids_xgb_coarse,)


@app.cell
def _(
    Path,
    XGBRegressor,
    X_cv_full,
    cv,
    finalize_checkpoint_run,
    groups_cv,
    notify_done,
    param_grids_xgb_coarse,
    pd,
    prepare_checkpoint_run,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    checkpoint_dir_xgb_coarse = Path("results/XGBoost/xgb_coarse_results")
    checkpoint_dir_xgb_coarse.mkdir(
        parents=True,
        exist_ok=True,
    )

    xgb_coarse_results = []

    for _profile in param_grids_xgb_coarse:
        _profile_id = (
            f"lr{str(_profile['learning_rate']).replace('.', '')}"
            f"_n{_profile['n_estimators']}"
        )

        _run = prepare_checkpoint_run(
            checkpoint_dir=checkpoint_dir_xgb_coarse,
            prefix="xgb_coarse",
            profile_id=_profile_id,
            param_grid=_profile,
        )

        if _run["skip"]:
            xgb_coarse_results.append(_run["existing"])
            continue

        _grid_xgb_coarse = run_grouped_grid_search(
            model=XGBRegressor(
                random_state=42,
                n_jobs=1,
                tree_method="hist",
            ),
            param_grid=_profile,
            X=X_cv_full,
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        )

        _results = process_grid_search_results(
            _grid_xgb_coarse,
            _run["checkpoint_path"],
        )

        xgb_coarse_results.append(_results)

        finalize_checkpoint_run(
            profile_id=_profile_id,
            checkpoint_path=_run["checkpoint_path"],
            running_path=_run["running_path"],
            start_time=_run["start_time"],
        )

    xgb_coarse_results_df = pd.concat(
        xgb_coarse_results,
        ignore_index=True,
    )

    notify_done()
    return xgb_coarse_results_df


@app.function
def ensure_grid_search_format(params):
    return {
        key: value if isinstance(value, list) else [value]
        for key, value in params.items()
    }


@app.cell
def _(mo):
    mo.md(r"""
    ### Objective-function screening

    Several training objectives are compared on representative XGBoost structures.

    Standard squared and absolute errors are evaluated alongside pseudo-Huber and quantile regression. A custom asymmetric objective is also introduced to penalize RUL overestimation more strongly during training.
    """)
    return


@app.cell
def _():
    xgb_screening_structures = [
        {
            "profile": "balanced_depth5",
            "max_depth": 5,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
        },
        {
            "profile": "accuracy_depth7",
            "max_depth": 7,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
        },
    ]
    return (xgb_screening_structures,)


@app.cell
def _(np):
    def make_asymmetric_mse(overestimation_weight):
        def asymmetric_mse(y_true, y_pred):
            error = y_pred - y_true

            weights = np.where(
                error > 0,
                overestimation_weight,
                1.0,
            )

            grad = weights * error
            hess = weights

            return grad, hess

        return asymmetric_mse

    return (make_asymmetric_mse,)


@app.cell
def _(mo):
    mo.md(r"""
    the asymmetric objective increases the contribution of positive prediction errors, where predicted RUL exceeds the true RUL.

    Its overestimation weight therefore controls how strongly optimistic prediction are penalized relative to conservative ones.
    """)
    return


@app.cell
def _(ParameterGrid, Path, datetime, pd, time):
    def prepare_checkpoint_run(
        checkpoint_dir: Path,
        prefix: str,
        profile_id: str,
        param_grid: dict | list[dict],
    ):
        checkpoint_path = checkpoint_dir / f"{prefix}_{profile_id}.csv"

        running_path = checkpoint_dir / f"{prefix}_{profile_id}.running"

        expected_rows = len(list(ParameterGrid(param_grid)))

        if checkpoint_path.exists():
            existing = pd.read_csv(checkpoint_path)

            if len(existing) == expected_rows:
                print(f"[SKIP] {profile_id} ({len(existing)}/{expected_rows} results)")

                if running_path.exists():
                    running_path.unlink()

                return {
                    "skip": True,
                    "existing": existing,
                    "checkpoint_path": checkpoint_path,
                    "running_path": running_path,
                    "start_time": None,
                }

            print(
                f"[INVALID CHECKPOINT] {profile_id} "
                f"({len(existing)}/{expected_rows} results) "
                "-> recompute"
            )

        print(
            f"\n{'=' * 70}\n"
            f"[START] {profile_id}\n"
            f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"{'=' * 70}"
        )

        running_path.write_text(f"Started: {datetime.now().isoformat()}")

        return {
            "skip": False,
            "existing": None,
            "checkpoint_path": checkpoint_path,
            "running_path": running_path,
            "start_time": time.perf_counter(),
        }

    return (prepare_checkpoint_run,)


@app.cell
def _(Path, time):
    def finalize_checkpoint_run(
        profile_id: str,
        checkpoint_path: Path,
        running_path: Path,
        start_time: float,
    ):
        if running_path.exists():
            running_path.unlink()

        elapsed = time.perf_counter() - start_time

        print(
            f"[DONE] {profile_id}\n"
            f"Duration: {elapsed / 60:.1f} min\n"
            f"Checkpoint: {checkpoint_path}"
        )

        return elapsed

    return (finalize_checkpoint_run,)


@app.cell
def _(make_asymmetric_mse):
    xgb_objective_profiles = {
        "squared_error": {
            "objective": "reg:squarederror",
        },
        "absolute_error": {
            "objective": "reg:absoluteerror",
        },
        "huber_0.5": {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.5,
        },
        "huber_1.0": {
            "objective": "reg:pseudohubererror",
            "huber_slope": 1.0,
        },
        "huber_2.0": {
            "objective": "reg:pseudohubererror",
            "huber_slope": 2.0,
        },
        "quantile_045": {
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.45,
        },
        "quantile_040": {
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.40,
        },
        "quantile_035": {
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.35,
        },
        "quantile_030": {
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.30,
        },
        "asym_125": {
            "objective": make_asymmetric_mse(1.25),
        },
        "asym_150": {
            "objective": make_asymmetric_mse(1.50),
        },
        "asym_200": {
            "objective": make_asymmetric_mse(2.00),
        },
        "asym_300": {
            "objective": make_asymmetric_mse(3.00),
        },
    }
    return (xgb_objective_profiles,)


@app.cell
def _(xgb_objective_profiles, xgb_screening_structures):
    xgb_objective_runs = {}

    for _structure in xgb_screening_structures:
        _structure_name = _structure["profile"]

        for _objective_name, _objective in xgb_objective_profiles.items():
            _model_params = {
                "objective": _objective["objective"],
            }

            if "huber_slope" in _objective:
                _model_params["huber_slope"] = _objective["huber_slope"]

            if "quantile_alpha" in _objective:
                _model_params["quantile_alpha"] = _objective["quantile_alpha"]

            _param_grid = {
                "max_depth": _structure["max_depth"],
                "min_child_weight": _structure["min_child_weight"],
                "subsample": _structure["subsample"],
                "colsample_bytree": _structure["colsample_bytree"],
            }

            _profile_id = f"{_structure_name}_{_objective_name}"

            xgb_objective_runs[_profile_id] = {
                "objective_name": _objective_name,
                "model_params": _model_params,
                "param_grid": _param_grid,
            }
    return (xgb_objective_runs,)


@app.cell
def _(
    Path,
    XGBRegressor,
    X_cv_full,
    cv,
    finalize_checkpoint_run,
    groups_cv,
    prepare_checkpoint_run,
    process_grid_search_results,
    run_grouped_grid_search,
    xgb_objective_runs,
    y_cv,
):
    checkpoint_dir_xgb_objectives = Path("results/XGBoost/xgb_objectives_results")

    checkpoint_dir_xgb_objectives.mkdir(
        parents=True,
        exist_ok=True,
    )

    xgb_objectives_results = []

    for _name, _config in xgb_objective_runs.items():
        _profile_id = _name

        _model_params = _config["model_params"]

        _param_grid = ensure_grid_search_format(_config["param_grid"])

        _run = prepare_checkpoint_run(
            checkpoint_dir=checkpoint_dir_xgb_objectives,
            prefix="xgb_b",
            profile_id=_profile_id,
            param_grid=_param_grid,
        )

        if _run["skip"]:
            xgb_objectives_results.append(_run["existing"])
            continue

        _model = XGBRegressor(
            random_state=42,
            n_jobs=1,
            learning_rate=0.03,
            n_estimators=800,
            tree_method="hist",
            **_model_params,
        )

        if _config["objective_name"].startswith("asym_"):
            assert callable(_model.get_params()["objective"])

        _grid_xgb_objectives = run_grouped_grid_search(
            model=_model,
            param_grid=_param_grid,
            X=X_cv_full,
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        )

        _results = process_grid_search_results(
            _grid_xgb_objectives,
            _run["checkpoint_path"],
        )

        _results["objective_profile"] = _config["objective_name"]

        _results["structure_profile"] = _profile_id.removesuffix(
            f"_{_config['objective_name']}"
        )

        _results.to_csv(
            _run["checkpoint_path"],
            index=False,
        )

        xgb_objectives_results.append(_results)

        finalize_checkpoint_run(
            profile_id=_profile_id,
            checkpoint_path=_run["checkpoint_path"],
            running_path=_run["running_path"],
            start_time=_run["start_time"],
        )
    return (xgb_objectives_results,)


@app.cell
def _(notify_done, pd, xgb_objectives_results):
    xgb_objectives_tuning_results_df = pd.concat(
        xgb_objectives_results,
        ignore_index=True,
    )

    xgb_objectives_tuning_results_df.round(3)

    notify_done()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The screening reveals three behaviours worth investigating further. Pseudo-Huber for accuracy, quantile regression for direct control of optimistic predictions, and the custom asymmetric objective for overestimation-sensitive training.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Feature ablation

    Feature subsets are evaluated separately for the retained objective profiles, since different training objectives may favour different input representations.
    """)
    return


@app.cell
def _(make_asymmetric_mse):
    xgb_ablation_profiles = {
        "accuracy": {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.5,
            "max_depth": 7,
            "min_child_weight": 5,
        },
        "balanced_quantile": {
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.45,
            "max_depth": 5,
            "min_child_weight": 1,
        },
        "asymmetric": {
            "objective": make_asymmetric_mse(3.0),
            "max_depth": 5,
            "min_child_weight": 1,
        },
        "conservative_quantile": {
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.40,
            "max_depth": 5,
            "min_child_weight": 1,
        },
    }
    return (xgb_ablation_profiles,)


@app.cell(disabled=True)
def _(
    XGBRegressor,
    cv,
    df_train_temporal,
    evaluate_grouped_cv,
    feature_sets,
    groups_cv,
    notify_done,
    xgb_ablation_profiles,
    y_cv,
):
    feature_set_results_xgb = []

    for _profile_name, _specs in xgb_ablation_profiles.items():
        _model_params = {
            "objective": _specs["objective"],
            "max_depth": _specs["max_depth"],
            "min_child_weight": _specs["min_child_weight"],
        }

        if "huber_slope" in _specs:
            _model_params["huber_slope"] = _specs["huber_slope"]

        if "quantile_alpha" in _specs:
            _model_params["quantile_alpha"] = _specs["quantile_alpha"]

        _model = XGBRegressor(
            tree_method="hist",
            n_jobs=1,
            learning_rate=0.03,
            n_estimators=800,
            subsample=0.8,
            colsample_bytree=1.0,
            random_state=42,
            **_model_params,
        )

        for _name, _features in feature_sets.items():
            _metrics = evaluate_grouped_cv(
                model=_model,
                X=df_train_temporal[_features],
                y=y_cv,
                groups=groups_cv,
                cv=cv,
            )

            feature_set_results_xgb.append(
                {
                    "feature_set": _name,
                    "ablation_profiles": _profile_name,
                    "n_features": len(_features),
                    **_metrics,
                }
            )

    notify_done()
    return (feature_set_results_xgb,)


@app.cell
def _(feature_set_results_xgb, pd):
    feature_set_results_xgb_df = pd.DataFrame(feature_set_results_xgb)

    feature_set_results_xgb_df.round(3)
    return


@app.cell
def _(df_train_temporal, feature_sets):
    final_feature_set_xgb_accuracy = feature_sets["no_means"]

    final_feature_set_xgb_asymmetric = feature_sets["no_means"]

    final_feature_set_xgb_quantile = feature_sets["no_means_no_deltas"]

    X_cv_xgb_accuracy = df_train_temporal[final_feature_set_xgb_accuracy]

    X_cv_xgb_asymmetric = df_train_temporal[final_feature_set_xgb_asymmetric]

    X_cv_xgb_quantile = df_train_temporal[final_feature_set_xgb_quantile]
    return X_cv_xgb_accuracy, X_cv_xgb_asymmetric, X_cv_xgb_quantile


@app.cell
def _(mo):
    mo.md(r"""
    Rolling-mean features are not retained by any of the selected XGBoost branches.

    The pseudo-Huber and asymmetric profiles keep raw sensors, cycles, deltas, and slopes while the quantile profile also removes delta features.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Structural refinement

    With the feature sets fixed, the three retained XGBoost branches are tuned independently.

    The pseudo-Huber branch uses an accuracy-oriented search space, while the quantile and asymmetric branches share a broader risk-oriented structural search.
    """)
    return


@app.cell
def _():
    param_grid_xgb_a_accuracy = {
        "max_depth": [5, 6, 7, 8],
        "min_child_weight": [3, 5, 8],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.85, 1.0],
    }

    param_grid_xgb_a_risk = {
        "max_depth": [4, 5, 6, 7],
        "min_child_weight": [0.5, 1, 2, 4],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.85, 1.0],
    }
    return param_grid_xgb_a_accuracy, param_grid_xgb_a_risk


@app.cell
def _(
    X_cv_xgb_accuracy,
    X_cv_xgb_asymmetric,
    X_cv_xgb_quantile,
    make_asymmetric_mse,
    param_grid_xgb_a_accuracy,
    param_grid_xgb_a_risk,
):
    tuning_a_profiles = ("accuracy_huber", "balanced_quantile", "asymmetric")

    xgb_tuning_a_runs = {}

    for _profile in tuning_a_profiles:
        if "accuracy_huber" in _profile:
            _model_params = {
                "objective": "reg:pseudohubererror",
                "huber_slope": 0.5,
            }

            _param_grid = param_grid_xgb_a_accuracy

            _X_cv = X_cv_xgb_accuracy

        if "balanced_quantile" in _profile:
            _model_params = {
                "objective": "reg:quantileerror",
                "quantile_alpha": 0.45,
            }

            _param_grid = param_grid_xgb_a_risk

            _X_cv = X_cv_xgb_quantile

        if "asymmetric" in _profile:
            _model_params = {
                "objective": make_asymmetric_mse(3.0),
            }

            _param_grid = param_grid_xgb_a_risk

            _X_cv = X_cv_xgb_asymmetric

        xgb_tuning_a_runs[_profile] = {
            "model_params": _model_params,
            "param_grid": _param_grid,
            "X_cv": _X_cv,
        }
    return tuning_a_profiles, xgb_tuning_a_runs


@app.cell
def _(
    Path,
    XGBRegressor,
    cv,
    finalize_checkpoint_run,
    groups_cv,
    prepare_checkpoint_run,
    process_grid_search_results,
    run_grouped_grid_search,
    xgb_tuning_a_runs,
    y_cv,
):
    checkpoint_dir_xgb_tuning_a = Path("results/XGBoost/xgb_tuning_a_results")

    checkpoint_dir_xgb_tuning_a.mkdir(
        parents=True,
        exist_ok=True,
    )

    xgb_tuning_a_results = []

    for _name, _config in xgb_tuning_a_runs.items():
        _profile_id = _name
        _model_params = _config["model_params"]
        _param_grid = ensure_grid_search_format(_config["param_grid"])
        _X_cv = _config["X_cv"]

        _run = prepare_checkpoint_run(
            checkpoint_dir=checkpoint_dir_xgb_tuning_a,
            prefix="xgb_b",
            profile_id=_profile_id,
            param_grid=_param_grid,
        )

        if _run["skip"]:
            xgb_tuning_a_results.append(_run["existing"])
            continue

        _model = XGBRegressor(
            learning_rate=0.03,
            n_estimators=800,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            **_model_params,
        )

        _grid_xgb_tuning_a = run_grouped_grid_search(
            model=_model,
            param_grid=_param_grid,
            X=_X_cv,
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        )

        _results = process_grid_search_results(
            _grid_xgb_tuning_a,
            _run["checkpoint_path"],
        )

        xgb_tuning_a_results.append(_results)

    finalize_checkpoint_run(
        profile_id=_profile_id,
        checkpoint_path=_run["checkpoint_path"],
        running_path=_run["running_path"],
        start_time=_run["start_time"],
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    The strongest configurations lie close to several search boundaries. Focused searches are therefore performed before fixing the structure of each branches.
    """)
    return


@app.cell
def _():
    xgb_a_boundary_grids = {
        "accuracy_huber": {
            "max_depth": [7, 8, 9],
            "min_child_weight": [8, 12, 16],
            "subsample": [0.5, 0.6, 0.7],
            "colsample_bytree": [1.0],
        },
        "balanced_quantile": {
            "max_depth": [5, 6, 7],
            "min_child_weight": [2, 4, 8],
            "subsample": [0.5, 0.6, 0.7],
            "colsample_bytree": [0.85, 1.0],
        },
        "asymmetric": {
            "max_depth": [7, 8, 9],
            "min_child_weight": [2, 4, 8],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.70, 0.85, 1.0],
        },
    }
    return (xgb_a_boundary_grids,)


@app.cell
def _(
    X_cv_xgb_accuracy,
    X_cv_xgb_asymmetric,
    X_cv_xgb_quantile,
    make_asymmetric_mse,
    tuning_a_profiles,
    xgb_a_boundary_grids,
):
    xgb_a_boundary_runs = {}

    for _profile in tuning_a_profiles:
        _param_grid = xgb_a_boundary_grids[_profile]

        if "accuracy_huber" in _profile:
            _model_params = {
                "objective": "reg:pseudohubererror",
                "huber_slope": 0.5,
            }

            _X_cv = X_cv_xgb_accuracy

        if "balanced_quantile" in _profile:
            _model_params = {
                "objective": "reg:quantileerror",
                "quantile_alpha": 0.45,
            }

            _X_cv = X_cv_xgb_quantile

        if "asymmetric" in _profile:
            _model_params = {
                "objective": make_asymmetric_mse(3.0),
            }

            _X_cv = X_cv_xgb_asymmetric

        xgb_a_boundary_runs[_profile] = {
            "model_params": _model_params,
            "param_grid": _param_grid,
            "X_cv": _X_cv,
        }
    return (xgb_a_boundary_runs,)


@app.cell
def _(
    Path,
    XGBRegressor,
    cv,
    finalize_checkpoint_run,
    groups_cv,
    notify_done,
    prepare_checkpoint_run,
    process_grid_search_results,
    run_grouped_grid_search,
    xgb_a_boundary_runs,
    y_cv,
):
    checkpoint_dir_xgb_a_boundary = Path("results/XGBoost/xgb_a_boundary_results")

    checkpoint_dir_xgb_a_boundary.mkdir(
        parents=True,
        exist_ok=True,
    )

    xgb_a_boundary_results = []

    for _name, _config in xgb_a_boundary_runs.items():
        _profile_id = _name
        _model_params = _config["model_params"]
        _param_grid = ensure_grid_search_format(_config["param_grid"])
        _X_cv = _config["X_cv"]

        _run = prepare_checkpoint_run(
            checkpoint_dir=checkpoint_dir_xgb_a_boundary,
            prefix="xgb_b",
            profile_id=_profile_id,
            param_grid=_param_grid,
        )

        if _run["skip"]:
            xgb_a_boundary_results.append(_run["existing"])
            continue

        _model = XGBRegressor(
            learning_rate=0.03,
            n_estimators=800,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            **_model_params,
        )

        _grid_xgb_a_boundary = run_grouped_grid_search(
            model=_model,
            param_grid=_param_grid,
            X=_X_cv,
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        )

        _results = process_grid_search_results(
            _grid_xgb_a_boundary,
            _run["checkpoint_path"],
        )

        _results.to_csv(
            _run["checkpoint_path"],
            index=False,
        )

        xgb_a_boundary_results.append(_results)

        finalize_checkpoint_run(
            profile_id=_profile_id,
            checkpoint_path=_run["checkpoint_path"],
            running_path=_run["running_path"],
            start_time=_run["start_time"],
        )

    notify_done()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The quantile and asymmetric branches still favour edge configurations after the first refinement, so their depth and child-weight regions are extended once more
    """)
    return


@app.cell
def _():
    param_grid_xgb_quantile_boundary_2 = {
        "max_depth": [7, 8, 9],
        "min_child_weight": [8, 12, 16],
        "subsample": [0.7],
        "colsample_bytree": [1.0],
    }

    param_grid_xgb_asymmetric_boundary_2 = {
        "max_depth": [9, 10, 11],
        "min_child_weight": [4],
        "subsample": [0.7],
        "colsample_bytree": [1.0],
    }
    return (
        param_grid_xgb_asymmetric_boundary_2,
        param_grid_xgb_quantile_boundary_2,
    )


@app.cell(disabled=True)
def _(
    XGBRegressor,
    X_cv_xgb_quantile,
    cv,
    groups_cv,
    param_grid_xgb_quantile_boundary_2,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    grid_xgb_quantile_boundary_2 = run_grouped_grid_search(
        model=XGBRegressor(
            learning_rate=0.03,
            n_estimators=800,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            objective="reg:quantileerror",
            quantile_alpha=0.45,
        ),
        param_grid=param_grid_xgb_quantile_boundary_2,
        X=X_cv_xgb_quantile,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    grid_xgb_quantile_boundary_2_results_df = process_grid_search_results(
        grid_xgb_quantile_boundary_2,
        "results/XGBoost/xgb_a_boundary_2_results/quantile_2.csv",
    )
    grid_xgb_quantile_boundary_2_results_df
    return


@app.cell(disabled=True)
def _(
    XGBRegressor,
    X_cv_xgb_asymmetric,
    cv,
    groups_cv,
    make_asymmetric_mse,
    notify_done,
    param_grid_xgb_asymmetric_boundary_2,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    grid_xgb_boundary_asymmetric_2 = run_grouped_grid_search(
        model=XGBRegressor(
            learning_rate=0.03,
            n_estimators=800,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            objective=make_asymmetric_mse(3.0),
        ),
        param_grid=param_grid_xgb_asymmetric_boundary_2,
        X=X_cv_xgb_asymmetric,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    grid_xgb_boundary_asymmetric_2_results_df = process_grid_search_results(
        grid_xgb_boundary_asymmetric_2,
        "results/XGBoost/xgb_a_boundary_2_results/asymmetric_2.csv",
    )

    grid_xgb_boundary_asymmetric_2_results_df

    notify_done()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The structural searches lead to three distinct configurations:

    - pseudo-Huber: depth 7 with a minimum child weight of 8.
    - quantile: depth of 7 with a minimum child weight of 12.
    - asymmetric: depth 9 with a minimum child weight of 4.

    All three retain a subsample ratio of 0.7 and use the full feature subset available at each split.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Regularization

    With tree structure fixed, the next stage focuses on XGBoost regularization.

    Gamma, L1 regularization (`reg_alpha`), and L2 regularization (`reg_lambda`) are varied independently for the three retained objective profiles.
    """)
    return


@app.cell
def _(make_asymmetric_mse):
    xgb_tuning_b_profiles = {
        "accuracy": {
            "max_depth": 7,
            "min_child_weight": 8,
            "subsample": 0.7,
            "colsample_bytree": 1.0,
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.5,
        },
        "quantile": {
            "max_depth": 7,
            "min_child_weight": 12,
            "subsample": 0.7,
            "colsample_bytree": 1.0,
            "objective": "reg:quantileerror",
            "quantile_alpha": 0.45,
        },
        "asymmetric": {
            "max_depth": 9,
            "min_child_weight": 4,
            "subsample": 0.7,
            "colsample_bytree": 1.0,
            "objective": make_asymmetric_mse(3.0),
        },
    }

    param_grid_xgb_b = {
        "gamma": [
            0.0,
            0.1,
            1.0,
        ],
        "reg_alpha": [
            0.0,
            0.01,
            0.1,
        ],
        "reg_lambda": [
            0.5,
            1.0,
            5.0,
        ],
    }
    return param_grid_xgb_b, xgb_tuning_b_profiles


@app.cell
def _(
    X_cv_xgb_accuracy,
    X_cv_xgb_asymmetric,
    X_cv_xgb_quantile,
    xgb_tuning_b_profiles,
):
    xgb_tuning_b_runs = {}

    for _name, _params in xgb_tuning_b_profiles.items():
        if "accuracy" in _name:
            _X_cv = X_cv_xgb_accuracy

        if "quantile" in _name:
            _X_cv = X_cv_xgb_quantile

        if "asymmetric" in _name:
            _X_cv = X_cv_xgb_asymmetric

        xgb_tuning_b_runs[_name] = {
            "model_params": _params,
            "X_cv": _X_cv,
        }
    return (xgb_tuning_b_runs,)


@app.cell(disabled=True)
def _(
    Path,
    XGBRegressor,
    cv,
    finalize_checkpoint_run,
    groups_cv,
    notify_done,
    param_grid_xgb_b,
    prepare_checkpoint_run,
    process_grid_search_results,
    run_grouped_grid_search,
    xgb_tuning_b_runs,
    y_cv,
):
    xgb_tuning_b_results = []

    checkpoint_dir_xgb_tuning_b = Path("results/XGBoost/xgb_tuning_b_results")
    checkpoint_dir_xgb_tuning_b.mkdir(
        parents=True,
        exist_ok=True,
    )

    for _name, _config in xgb_tuning_b_runs.items():
        _profile_id = _name
        _model_params = _config["model_params"]
        _param_grid = ensure_grid_search_format(param_grid_xgb_b)
        _X_cv = _config["X_cv"]

        _run = prepare_checkpoint_run(
            checkpoint_dir=checkpoint_dir_xgb_tuning_b,
            prefix="xgb_b",
            profile_id=_profile_id,
            param_grid=_param_grid,
        )

        if _run["skip"]:
            xgb_tuning_b_results.append(_run["existing"])
            continue

        _model = XGBRegressor(
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            learning_rate=0.03,
            n_estimators=800,
            **_model_params,
        )

        _grid_xgb_tuning_b = run_grouped_grid_search(
            model=_model,
            param_grid=_param_grid,
            X=_X_cv,
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        )

        _results = process_grid_search_results(
            _grid_xgb_tuning_b,
            _run["checkpoint_path"],
        )

        xgb_tuning_b_results.append(_results)

        finalize_checkpoint_run(
            profile_id=_profile_id,
            checkpoint_path=_run["checkpoint_path"],
            running_path=_run["running_path"],
            start_time=_run["start_time"],
        )

    notify_done()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The pseudo-Huber branch is sufficiently stable within the initial regularization range.

    The quantile and asymmetric profiles remain competitive at the upper L2 boundary, so `reg_lambda` is extended locally before fixing their final regularization settings.
    """)
    return


@app.cell(disabled=True)
def _(
    XGBRegressor,
    X_cv_xgb_quantile,
    cv,
    groups_cv,
    notify_done,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    param_grid_xgb_b_quantile_boundary = {
        "gamma": [2.0],
        "reg_alpha": [0.1],
        "reg_lambda": [5.0, 10.0, 20.0],
    }

    grid_xgb_b_quantile_boundary = run_grouped_grid_search(
        model=XGBRegressor(
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            learning_rate=0.03,
            n_estimators=800,
            max_depth=7,
            min_child_weight=12,
            subsample=0.7,
            colsample_bytree=1.0,
            objective="reg:quantileerror",
            quantile_alpha=0.45,
        ),
        param_grid=param_grid_xgb_b_quantile_boundary,
        X=X_cv_xgb_quantile,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    _results = process_grid_search_results(
        grid_xgb_b_quantile_boundary,
        "results/XGBoost/xgb_tuning_b_results/xgb_b_quantile_boundary.csv",
    )

    notify_done()
    return


@app.cell(disabled=True)
def _(
    XGBRegressor,
    X_cv_xgb_asymmetric,
    cv,
    groups_cv,
    make_asymmetric_mse,
    process_grid_search_results,
    run_grouped_grid_search,
    y_cv,
):
    param_grid_xgb_b_asymmetric_boundary = {
        "gamma": [0.0],
        "reg_alpha": [0.1],
        "reg_lambda": [5.0, 10.0, 20.0],
    }

    grid_xgb_b_asymmetric_boundary = run_grouped_grid_search(
        model=XGBRegressor(
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            learning_rate=0.03,
            n_estimators=800,
            max_depth=9,
            min_child_weight=4,
            subsample=0.7,
            colsample_bytree=1.0,
            objective=make_asymmetric_mse(3.0),
        ),
        param_grid=param_grid_xgb_b_asymmetric_boundary,
        X=X_cv_xgb_asymmetric,
        y=y_cv,
        groups=groups_cv,
        cv=cv,
    )

    _results = process_grid_search_results(
        grid_xgb_b_asymmetric_boundary,
        "results/XGBoost/xgb_tuning_b_results/xgb_b_asymmetric_boundary.csv",
    )

    _results
    return


@app.cell
def _(mo):
    mo.md(r"""
    Structure and regularization are now fixed for all three XGBoost branches. The remaining experiments therefore focus only on objective-specific behaviour and boosting convergence.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Objective-specific tuning

    With feature sets, tree structure, and regularization fixed, the next stage focuses only on the parameter controlling each objective's behaviour.

    The pseudo-Huber slope, quantile level, and asymmetric overestimation weight are tuned independently for their respective branches.
    """)
    return


@app.cell
def _(
    X_cv_xgb_accuracy,
    X_cv_xgb_asymmetric,
    X_cv_xgb_quantile,
    make_asymmetric_mse,
):
    xgb_c_base_profiles = {
        "accuracy": {
            "X_cv": X_cv_xgb_accuracy,
            "model_params": {
                "objective": "reg:pseudohubererror",
                "huber_slope": 0.5,
                "learning_rate": 0.03,
                "n_estimators": 800,
                "max_depth": 7,
                "min_child_weight": 8,
                "subsample": 0.7,
                "colsample_bytree": 1.0,
                "gamma": 0.1,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        "quantile": {
            "X_cv": X_cv_xgb_quantile,
            "model_params": {
                "objective": "reg:quantileerror",
                "quantile_alpha": 0.45,
                "learning_rate": 0.03,
                "n_estimators": 800,
                "max_depth": 7,
                "min_child_weight": 12,
                "subsample": 0.7,
                "colsample_bytree": 1.0,
                "gamma": 2.0,
                "reg_alpha": 0.1,
                "reg_lambda": 20.0,
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        "asymmetric": {
            "X_cv": X_cv_xgb_asymmetric,
            "model_params": {
                "objective": make_asymmetric_mse(3.0),
                "learning_rate": 0.03,
                "n_estimators": 800,
                "max_depth": 9,
                "min_child_weight": 4,
                "subsample": 0.7,
                "colsample_bytree": 1.0,
                "gamma": 0.0,
                "reg_alpha": 0.1,
                "reg_lambda": 20.0,
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 1,
            },
        },
    }
    return (xgb_c_base_profiles,)


@app.cell
def _(
    X_cv_xgb_accuracy,
    X_cv_xgb_asymmetric,
    X_cv_xgb_quantile,
    make_asymmetric_mse,
    xgb_c_base_profiles,
):
    xgb_c1_runs = {
        "accuracy": {
            "model_params": (xgb_c_base_profiles["accuracy"]["model_params"].copy()),
            "param_grid": {
                "huber_slope": [
                    0.30,
                    0.40,
                    0.50,
                    0.60,
                    0.70,
                ],
            },
            "X_cv": X_cv_xgb_accuracy,
            "objective_value": None,
        },
        "quantile": {
            "model_params": (xgb_c_base_profiles["quantile"]["model_params"].copy()),
            "param_grid": {
                "quantile_alpha": [
                    0.35,
                    0.375,
                    0.40,
                    0.425,
                    0.45,
                    0.475,
                    0.50,
                ],
            },
            "X_cv": X_cv_xgb_quantile,
            "objective_value": None,
        },
    }

    for _weight in [
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
    ]:
        _params = xgb_c_base_profiles["asymmetric"]["model_params"].copy()

        _params["objective"] = make_asymmetric_mse(_weight)

        xgb_c1_runs[f"asymmetric_{_weight}"] = {
            "model_params": _params,
            "param_grid": {},
            "X_cv": X_cv_xgb_asymmetric,
            "objective_value": _weight,
        }
    return (xgb_c1_runs,)


@app.cell(disabled=True)
def _(
    Path,
    XGBRegressor,
    cv,
    finalize_checkpoint_run,
    groups_cv,
    notify_done,
    pd,
    prepare_checkpoint_run,
    process_grid_search_results,
    run_grouped_grid_search,
    xgb_c1_runs,
    y_cv,
):
    checkpoint_dir_xgb_c1 = Path("results/XGBoost/xgb_c1_results")
    checkpoint_dir_xgb_c1.mkdir(
        parents=True,
        exist_ok=True,
    )

    xgb_c1_results = []

    for _name, _config in xgb_c1_runs.items():
        _model_params = _config["model_params"]
        _param_grid = ensure_grid_search_format(_config["param_grid"])
        _X_cv = _config["X_cv"]

        _run = prepare_checkpoint_run(
            checkpoint_dir=checkpoint_dir_xgb_c1,
            prefix="xgb_c1",
            profile_id=_name,
            param_grid=_param_grid,
        )

        if _run["skip"]:
            xgb_c1_results.append(_run["existing"])
            continue

        _model = XGBRegressor(
            **_model_params,
        )

        _grid_xgb_c1 = run_grouped_grid_search(
            model=_model,
            param_grid=_param_grid,
            X=_X_cv,
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        )

        _results = process_grid_search_results(
            _grid_xgb_c1,
            _run["checkpoint_path"],
        )

        _results["profile"] = _name

        _results.to_csv(
            _run["checkpoint_path"],
            index=False,
        )

        xgb_c1_results.append(_results)

        finalize_checkpoint_run(
            profile_id=_name,
            checkpoint_path=_run["checkpoint_path"],
            running_path=_run["running_path"],
            start_time=_run["start_time"],
        )

    xgb_c1_results_df = pd.concat(
        xgb_c1_results,
        ignore_index=True,
    )

    notify_done()
    return xgb_c1_results_df


@app.cell
def _(mo):
    mo.md(r"""
    The search retains a pseudo-Huber slope of 0.4, a quantile level of 0.475, and an asymmetric overestimation weight of 3.5.

    These settings preserve the intended distinction between the three branches: predictive accuracy, quantile-based control of optimistic predictions, and asymmetric penalization of overestimation.
    """)
    return


@app.cell
def _(make_asymmetric_mse, xgb_c_base_profiles):
    xgb_c_selected_profiles = {
        name: {
            "X_cv": profile["X_cv"],
            "model_params": profile["model_params"].copy(),
        }
        for name, profile in xgb_c_base_profiles.items()
    }

    xgb_c_selected_profiles["accuracy"]["model_params"]["huber_slope"] = 0.40

    xgb_c_selected_profiles["quantile"]["model_params"]["quantile_alpha"] = 0.475

    xgb_c_selected_profiles["asymmetric"]["model_params"]["objective"] = (
        make_asymmetric_mse(3.5)
    )
    return (xgb_c_selected_profiles,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Boosting convergence

    The final tuning stage compares several learning-rate / estimator schedules while keeping the previously selected structure, regularization, and objective settings fixed.

    Smaller learning rates are paired with more boosting stages to verify whether finer updates improve generalization for each XGBoost branch.
    """)
    return


@app.cell
def _():
    xgb_c2_param_grid = [
        {
            "learning_rate": [0.05],
            "n_estimators": [500],
        },
        {
            "learning_rate": [0.03],
            "n_estimators": [800],
        },
        {
            "learning_rate": [0.02],
            "n_estimators": [1200],
        },
        {
            "learning_rate": [0.015],
            "n_estimators": [1600],
        },
    ]
    return (xgb_c2_param_grid,)


@app.cell
def _(xgb_c2_param_grid, xgb_c_selected_profiles):
    xgb_c2_runs = {}

    for _profile_name, _profile in xgb_c_selected_profiles.items():
        xgb_c2_runs[_profile_name] = {
            "model_params": _profile["model_params"].copy(),
            "param_grid": xgb_c2_param_grid,
            "X_cv": _profile["X_cv"],
        }
    return (xgb_c2_runs,)


@app.cell
def _(
    Path,
    XGBRegressor,
    cv,
    finalize_checkpoint_run,
    groups_cv,
    notify_done,
    pd,
    prepare_checkpoint_run,
    process_grid_search_results,
    run_grouped_grid_search,
    xgb_c2_runs,
    y_cv,
):
    checkpoint_dir_xgb_c2 = Path("results/XGBoost/xgb_c2_results")
    checkpoint_dir_xgb_c2.mkdir(
        parents=True,
        exist_ok=True,
    )

    xgb_c2_results = []

    for _name, _config in xgb_c2_runs.items():
        _model_params = _config["model_params"]
        _param_grid = _config["param_grid"]

        _X_cv = _config["X_cv"]

        _run = prepare_checkpoint_run(
            checkpoint_dir=checkpoint_dir_xgb_c2,
            prefix="xgb_c2",
            profile_id=_name,
            param_grid=_param_grid,
        )

        if _run["skip"]:
            xgb_c2_results.append(_run["existing"])
            continue

        _model = XGBRegressor(
            **_model_params,
        )

        _grid_xgb_c2 = run_grouped_grid_search(
            model=_model,
            param_grid=_param_grid,
            X=_X_cv,
            y=y_cv,
            groups=groups_cv,
            cv=cv,
        )

        _results = process_grid_search_results(
            _grid_xgb_c2,
            _run["checkpoint_path"],
        )

        _results["profile"] = _name

        _results.to_csv(
            _run["checkpoint_path"],
            index=False,
        )

        xgb_c2_results.append(_results)

        finalize_checkpoint_run(
            profile_id=_name,
            checkpoint_path=_run["checkpoint_path"],
            running_path=_run["running_path"],
            start_time=_run["start_time"],
        )

    xgb_c2_results_df = pd.concat(
        xgb_c2_results,
        ignore_index=True,
    )

    notify_done()
    return xgb_c2_results_df


@app.cell
def _(mo):
    mo.md(r"""
    The convergence search retains `0.02 / 1200` for both the pseudo-Huber and quantile profiles, while the asymmetric branch benefits slightly from the slower `0.015 / 1600` schedule.

    Improvements have become marginal at this stage, so the XGBoost tuning process is considered complete.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### XGBoost finalists

    The final tuned configurations are summarized below.
    """)
    return


@app.cell
def _(make_asymmetric_mse, pd):
    xgb_finalists = [
        {
            "Model": "XGBoost",
            "Profile": "pseudohuber",
            "MAE": 8.744644,
            "RMSE": 13.997903,
            "overestimation_rate": 0.460580,
            "mean_signed_error": 1.020552,
            "mean_overestimation": 10.569036,
            "params": {
                "objective": "reg:pseudohubererror",
                "huber_slope": 0.40,
                "learning_rate": 0.02,
                "n_estimators": 1200,
                "max_depth": 7,
                "min_child_weight": 8,
                "subsample": 0.7,
                "colsample_bytree": 1.0,
                "gamma": 0.1,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        {
            "Model": "XGBoost",
            "Profile": "quantile",
            "MAE": 9.099599,
            "RMSE": 14.090000,
            "overestimation_rate": 0.408841,
            "mean_signed_error": 0.207232,
            "mean_overestimation": 11.363608,
            "params": {
                "objective": "reg:quantileerror",
                "quantile_alpha": 0.475,
                "learning_rate": 0.02,
                "n_estimators": 1200,
                "max_depth": 7,
                "min_child_weight": 12,
                "subsample": 0.7,
                "colsample_bytree": 1.0,
                "gamma": 2.0,
                "reg_alpha": 0.1,
                "reg_lambda": 20.0,
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        {
            "Model": "XGBoost",
            "Profile": "asymmetric",
            "MAE": 9.499238,
            "RMSE": 13.970187,
            "overestimation_rate": 0.440060,
            "mean_signed_error": -0.737166,
            "mean_overestimation": 9.936854,
            "params": {
                "objective": make_asymmetric_mse(3.5),
                "learning_rate": 0.015,
                "n_estimators": 1600,
                "max_depth": 9,
                "min_child_weight": 4,
                "subsample": 0.7,
                "colsample_bytree": 1.0,
                "gamma": 0.0,
                "reg_alpha": 0.1,
                "reg_lambda": 20.0,
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 1,
            },
        },
    ]

    xgb_finalists_df = pd.DataFrame(xgb_finalists)

    xgb_finalists_df
    return (xgb_finalists,)


@app.cell
def _(mo):
    mo.md(r"""
    The pseudo-Huber configuration achieves the lowest MAE, while the
    asymmetric profile reaches the lowest RMSE.

    The quantile configuration records the lowest overestimation rate.
    Since no profile dominates across all evaluation criteria, all three are
    retained for later finalist comparisons.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Tabular model finalists

    The retained candidates from Gradient Boosting, Extra Trees, Random Forest, and XGBoost are now combined into a common comparison table.

    These results summarize grouped cross-validation model-selection scores only. They preserve the most relevant accuracy-risk trade-offs but must not be interpreted as development-validation or official-test performance.
    """)
    return


@app.cell
def _(et_finalists, gb_finalists, pd, rf_finalists, xgb_finalists):
    tabular_finalists = [
        *gb_finalists,
        *et_finalists,
        *rf_finalists,
        *xgb_finalists,
    ]

    tabular_finalists_df = pd.DataFrame(tabular_finalists)
    tabular_finalists_columns = [
        "Model",
        "Profile",
        "MAE",
        "RMSE",
        "overestimation_rate",
        "mean_signed_error",
        "mean_overestimation",
    ]

    tabular_finalists_df[tabular_finalists_columns].sort_values("MAE").round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    XGBoost provides the strongest accuracy among the retained tabular candidates, with the pseudo-Huber profile achieving the lowest cross-validated MAE.

    However, the comparison also confirms that minimizing MAE alone does not determine a unique best model. Several candidates provide different compromises between absolute prediction error, optimistic prediction frequency, and signed bias.

    The alternatives are therefore preserved rather than selecting a final winner from cross-validation alone.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Conclusion

    This notebook established the tabular machine-learning benchmark for FD001.

    Model development combined grouped cross-validation, model-specific feature ablation, and progressive hyperparameter tuning. In addition to MAE and RMSE, overestimation-related metrics were monitored throughout the process to account for the asymmetric operational consequences of optimistic RUL predictions.

    Gradient Boosting, Extra Trees, Random Forest, and XGBoost produced several complementary finalists. XGBoost achieves the strongest overall cross-validated accuracy and also demonstrated how alternative objectives can explicitly modify the accuracy-risk trade-off.

    The 20-engine development-validation subset has already informed historical model comparisons and is not an independent final holdout. The official FD001 test remains reserved for evaluation after model selection is frozen. No primary model has been frozen at this stage.
    """)
    return


if __name__ == "__main__":
    app.run()
