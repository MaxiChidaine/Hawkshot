import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # FD001 baseline models

    The previous notebook transformed the FD001 run-to-failure trajectories into a supervised regression problem and defined several feature configurations for RUL prediction.

    The objective of this notebook is to establish reproducible baseline performances and progressively evaluate different model families.

    The experiments begin with deliberately simple references before introducing increasingly expressive models. At each stage, only one modelling choice is changed so its contribution can be interpreted independently.

    The capped RUL at 125 cycles remains the modelling target. The same engine-level training and validation split defined previously is reconstructed to ensure that all experiments are evaluated under identical conditions.

    Model performance is primarily evaluated using Mean Absolute Error (MAE), expressed directly in cycles, while Root Mean Squared Error (RMSE) is used as a complementary metric that gives greater weight to large prediction errors.

    The official FD001 test set remains untouched throughout this notebook.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Data preparation and evaluation setup

    To keep this notebook independently executable, the cleaned FD001 dataset, capped RUL target, and fixed engine-level train/validation split are reconstructed below.

    The split is performed by engine rather than by individual observations. All cycles belonging to a given engine therefore remain entirely within either the training or validation set.

    The same `random_state` and split ratio used in the feature-preparation notebook are retained so that model comparisons are performed on an unchanged validation set.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
    )
    from sklearn.svm import SVR

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
        DecisionTreeRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        Lasso,
        LinearRegression,
        Pipeline,
        RandomForestRegressor,
        Ridge,
        SVR,
        StandardScaler,
        add_temporal_features,
        available_sensors,
        df_filtered,
        mean_absolute_error,
        mo,
        np,
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
def _(mo):
    mo.md(r"""
    ## 2. Naive constant baselines

    Before training a predictive model, a minimum reference performance is required.

    Two constant predictors are considered. Neither uses any information from the input features :
    - The training-target median provides the optimal constant prediction with respect to MAE.
    - The training-target mean provides the optimal constant prediction with respect to MSE and RMSE.

    These baselines represent the performance that can be obtained without learning anything about engine age or condition.
    """)
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
def _(mo):
    mo.md(r"""
    The constant baselines produce errors of approximately 36 cycles MAE and 42-45 cycles RMSE. Any useful predictive model should therefore improve substantially upon these references.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Linear regression baselines

    ### 3.1 Cycle only

    The first predictive baseline uses only the current operating cycle.

    This experiment measures how much RUL information can be obtained from engine age alone, without using any sensor measurements. It provides an important reference for determining whether sensor-based models learn information beyond the simple progression of operating time.
    """)
    return


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
def _(mo):
    mo.md(r"""
    The cycle-only regression substantially improves over the constant baselines, reaching a MAE of approximately 20.5 cycles.

    However, the visualisation highlights an important limitation. A single linear relationship assigns the same predicted RUL to every engine observed at the same cycle, even though their true remaining lifetimes may differ considerably.

    The model also cannot reproduce the capped target structure. Early life predictions may exceed the 125-cycle cap, while the linear trend cannot represent both the initial plateau and the later RUL decline simultaneously.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.2 Raw sensors

        The second baseline removes engine age and uses only the 14 retained sensor measurements.

        A `StandardScaler` is fitted within the modelling pipeline so that preprocessing parameters are learned exclusively from the training data. Although scaling is not required for ordinary least-squares regression to produce valid predictions, it places the sensor variables on comparable scales and prepares a consistent workflow for later regularised linear models.
    """)
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
def _(mo):
    mo.md(r"""
    Using only the instantaneous sensor measurements reduces the validation MAE from 20.5 to 16.5 cycles.

    The sensor state therefore contains predictive information that cannot be obtained from operating age alone.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.3 Raw sensors and operating cycle

        The operating cycle is now added to the raw sensor measurements while keeping the same regression model and preprocessing pipeline.

        This isolates the incremental contribution of engine age once the current sensor state is already known.
    """)
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
def _(mo):
    mo.md(r"""
    Combining the sensor measurements with the operating cycle further reduces the MAE to 14.5 cycles.

        Engine age and instantaneous sensor condition provide complementary information for RUL prediction.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.4 Raw sensors and temporal features

        The previous notebook introduced causal rolling means, deltas, and slopes to represent recent sensor evolution.

        These features are now evaluated without the operating cycle in order to determine whether recent sensor dynamics provide information beyond instantaneous sensor measurements alone.

        Temporal features are generated separately for the training and validation engine sets using only current and past observations.
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
    return mae_raw_temporal, rmse_raw_temporal, temporal_features


@app.cell
def _(mo):
    mo.md(r"""
    Adding temporal information further improves the linear model, reducing the MAE from approximately 16.5 to 14.0 cycles.

        Recent sensor evolution therefore contains predictive information that is not fully represented by instantaneous sensor values.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.5 Raw sensors, temporal features and operating cycle

        The final linear feature configuration combines the three information sources identified so far:

        - current sensor state,
        - recent sensor dynamics,
        - absolute operating age.

        This experiment evaluates whether the operating cycle remains informative after the model already has access to temporal sensor behaviour.
    """)
    return


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
    return (
        X_train_raw_temporal_cycle,
        X_validation_temporal_cycle,
        mae_raw_temporal_cycle,
        model_raw_temporal_cycle,
        raw_temporal_cycle_features,
        rmse_raw_temporal_cycle,
    )


@app.cell
def _(mo):
    mo.md(r"""
    The combined feature set achieves the strongest ordinary linear-regression performance, with an MAE of approximately 12.9 cycles and an RMSE of 15.9 cycles.

    This improvement over the temporal-only model indicates that recent sensor dynamics do not completely replace the information provided by absolute engine age.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.6 Linear baseline comparison

        The previous experiments progressively introduced engine age, instantaneous sensor measurements, and temporal sensor information while keeping the same linear modelling framework.

        The results are summarised below to quantify the contribution of each feature configurations under identical training and validation conditions.
    """)
    return


@app.cell
def _(
    mae_baseline_mean,
    mae_baseline_median,
    mae_cycle_only,
    mae_raw_sensors,
    mae_raw_temporal,
    mae_raw_temporal_cycle,
    mae_sensors_cycle,
    pd,
    rmse_baseline_mean,
    rmse_baseline_median,
    rmse_cycle_only,
    rmse_raw_sensors,
    rmse_raw_temporal,
    rmse_raw_temporal_cycle,
    rmse_sensors_cycle,
):
    models = [
        "constant_median",
        "constant_mean",
        "cycle_only",
        "raw_sensors",
        "sensors_cycle",
        "raw_temporal",
        "raw_temporal_cycle",
    ]
    maes = [
        mae_baseline_median,
        mae_baseline_mean,
        mae_cycle_only,
        mae_raw_sensors,
        mae_sensors_cycle,
        mae_raw_temporal,
        mae_raw_temporal_cycle,
    ]
    rmses = [
        rmse_baseline_median,
        rmse_baseline_mean,
        rmse_cycle_only,
        rmse_raw_sensors,
        rmse_sensors_cycle,
        rmse_raw_temporal,
        rmse_raw_temporal_cycle,
    ]

    pd.DataFrame({"model": models, "MAE": maes, "RMSE": rmses}).round(2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Performance improves consistently as additional information is introduced.

    The cycle-only model already provides a substantial improvement over the constant baselines, confirming that engine age contains strong information. Raw sensor measurements outperform the cycle-only model, showing that instantaneous engine condition provides additional predictive information.

    Combining sensors with the operating cycle further improves performance, while temporal features provide another significant gain by describing recent sensor evolution.

    The strongest ordinary linear-regression configuration combines raw sensors, temporal features and operating cycle, reaching a validation MAE of approximately 12.9 cycles and an RMSE of 15.9 cycles.

    These results establish the complete linear feature set as the reference configuration for the regularised models evaluated in the next section.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Regularised linear models

    The complete linear feature set contains more than one hundred predictors, many of which represent related transformations of the same underlying sensors.

    This creates substantial potential redundancy and may lead ordinary linear regression to distribute information across large or unstable coefficients.

    Ridge and Lasso regression are therefore explored as regularised alternatives.

    Ridge applies an L2 penalty that reduces coefficient magnitude while generally retaining all predictors. Lasso applies an L1 penalty and can force some coefficients exactly to zero, providing an implicit form of feature selection.

    Because regularisation depends on coefficient magnitude, all predictors are standardised within the modelling pipeline.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.1 Ridge regression

    Ridge regularisation strength is controlled by the hyperparameter `alpha`.

    A small exploratory grid is evaluated on the fixed validation set to observe how increasing regularisation affects predictive performance. This is an exploratory comparison rather than final hyperparameter tuning, which will be performed later for the strongest model candidates.
    """)
    return


@app.cell
def _(
    Pipeline,
    Ridge,
    StandardScaler,
    X_train_raw_temporal_cycle,
    X_validation_temporal_cycle,
    mean_absolute_error,
    pd,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    _alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    maes_ridge = []
    rmses_ridge = []

    for _alpha in _alphas:
        model_ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(_alpha))])

        model_ridge.fit(X_train_raw_temporal_cycle, y_train)

        y_pred_ridge = model_ridge.predict(X_validation_temporal_cycle)

        mae_ridge = mean_absolute_error(y_validation, y_pred_ridge)

        maes_ridge.append(mae_ridge)

        rmse_ridge = root_mean_squared_error(y_validation, y_pred_ridge)

        rmses_ridge.append(rmse_ridge)

    pd.DataFrame({"alpha": _alphas, "MAE": maes_ridge, "RMSE": rmses_ridge}).round(3)
    return mae_ridge, model_ridge, rmse_ridge


@app.cell
def _(model_ridge):
    _coef = model_ridge.named_steps["ridge"].coef_
    return


@app.cell
def _(
    Pipeline,
    Ridge,
    StandardScaler,
    X_train_raw_temporal_cycle,
    model_raw_temporal_cycle,
    np,
    pd,
    y_train,
):
    model_ridge_alpha_10 = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=10))]
    )

    model_ridge_alpha_1000 = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1000))]
    )

    model_ridge_alpha_10.fit(X_train_raw_temporal_cycle, y_train)
    model_ridge_alpha_1000.fit(X_train_raw_temporal_cycle, y_train)

    coef_ridge_alpha_10 = model_ridge_alpha_10.named_steps["ridge"].coef_

    coef_ridge_alpha_1000 = model_ridge_alpha_1000.named_steps["ridge"].coef_

    coef_linear_regression = model_raw_temporal_cycle.named_steps["regressor"].coef_

    _coefs = [coef_linear_regression, coef_ridge_alpha_10, coef_ridge_alpha_1000]
    norm_L2 = []
    max_absolute = []

    for _coef in _coefs:
        norm_L2.append(np.linalg.norm(_coef))
        max_absolute.append(max(abs(_coef)))

    _models = ["linear_regression", "ridge_10", "ridge_1000"]

    pd.DataFrame(
        {
            "Model": _models,
            "L2_norm": norm_L2,
            "max_abs": max_absolute,
        }
    ).round(2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Moderate Ridge regularisation reduces the overall magnitude of the coefficients without materially changing validation performance.

    With `alpha=10`, the coefficient L2 norm decreases substantially while MAE remains almost unchanged. At `alpha=1000`, coefficients are compressed much more strongly and predictive performance begins to deteriorate.

    Ridge therefore substantially reduces coefficient magnitude but provides little evidence of a meaningful predictive advantage over ordinary linear regression for the current feature set.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.2 Lasso regression

    Lasso is evaluated next to determine whether redundant predictors can be removed while preserving predictive performance.

    As `alpha` increases, stronger L1 regularisation is expected to drive an increasing number of coefficients exactly to zero.

    The comparison therefore considers both predictive error and the number of active features.
    """)
    return


@app.cell
def _(
    Lasso,
    Pipeline,
    StandardScaler,
    X_train_raw_temporal_cycle,
    X_validation_temporal_cycle,
    mean_absolute_error,
    pd,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    _alphas = [0.001, 0.003, 0.01, 0.03, 0.1]
    maes_lasso = []
    rmses_lasso = []
    non_zero_counts = []

    for _alpha in _alphas:
        model_lasso = Pipeline(
            [("scaler", StandardScaler()), ("lasso", Lasso(_alpha, max_iter=30000))]
        )

        model_lasso.fit(X_train_raw_temporal_cycle, y_train)

        y_pred_lasso = model_lasso.predict(X_validation_temporal_cycle)

        mae_lasso = mean_absolute_error(y_validation, y_pred_lasso)

        maes_lasso.append(mae_lasso)

        rmse_lasso = root_mean_squared_error(y_validation, y_pred_lasso)

        rmses_lasso.append(rmse_lasso)

        n_not_zero_coef = 0

        for _coef in model_lasso.named_steps["lasso"].coef_:
            if _coef != 0:
                n_not_zero_coef += 1

        non_zero_counts.append(n_not_zero_coef)

    pd.DataFrame(
        {
            "alpha": _alphas,
            "MAE": maes_lasso,
            "RMSE": rmses_lasso,
            "features kept": non_zero_counts,
        }
    ).round(3)
    return mae_lasso, rmse_lasso


@app.cell
def _(mo):
    mo.md(r"""
    A moderate regularisation level around `alpha=0.03` provides the strongest result in the current exploratory grid.

        At this value, Lasso reduces the active feature set from 113 to 88 predictors while slightly improving both MAE and RMSE relative to the unregularised linear model.

        Increasing `alpha` further removes additional features but begins to degrade predictive performance.
    """)
    return


@app.cell
def _(Lasso, Pipeline, StandardScaler, X_train_raw_temporal_cycle, y_train):
    model_lasso_alpha_selected = Pipeline(
        [("scaler", StandardScaler()), ("lasso", Lasso(alpha=0.03, max_iter=30000))]
    )

    model_lasso_alpha_selected.fit(X_train_raw_temporal_cycle, y_train)

    coef_lasso = model_lasso_alpha_selected.named_steps["lasso"].coef_
    return (coef_lasso,)


@app.cell
def _(coef_lasso, pd, raw_temporal_cycle_features):
    df_lasso_coefs = pd.DataFrame(
        {"feature": raw_temporal_cycle_features, "coefficients": coef_lasso}
    )

    df_lasso_coefs[df_lasso_coefs["coefficients"] == 0]
    return (df_lasso_coefs,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.3 Exploratory feature selection

    The predictors assigned a zero coefficient by the selected Lasso model are examined as an exploratory indication of feature redundancy.

    A zero coefficient does not imply that the corresponding variable contains no information. Highly correlated representations of the same sensor behaviour may allow Lasso to retain one feature while discarding another.

    The selected non-zero features are therefore tested separately using an ordinary linear regression rather than being permanently removed from the prepared dataset
    """)
    return


@app.cell
def _(
    LinearRegression,
    Pipeline,
    StandardScaler,
    df_lasso_coefs,
    df_train_temporal,
    df_validation_temporal,
    mean_absolute_error,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    non_zero_coefficients = df_lasso_coefs[df_lasso_coefs["coefficients"] != 0]
    non_zero_features = non_zero_coefficients["feature"]

    X_train_non_zero = df_train_temporal[non_zero_features]
    X_validation_non_zero = df_validation_temporal[non_zero_features]

    model_non_zero_features = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )

    model_non_zero_features.fit(X_train_non_zero, y_train)

    y_pred_non_zero = model_non_zero_features.predict(X_validation_non_zero)

    mae_non_zero = mean_absolute_error(y_validation, y_pred_non_zero)

    rmse_non_zero = root_mean_squared_error(y_validation, y_pred_non_zero)

    f"MAE : {mae_non_zero}, RMSE: {rmse_non_zero}"
    return mae_non_zero, rmse_non_zero


@app.cell
def _(mo):
    mo.md(r"""
    Retraining ordinary linear regression using only the 88 features retained by Lasso produced almost identical validation performance to the complete 113-feature model.

    This suggests that the 25 removed predictors are largely redundant within the current linear modelling framework.

    Interestingly, several discarded representations belong to `sensor_9` and `sensor_14`, the two sensors previously identified as having the most heterogeneous fleet-level degradation behaviour. This observation is consistent with the earlier EDA, although it does not establish that these sensors are intrinsically uninformative.

    Although the reduced feature set preserves linear-regression performance, the removed predictors are not permanently discarded. Nonlinear models may exploit interactions or alternative representations that are not useful with a linear framework. The complete raw, temporal and cycle feature set is therefore retained for the following baseline experiments.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Tree-based baseline

    The previous experiments showed that adding regularisation to the linear model provides little additional predictive improvement. This suggests that the remaining modelling error may partly result from relationships that cannot be represented by a linear combination of the available predictors.

    Tree-based models are therefore evaluated using the complete raw, temporal and cycle feature set. Unlike the previous linear models, decision trees do not require feature standardisation because their predictions are based on threshold splits rather than feature magnitude.

    The objective of this section is not to perform exhaustive hyperparameter optimisation, but to determine whether nonlinear tree-based models provide a meaningful improvement over the established linear baselines.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.1 Decision tree

    A single decision tree is first evaluated to determine whether nonlinear relationships and interactions between predictors improve RUL estimation.

    An unrestricted tree is trained initially to illustrate the effect of allowing the model to grow without regularisation.
    """)
    return


@app.cell
def _(
    DecisionTreeRegressor,
    X_train_raw_temporal_cycle,
    X_validation_temporal_cycle,
    mean_absolute_error,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    model_dt = DecisionTreeRegressor(max_depth=None, random_state=42)
    model_dt.fit(X_train_raw_temporal_cycle, y_train)

    y_pred_train_dt = model_dt.predict(X_train_raw_temporal_cycle)
    y_pred_validation_dt = model_dt.predict(X_validation_temporal_cycle)

    mae_train_dt = mean_absolute_error(y_train, y_pred_train_dt)
    mae_validation_dt = mean_absolute_error(y_validation, y_pred_validation_dt)

    rmse_train_dt = root_mean_squared_error(y_train, y_pred_train_dt)
    rmse_validation_dt = root_mean_squared_error(y_validation, y_pred_validation_dt)

    f"MAE_train : {mae_train_dt}, MAE_validation {mae_validation_dt}, RMSE_train : {rmse_train_dt}, RMSE_validation : {rmse_validation_dt}"
    return mae_train_dt, mae_validation_dt, rmse_validation_dt


@app.cell
def _(mo):
    mo.md(r"""
    The unrestricted tree achieves zero training error but substantially worse validation performance, with a validation MAE of approximately 12.1 cycles and RMSE close to 19.6 cycles.

    The tree reaches a depth of 31 and contains several thousand leaves, indicating that it has sufficient capacity to reproduce the training observations almost exactly. The large difference between training and validation performance is therefore a clear indication of overfitting.

    Tree depth is consequently restricted in the following experiment to evaluate the trade-off between model complexity and generalisation.
    """)
    return


@app.cell
def _(
    DecisionTreeRegressor,
    X_train_raw_temporal_cycle,
    X_validation_temporal_cycle,
    mean_absolute_error,
    pd,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    _depths = [2, 4, 6, 8, 10, 12, 15, 20, 25, 31]
    n_leaves = []
    maes_train_dt_multiple_depths = []
    maes_validation_dt_multiple_depths = []
    rmses_train_dt_multiple_depths = []
    rmses_validation_dt_multiple_depths = []

    for _depth in _depths:
        model_dt_multiple_depths = DecisionTreeRegressor(
            max_depth=_depth, random_state=42
        )
        model_dt_multiple_depths.fit(X_train_raw_temporal_cycle, y_train)

        y_pred_train_dt_multiple_depths = model_dt_multiple_depths.predict(
            X_train_raw_temporal_cycle
        )
        y_pred_validation_dt_multiple_depths = model_dt_multiple_depths.predict(
            X_validation_temporal_cycle
        )

        mae_train_dt_multiple_depths = mean_absolute_error(
            y_train, y_pred_train_dt_multiple_depths
        )
        mae_validation_dt_multiple_depths = mean_absolute_error(
            y_validation, y_pred_validation_dt_multiple_depths
        )

        rmse_train_dt_multiple_depths = root_mean_squared_error(
            y_train, y_pred_train_dt_multiple_depths
        )
        rmse_validation_dt_multiple_depths = root_mean_squared_error(
            y_validation, y_pred_validation_dt_multiple_depths
        )

        maes_train_dt_multiple_depths.append(mae_train_dt_multiple_depths)
        maes_validation_dt_multiple_depths.append(mae_validation_dt_multiple_depths)
        rmses_train_dt_multiple_depths.append(rmse_train_dt_multiple_depths)
        rmses_validation_dt_multiple_depths.append(rmse_validation_dt_multiple_depths)
        n_leaves.append(model_dt_multiple_depths.get_n_leaves())

    pd.DataFrame(
        {
            "depth": _depths,
            "n_leaves": n_leaves,
            "MAE_train": maes_train_dt_multiple_depths,
            "MAE_validation": maes_validation_dt_multiple_depths,
            "RMSE_train": rmses_train_dt_multiple_depths,
            "RMSE_validation": rmses_validation_dt_multiple_depths,
        }
    ).round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Validation performance initially improves as tree depth increases, showing that the shallowest trees underfit the data. The best validation MAE in the explored range is obtained around a depth of 10, while deeper trees continue to reduce training error without improving validation performance.

    Beyond this region, the growing train-validation gap indicates progressively stronger overfitting. A depth of 10 is therefore retained as a representative Decision tree baseline, achieving approximately 10.6 cycles MAE on the validation set.

    Although this is already an improvement over the linear models, the relatively high validation RMSE indicates that the single tree still produces substantial large prediction errors.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.2 Random Forest

    A Random Forest combines multiple decision trees trained on different bootstrap samples and random subsets of predictors. Their predictions are averaged, which generally reduces the variance and instability observed with a single decision tree.

    A first representative configuration with 100 trees and a maximum depth of 10 is evaluated. Detailed hyperparameter tuning is intentionally deferred to the later model-selection stage.
    """)
    return


@app.cell
def _(
    RandomForestRegressor,
    X_train_raw_temporal_cycle,
    X_validation_temporal_cycle,
    mean_absolute_error,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    model_rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )

    model_rf.fit(X_train_raw_temporal_cycle, y_train)

    y_pred_train_rf = model_rf.predict(X_train_raw_temporal_cycle)
    y_pred_validation_rf = model_rf.predict(X_validation_temporal_cycle)

    mae_train_rf = mean_absolute_error(y_train, y_pred_train_rf)
    mae_validation_rf = mean_absolute_error(y_validation, y_pred_validation_rf)

    rmse_train_rf = root_mean_squared_error(y_train, y_pred_train_rf)
    rmse_validation_rf = root_mean_squared_error(y_validation, y_pred_validation_rf)

    f"MAE_train : {mae_train_rf}, MAE_validation {mae_validation_rf}, RMSE_train : {rmse_train_rf}, RMSE_validation : {rmse_validation_rf}"
    return mae_train_rf, mae_validation_rf, rmse_validation_rf


@app.cell
def _(mo):
    mo.md(r"""
    Random Forest substantially improves over the single Decision Tree, reducing validation MAE from approximately 10.6 to 8.8 cycles and RMSE from 16.7 to 13.3 cycles.

    Training error remains lower than validation error, indicating some degree of overfitting, but the validation improvement confirms that averaging multiple trees considerably improves generalisation.

    Random Forest therefore represents the first nonlinear model family to clearly outperform all previous linear baselines.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.3 Extra Trees

    Extra Trees follows a similar ensemble principle to Random Forest but introduces additional randomness when selecting split thresholds. This stronger randomisation can further decorrelate the individual trees and reduce ensemble variance.

    To allow a direct comparison, the same number of trees and maximum depth used for the Random Forest baseline are retained.
    """)
    return


@app.cell
def _(
    ExtraTreesRegressor,
    X_train_raw_temporal_cycle,
    X_validation_temporal_cycle,
    mean_absolute_error,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    model_et = ExtraTreesRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )

    model_et.fit(X_train_raw_temporal_cycle, y_train)

    y_pred_train_et = model_et.predict(X_train_raw_temporal_cycle)
    y_pred_validation_et = model_et.predict(X_validation_temporal_cycle)

    mae_train_et = mean_absolute_error(y_train, y_pred_train_et)
    mae_validation_et = mean_absolute_error(y_validation, y_pred_validation_et)

    rmse_train_et = root_mean_squared_error(y_train, y_pred_train_et)
    rmse_validation_et = root_mean_squared_error(y_validation, y_pred_validation_et)

    f"MAE_train : {mae_train_et}, MAE_validation {mae_validation_et}, RMSE_train : {rmse_train_et}, RMSE_validation : {rmse_validation_et}"
    return mae_train_et, mae_validation_et, rmse_validation_et


@app.cell
def _(mo):
    mo.md(r"""
    Extra Trees achieves a validation MAE of approximately 8.7 cycles and an RMSE of 12.9 cycles, slightly improving over Random Forest on both metrics.

    Its training error is also higher than that of Random Forest while validation performance is slightly better, suggesting that the additional randomisation is consistent with reduced overfitting without sacrificing predictive accuracy..

    Extra Trees therefore becomes one of the strongest baseline candidates identified so far.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.4 Gradient Boosting

    Random Forest and Extra Trees reduce prediction variance by averaging independently constructed trees. Gradient Boosting follows a different strategy. Trees are added sequentially, with each new tree attempting to correct errors left by the current ensemble.

    Three parameters are explored at a coarse level:
    - tree depth, controlling the complexity of each individual learner,
    - learning rate, controlling the contribution of each new tree,
    - number of estimators, controlling the number of sequential correction stages.
    """)
    return


@app.cell
def _(
    GradientBoostingRegressor,
    X_train_raw_temporal_cycle,
    X_validation_temporal_cycle,
    mean_absolute_error,
    pd,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    list_learning_rate = [0.03, 0.05, 0.1, 0.2]
    list_n_estimators = [400, 200, 100, 50]
    _max_depths = [1, 2, 3, 4, 5]

    maes_train_gb = []
    maes_validation_gb = []
    rmses_train_gb = []
    rmses_validation_gb = []
    list_dataframe_learning_rate = []
    list_dataframe_n_estimators = []
    list_dataframe_depth = []

    hyperparameters_couples = list(zip(list_learning_rate, list_n_estimators))

    for _depth in _max_depths:
        for _learning_rate, _n_estimators in hyperparameters_couples:
            model_gb = GradientBoostingRegressor(
                n_estimators=_n_estimators,
                learning_rate=_learning_rate,
                max_depth=_depth,
                random_state=42,
            )

            model_gb.fit(X_train_raw_temporal_cycle, y_train)

            y_pred_train_gb = model_gb.predict(X_train_raw_temporal_cycle)
            y_pred_validation_gb = model_gb.predict(X_validation_temporal_cycle)

            mae_train_gb = mean_absolute_error(y_train, y_pred_train_gb)
            mae_validation_gb = mean_absolute_error(y_validation, y_pred_validation_gb)

            rmse_train_gb = root_mean_squared_error(y_train, y_pred_train_gb)
            rmse_validation_gb = root_mean_squared_error(
                y_validation, y_pred_validation_gb
            )

            maes_train_gb.append(mae_train_gb)
            maes_validation_gb.append(mae_validation_gb)
            rmses_train_gb.append(rmse_train_gb)
            rmses_validation_gb.append(rmse_validation_gb)

            list_dataframe_learning_rate.append(_learning_rate)
            list_dataframe_n_estimators.append(_n_estimators)
            list_dataframe_depth.append(_depth)

    pd.DataFrame(
        {
            "depth": list_dataframe_depth,
            "learning_rate": list_dataframe_learning_rate,
            "n_estimators": list_dataframe_n_estimators,
            "MAE_train": maes_train_gb,
            "MAE_validation": maes_validation_gb,
            "RMSE_train": rmses_train_gb,
            "RMSE_validation": rmses_validation_gb,
        }
    ).round(3)
    return mae_train_gb, mae_validation_gb, rmse_validation_gb


@app.cell
def _(mo):
    mo.md(r"""
    Gradient Boosting provides another substantial improvement over the linear and single-tree baselines.

    Across the explored configurations, lower learning rates combined with a larger number of estimators generally provide stronger validation performance. Increasing tree depth improves MAE up to the deepest values explored, but also progressively increases the difference between training and validation error.

    The lowest validation MAE is obtained with a maximum depth of 5, learning rate of 0.03 and 400 estimators, reaching approximately 8.67 cycles. However, slightly shallower configurations achieve comparable RMSE with a smaller train-validation gap.

    These results show that Gradient Boosting is highly competitive with Extra Trees, but they do not justify selecting a final configuration yet. More systematic model selection using group-based cross-validation is deferred to the next notebook.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Support Vector Regression

    The final baseline family considered in this notebook is Support Vector Regression with a radial basis function (RBF) kernel.

    Unlike the tree-based approaches, RBF SVR models nonlinear relationships through similarities between observations in feature space. Because these similarities depend on distances between predictors, feature standardisation is required and is therefore included within a pipeline.

    Three hyperparameters primarily control the behaviour of the model: `epsilon`, which defines the width of the error-insensitive region. `C`, which controls the penalty applied to prediction errors outside this region. `gamma`, which controls the locality of the RBF kernel.

    To limit computational cost and preserve the exploratory objective of this notebook, `gamma='scale'` is retained while `epsilon` and `C` are examined separately.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.1 Epsilon sensitivity

    The regularisation parameter `C` is initially fixed at 10 while several values of `epsilon` are compared. Smaller values impose a narrower tolerance around the regression function, whereas larger values allow greater prediction deviations before contributing to the optimisation objective.
    """)
    return


@app.cell
def _(
    Pipeline,
    SVR,
    StandardScaler,
    X_train_raw_temporal_cycle,
    X_validation_temporal_cycle,
    mean_absolute_error,
    pd,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    list_epsilons = [0.1, 0.5, 1]

    _maes_SVR_train = []
    _maes_SVR_validation = []
    _rmses_SVR_train = []
    _rmses_SVR_validation = []

    for epsilons in list_epsilons:
        _model_SVR = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=10, epsilon=epsilons, gamma="scale")),
            ]
        )

        _model_SVR.fit(X_train_raw_temporal_cycle, y_train)

        _y_pred_train_SVR = _model_SVR.predict(X_train_raw_temporal_cycle)
        _y_pred_validation_SVR = _model_SVR.predict(X_validation_temporal_cycle)

        _mae_SVR_train = mean_absolute_error(y_train, _y_pred_train_SVR)
        _mae_SVR_validation = mean_absolute_error(y_validation, _y_pred_validation_SVR)

        _rmse_SVR_train = root_mean_squared_error(y_train, _y_pred_train_SVR)
        _rmse_SVR_validation = root_mean_squared_error(
            y_validation, _y_pred_validation_SVR
        )

        _maes_SVR_train.append(_mae_SVR_train)
        _maes_SVR_validation.append(_mae_SVR_validation)

        _rmses_SVR_train.append(_rmse_SVR_train)
        _rmses_SVR_validation.append(_rmse_SVR_validation)

    pd.DataFrame(
        {
            "epsilon": list_epsilons,
            "mae_train": _maes_SVR_train,
            "mae_validation": _maes_SVR_validation,
            "rmse_train": _rmses_SVR_train,
            "rmse_validation": _rmses_SVR_validation,
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Validation MAE changes relatively little between `epsilon=0.1`, `0.5` and `1`, while RMSE is lowest around `epsilon=0.5`.

    The very small MAE advantage obtained with `epsilon=0.1` is not accompanied by an improvement in RMSE, suggesting slightly larger extreme errors. An exploratory value of `epsilon=0.5` is therefore retained as a balanced compromise for the following comparison of `C`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.2 Regularisation strength

    With `epsilon=0.5` fixed, the penalty parameter `C` is varied across several orders of magnitude.

    A low value of `C` allows larger deviations from the training observations and therefore produces stronger effective regularisation. Increasing `C` penalises these deviations more strongly and allows the model to fit the training data more closely.
    """)
    return


@app.cell
def _(
    Pipeline,
    SVR,
    StandardScaler,
    X_train_raw_temporal_cycle,
    X_validation_temporal_cycle,
    mean_absolute_error,
    pd,
    root_mean_squared_error,
    y_train,
    y_validation,
):
    list_C = [1, 10, 100]

    _maes_SVR_train = []
    _maes_SVR_validation = []
    _rmses_SVR_train = []
    _rmses_SVR_validation = []

    for c in list_C:
        _model_SVR = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=c, epsilon=0.5, gamma="scale")),
            ]
        )

        _model_SVR.fit(X_train_raw_temporal_cycle, y_train)

        _y_pred_train_SVR = _model_SVR.predict(X_train_raw_temporal_cycle)
        _y_pred_validation_SVR = _model_SVR.predict(X_validation_temporal_cycle)

        _mae_SVR_train = mean_absolute_error(y_train, _y_pred_train_SVR)
        _mae_SVR_validation = mean_absolute_error(y_validation, _y_pred_validation_SVR)

        _rmse_SVR_train = root_mean_squared_error(y_train, _y_pred_train_SVR)
        _rmse_SVR_validation = root_mean_squared_error(
            y_validation, _y_pred_validation_SVR
        )

        _maes_SVR_train.append(_mae_SVR_train)
        _maes_SVR_validation.append(_mae_SVR_validation)

        _rmses_SVR_train.append(_rmse_SVR_train)
        _rmses_SVR_validation.append(_rmse_SVR_validation)

    pd.DataFrame(
        {
            "C": list_C,
            "mae_train": _maes_SVR_train,
            "mae_validation": _maes_SVR_validation,
            "rmse_train": _rmses_SVR_train,
            "rmse_validation": _rmses_SVR_validation,
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    The effect of `C` is clearly visible. With `C=1`, both training and validation errors remain relatively high, indicating underfitting. Increasing the parameter to `C=10` substantially improves both datasets and produces closely matched training and validation performance.

    At `C=100`, training MAE falls sharply to approximately 5.6 cycles while validation MAE deteriorates to approximately 9.8 cycles. The widening gap indicates that the additional flexibility primarily improves the fit to the training observations rather than generalisation.

    The exploratory configuration `C=10`,`epsilon=0.5`, and `gamma='scale'` therefore provides the strongest compromise, reaching approximately 9.54 cycles validation MAE and 13.36 cycles RMSE.

    SVR clearly outperforms the linear models and the single Decision Tree, although it remains behind the strongest tree ensembles in the current experiments.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Baseline model comparison

    The representative results from all model families are gathered below. For models where several exploratory configurations were evaluated, a single representative configuration is retained based primarily on validation MAE, while RMSE and train-validation behaviour are used as complementary indicators.

    The purpose of this comparison is to identify promising model families for subsequent cross-validated model selection rather than to declare a final model from the fixed validation split.
    """)
    return


@app.cell
def _(
    mae_baseline_mean,
    mae_baseline_median,
    mae_cycle_only,
    mae_lasso,
    mae_non_zero,
    mae_raw_sensors,
    mae_raw_temporal,
    mae_raw_temporal_cycle,
    mae_ridge,
    mae_sensors_cycle,
    mae_train_dt,
    mae_train_et,
    mae_train_gb,
    mae_train_rf,
    mae_validation_dt,
    mae_validation_et,
    mae_validation_gb,
    mae_validation_rf,
    pd,
    rmse_baseline_mean,
    rmse_baseline_median,
    rmse_cycle_only,
    rmse_lasso,
    rmse_non_zero,
    rmse_raw_sensors,
    rmse_raw_temporal,
    rmse_raw_temporal_cycle,
    rmse_ridge,
    rmse_sensors_cycle,
    rmse_validation_dt,
    rmse_validation_et,
    rmse_validation_gb,
    rmse_validation_rf,
):
    baseline_results = [
        {
            "model": "Constant median",
            "feature": "None",
            "configuration": "median = 103",
            "mae_train": "---",
            "mae_validation": mae_baseline_median,
            "rmse_validation": rmse_baseline_median,
        },
        {
            "model": "Constant mean",
            "feature": "None",
            "configuration": "mean = 86.96",
            "mae_train": "---",
            "mae_validation": mae_baseline_mean,
            "rmse_validation": rmse_baseline_mean,
        },
        {
            "model": "Linear Regression",
            "feature": "Cycle only",
            "configuration": "---",
            "mae_train": "---",
            "mae_validation": mae_cycle_only,
            "rmse_validation": rmse_cycle_only,
        },
        {
            "model": "Linear Regression",
            "feature": "Raw sensors",
            "configuration": "---",
            "mae_train": "---",
            "mae_validation": mae_raw_sensors,
            "rmse_validation": rmse_raw_sensors,
        },
        {
            "model": "Linear Regression",
            "feature": "Raw + cycle",
            "configuration": "---",
            "mae_train": "---",
            "mae_validation": mae_sensors_cycle,
            "rmse_validation": rmse_sensors_cycle,
        },
        {
            "model": "Linear Regression",
            "feature": "Raw + temporal",
            "configuration": "---",
            "mae_train": "---",
            "mae_validation": mae_raw_temporal,
            "rmse_validation": rmse_raw_temporal,
        },
        {
            "model": "Linear Regression",
            "feature": "Raw + temporal + cycle",
            "configuration": "---",
            "mae_train": "---",
            "mae_validation": mae_raw_temporal_cycle,
            "rmse_validation": rmse_raw_temporal_cycle,
        },
        {
            "model": "Ridge",
            "feature": "Raw + temporal + cycle",
            "configuration": "α = 10",
            "mae_train": "---",
            "mae_validation": mae_ridge,
            "rmse_validation": rmse_ridge,
        },
        {
            "model": "Lasso",
            "feature": "Raw + temporal + cycle",
            "configuration": "α = 0.03",
            "mae_train": "---",
            "mae_validation": mae_lasso,
            "rmse_validation": rmse_lasso,
        },
        {
            "model": "Linear Regression",
            "feature": "Lasso-selected features",
            "configuration": "88 features",
            "mae_train": "---",
            "mae_validation": mae_non_zero,
            "rmse_validation": rmse_non_zero,
        },
        {
            "model": "Decision Tree",
            "feature": "Raw + temporal + cycle",
            "configuration": "depth = 10",
            "mae_train": mae_train_dt,
            "mae_validation": mae_validation_dt,
            "rmse_validation": rmse_validation_dt,
        },
        {
            "model": "Random Forest",
            "feature": "Raw + temporal + cycle",
            "configuration": "depth = 10, 100 trees",
            "mae_train": mae_train_rf,
            "mae_validation": mae_validation_rf,
            "rmse_validation": rmse_validation_rf,
        },
        {
            "model": "Extra Trees",
            "feature": "Raw + temporal + cycle",
            "configuration": "depth = 10, 100 trees",
            "mae_train": mae_train_et,
            "mae_validation": mae_validation_et,
            "rmse_validation": rmse_validation_et,
        },
        {
            "model": "Gradient Boosting",
            "feature": "Raw + temporal + cycle",
            "configuration": "depth = 5, lr = 0.03, 400 trees",
            "mae_train": mae_train_gb,
            "mae_validation": mae_validation_gb,
            "rmse_validation": rmse_validation_gb,
        },
        {
            "model": "SVR RBF",
            "feature": "Raw + temporal + cycle",
            "configuration": "C = 10, ε = 0.5",
            "mae_train": 9.27,
            "mae_validation": 9.54,
            "rmse_validation": 13.36,
        },
    ]

    pd.DataFrame(baseline_results)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The progression across experiments reveals two main findings.

    First, feature engineering provides substantial gains even within a simple linear model. Moving from operating cycle alone to raw sensor measurements, temporal features, and operating age progressively reduces validation MAE from approximately 20.5 to 12.9 cycles. This confirms that engine age, instantaneous condition, and recent sensor dynamics carry complementary predictive information.

    Second, model nonlinearity provides an additional major improvement. Regularised linear models remain close to the ordinary linear-regression baseline, whereas ensemble tree methods reduce validation MAE to approximately 8.7-8.8 cycles. SVR also benefits from nonlinear modelling but remains moderately behind the strongest ensembles.

    Gradient Boosting achieves the lowest validation MAE in the current exploratory experiments, while Extra Trees obtains an almost identical MAE and a slightly lower RMSE. Random Forest also remains competitive. The differences between these ensemble methods are sufficiently small that selecting a final model from this single validation split would be premature.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Conclusions and models retained for further study

    This notebook established a sequence of increasingly expressive RUL prediction baselines on FD001 while preserving the same engine-level train/validation split and leaving the official test set untouched.

    The experiments first confirmed the value of the engineered feature representation. Raw sensor measurements outperform operating cycle alone, temporal features provide additional predictive information, and the strongest linear set combines raw sensors, temporal dynamics and absolute operating age.

    Ridge regularisation reduces coefficient magnitude without providing a meaningful predictive gain, while Lasso demonstrates that a subset of predictors can be removed with almost no loss in linear-regression performance. This suggests substantial redundancy among the engineered features but does not justify permanently removing them before nonlinear modelling.

    Nonlinear models provide the largest performance improvement. A single Decision Tree improves over the linear baselines but exhibits substantial overfitting as depth increases. Random Forest and Extra Trees significantly reduce validation error through ensemble averaging, while Gradient Boosting reaches the strongest MAE observed in this notebook. RBF SVR also provides competitive nonlinear performance, although it remains behind the strongest tree ensembles.

    Based on these results, Gradient Boosting, Extra Trees and Random Forest are retained as the primary candidates for systematic model selection. SVR may also be retained as a secondary candidate because it represents a substantially different nonlinear modelling approach.

    The next notebook will replace the single validation comparison with group-based cross-validation, ensuring that complete engines remain isolated between folds. Hyperparameter optimisation, temporal-window comparison, sensor and feature-group ablations, and final feature selection will then be performed before any evaluation on the official FD001 test set.
    """)
    return


if __name__ == "__main__":
    app.run()
