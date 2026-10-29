from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)

from hawkshot.modeling.metrics import (
    mean_overestimation,
    mean_signed_error,
    overestimation_rate,
)


def evaluate_grouped_cv(model, X, y, groups, cv):
    maes = []
    rmses = []
    fit_times = []
    predict_times = []
    mean_signed_errors = []
    overestimation_rates = []
    mean_overestimations = []

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

        overestimation_rates.append(overestimation_rate(y_validation_fold, y_pred))
        mean_signed_errors.append(mean_signed_error(y_validation_fold, y_pred))
        mean_overestimations.append(mean_overestimation(y_validation_fold, y_pred))

    return {
        "mean_MAE": np.mean(maes),
        "std_MAE": np.std(maes),
        "mean_RMSE": np.mean(rmses),
        "std_RMSE": np.std(rmses),
        "mean_fit_time": np.mean(fit_times),
        "mean_predict_time": np.mean(predict_times),
        "overestimation_rate": np.mean(overestimation_rates),
        "mean_signed_error": np.mean(mean_signed_errors),
        "mean_overestimation": np.mean(mean_overestimations),
    }


def summarize_oof_predictions(oof_df, model_id):
    y_true = oof_df["y_true"]
    y_pred = oof_df["y_pred"]

    return {
        "Model": model_id,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "overestimation_rate": overestimation_rate(y_true, y_pred),
        "mean_overestimation": mean_overestimation(y_true, y_pred),
        "mean_signed_error": (y_pred - y_true).mean(),
        "n_engines": oof_df["engine_id"].nunique(),
        "n_observations": len(oof_df),
    }
