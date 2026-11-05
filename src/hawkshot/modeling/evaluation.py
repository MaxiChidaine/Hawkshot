from time import perf_counter
import pandas as pd
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


def analyze_oof_errors_by_rul(
    oof_df: pd.DataFrame,
    model_id: str,
) -> pd.DataFrame:
    """Analyze OOF prediction errors by true RUL intervals of 10 cycles."""

    rul_bins = list(range(0, 121, 10)) + [125, 126]

    rul_labels = [f"{start}-{start + 9}" for start in range(0, 120, 10)] + [
        "120-124",
        "125_capped",
    ]

    df = oof_df.copy()

    df["rul_band"] = pd.cut(
        df["y_true"],
        bins=rul_bins,
        labels=rul_labels,
        right=False,
    )

    if df["rul_band"].isna().any():
        raise ValueError("Some RUL values fall outside the expected range [0, 125].")

    results = []

    for band, group in df.groupby("rul_band", observed=True):
        scores = summarize_oof_predictions(
            group,
            model_id=model_id,
        )

        absolute_errors = (group["y_pred"] - group["y_true"]).abs()

        signed_errors = group["y_pred"] - group["y_true"]

        underestimations = -signed_errors[signed_errors < 0]

        results.append(
            {
                "rul_band": str(band),
                **scores,
                "p90_abs_error": absolute_errors.quantile(0.90),
                "overestimation_gt_10_rate": (signed_errors > 10).mean(),
                "overestimation_gt_20_rate": (signed_errors > 20).mean(),
                "p90_signed_error": signed_errors.quantile(0.90),
                "underestimation_rate": (signed_errors < 0).mean(),
                "mean_underestimation": underestimations.mean(),
                "underestimation_gt_20_rate": (signed_errors < -20).mean(),
            }
        )

    return pd.DataFrame(results)


def analyze_oof_errors_by_engine(
    oof_df: pd.DataFrame,
    model_id: str,
    rul_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Analyze OOF prediction errors for each engine.

    Parameters
    ----------
    oof_df : pd.DataFrame
        OOF predictions with engine_id, cycle, y_true and y_pred.
    model_id : str
        Identifier of the evaluated model.
    rul_range : tuple[int, int] | None
        Optional inclusive true-RUL interval.

    Returns
    -------
    pd.DataFrame
        Error metrics and large-overestimation streaks per engine.
    """
    required = {"engine_id", "cycle", "y_true", "y_pred"}

    if missing := required - set(oof_df.columns):
        raise ValueError(f"Missing OOF columns: {missing}")

    df = oof_df

    if rul_range is not None:
        lower, upper = rul_range

        if lower > upper:
            raise ValueError("Invalid RUL interval.")

        df = df.loc[df["y_true"].between(lower, upper)]

    if df.empty:
        raise ValueError("No observations in the selected RUL interval.")

    if df.duplicated(subset=["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle observations.")

    results = []

    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")

        signed_errors = group["y_pred"] - group["y_true"]
        large_overestimations = signed_errors > 20

        longest_streak = 0
        current_streak = 0
        previous_cycle = None

        for cycle, is_large in zip(
            group["cycle"],
            large_overestimations,
            strict=True,
        ):
            if is_large:
                if previous_cycle is not None and cycle == previous_cycle + 1:
                    current_streak += 1
                else:
                    current_streak = 1

                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 0

            previous_cycle = cycle

        results.append(
            {
                "Model": model_id,
                "engine_id": engine_id,
                "n_observations": len(group),
                "MAE": signed_errors.abs().mean(),
                "RMSE": signed_errors.pow(2).mean() ** 0.5,
                "overestimation_rate": (signed_errors > 0).mean(),
                "max_overestimation": max(0.0, signed_errors.max()),
                "n_overestimations_gt_20": int(large_overestimations.sum()),
                "longest_overestimation_gt_20_streak": longest_streak,
            }
        )

    return pd.DataFrame(results)
