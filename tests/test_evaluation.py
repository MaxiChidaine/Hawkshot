import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold

from hawkshot.modeling.evaluation import (
    evaluate_grouped_cv,
    summarize_oof_predictions,
    analyze_oof_errors_by_rul,
    analyze_oof_errors_by_engine,
)


@pytest.fixture
def sample_data():
    X = pd.DataFrame(
        {
            "sensor_2": [10.0, 20.0] * 6,
        }
    )

    y = pd.Series([2.0, 4.0] * 6)

    groups = pd.Series(
        [
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            5,
            6,
            6,
        ]
    )

    cv = GroupKFold(n_splits=3)

    return X, y, groups, cv


def test_evaluation_metrics_are_correct(sample_data):
    X, y, groups, cv = sample_data

    model = DummyRegressor(
        strategy="constant",
        constant=3.0,
    )

    results = evaluate_grouped_cv(model, X, y, groups, cv)

    assert results["mean_MAE"] == pytest.approx(1.0)
    assert results["mean_RMSE"] == pytest.approx(1.0)
    assert results["std_MAE"] == pytest.approx(0.0)
    assert results["std_RMSE"] == pytest.approx(0.0)

    assert results["overestimation_rate"] == pytest.approx(0.5)
    assert results["mean_signed_error"] == pytest.approx(0.0)
    assert results["mean_overestimation"] == pytest.approx(1.0)


def test_evaluation_handles_no_overestimation(sample_data):
    X, _, groups, cv = sample_data

    y = pd.Series([4.0] * len(X))

    model = DummyRegressor(
        strategy="constant",
        constant=3.0,
    )

    results = evaluate_grouped_cv(model, X, y, groups, cv)

    assert results["mean_MAE"] == pytest.approx(1.0)
    assert results["overestimation_rate"] == pytest.approx(0.0)
    assert results["mean_signed_error"] == pytest.approx(-1.0)
    assert results["mean_overestimation"] == pytest.approx(0.0)


class GroupIsolationRegressor(BaseEstimator, RegressorMixin):
    fitted_models = []

    def fit(self, X, y):
        self.training_engines_ = set(X["engine_id"])
        self.fitted_models.append(self)
        return self

    def predict(self, X):
        validation_engines = set(X["engine_id"])

        assert self.training_engines_.isdisjoint(validation_engines), (
            "Engine leakage detected: training and validation sets share engine IDs."
        )

        return [0.0] * len(X)


def test_evaluation_clones_models_and_separates_engines():
    GroupIsolationRegressor.fitted_models = []

    groups = pd.Series(
        [
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            5,
            6,
            6,
        ]
    )

    X = pd.DataFrame(
        {
            "engine_id": groups,
            "sensor_2": [10.0, 20.0] * 6,
        }
    )

    y = pd.Series([2.0, 4.0] * 6)
    cv = GroupKFold(n_splits=3)

    model = GroupIsolationRegressor()

    evaluate_grouped_cv(model, X, y, groups, cv)

    fitted = GroupIsolationRegressor.fitted_models

    assert len(fitted) == 3
    assert all(fold_model is not model for fold_model in fitted)
    assert len({id(fold_model) for fold_model in fitted}) == 3

    assert not hasattr(model, "training_engines_")


def test_summaryze_oof_predictions():
    oof_df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2],
            "y_true": [10.0, 20.0, 30.0],
            "y_pred": [12.0, 18.0, 31.0],
        }
    )

    scores = summarize_oof_predictions(
        oof_df,
        model_id="test_model",
    )

    assert scores["Model"] == "test_model"
    assert scores["MAE"] == pytest.approx(5 / 3)
    assert scores["RMSE"] == pytest.approx(3**0.5)
    assert scores["overestimation_rate"] == pytest.approx(2 / 3)
    assert scores["mean_signed_error"] == pytest.approx(1 / 3)
    assert scores["n_engines"] == 2
    assert scores["n_observations"] == 3


def test_analyze_oof_errors_by_rul():
    oof_df = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2],
            "y_true": [9, 10, 124, 125],
            "y_pred": [11, 12, 122, 120],
        }
    )

    results = analyze_oof_errors_by_rul(
        oof_df,
        model_id="test_model",
    )

    assert results["rul_band"].tolist() == [
        "0-9",
        "10-19",
        "120-124",
        "125_capped",
    ]

    first_band = results.set_index("rul_band").loc["0-9"]

    assert first_band["MAE"] == pytest.approx(2.0)
    assert first_band["p90_abs_error"] == pytest.approx(2.0)


def test_analyze_oof_large_overestimations():
    oof_df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1],
            "y_true": [20, 21, 22, 23, 24],
            "y_pred": [20, 31, 34, 44, 50],
        }
    )

    results = analyze_oof_errors_by_rul(
        oof_df,
        model_id="test_model",
    )

    band = results.iloc[0]

    assert band["rul_band"] == "20-29"
    assert band["overestimation_gt_10_rate"] == pytest.approx(0.6)
    assert band["overestimation_gt_20_rate"] == pytest.approx(0.4)
    assert band["p90_signed_error"] == pytest.approx(24.0)


def test_analyze_oof_errors_by_engine():
    oof_df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 2, 2],
            "cycle": [1, 2, 4, 5, 1, 2],
            "y_true": [50, 49, 47, 46, 30, 29],
            "y_pred": [75, 79, 71, 41, 40, 24],
        }
    )

    results = analyze_oof_errors_by_engine(
        oof_df,
        model_id="test_model",
    ).set_index("engine_id")

    engine_1 = results.loc[1]
    engine_2 = results.loc[2]

    assert engine_1["MAE"] == pytest.approx(21.0)
    assert engine_1["max_overestimation"] == pytest.approx(30.0)
    assert engine_1["n_overestimations_gt_20"] == 3
    assert engine_1["longest_overestimation_gt_20_streak"] == 2

    assert engine_2["n_overestimations_gt_20"] == 0
    assert engine_2["longest_overestimation_gt_20_streak"] == 0


def test_analyze_oof_underestimations():
    oof_df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1],
            "y_true": [20, 21, 22, 23, 24],
            "y_pred": [20, 31, 10, 2, 26],
        }
    )

    results = analyze_oof_errors_by_rul(
        oof_df,
        model_id="test_model",
    )

    band = results.iloc[0]

    assert band["rul_band"] == "20-29"
    assert band["underestimation_rate"] == pytest.approx(0.4)
    assert band["mean_underestimation"] == pytest.approx(16.5)
    assert band["underestimation_gt_20_rate"] == pytest.approx(0.2)
