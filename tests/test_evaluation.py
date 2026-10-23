import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold

from hawkshot.modeling.evaluation import evaluate_grouped_cv


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
