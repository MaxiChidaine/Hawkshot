import numpy as np
import pytest
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, RegressorMixin

from hawkshot.modeling.metrics import (
    overestimation_rate,
    mean_overestimation,
    mean_signed_error,
    mean_signed_error_scorer,
)


@pytest.fixture
def predictions():
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_pred = np.array([12.0, 9.0, 11.0, 10.0])
    return y_true, y_pred


def test_metrics_with_mixed_errors(predictions):
    y_true, y_pred = predictions

    assert overestimation_rate(y_true, y_pred) == pytest.approx(0.5)
    assert mean_signed_error(y_true, y_pred) == pytest.approx(0.5)
    assert mean_overestimation(y_true, y_pred) == pytest.approx(1.5)


def test_metrics_without_overestimation():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([8.0, 18.0, 28.0])

    assert overestimation_rate(y_true, y_pred) == pytest.approx(0.0)
    assert mean_signed_error(y_true, y_pred) == pytest.approx(-2.0)
    assert mean_overestimation(y_true, y_pred) == pytest.approx(0.0)


def test_perfect_predictions():
    y_true = np.array([10.0, 20.0])
    y_pred = y_true.copy()

    assert overestimation_rate(y_true, y_pred) == pytest.approx(0.0)
    assert mean_signed_error(y_true, y_pred) == pytest.approx(0.0)
    assert mean_overestimation(y_true, y_pred) == pytest.approx(0.0)


class FixedPredictor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([12.0, 9.0, 11.0, 10.0])


def test_gridsearch_scorer_conventions(predictions):
    y_true, _ = predictions
    X = np.zeros((4, 1))
    model = FixedPredictor()

    overestimation_scorer = make_scorer(
        overestimation_rate,
        greater_is_better=False,
    )

    # GridSearchCV receives the negative value.
    assert overestimation_scorer(model, X, y_true) == pytest.approx(-0.5)

    # Mean signed error remains signed, not negated.
    assert mean_signed_error_scorer(model, X, y_true) == pytest.approx(0.5)
