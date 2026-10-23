from types import SimpleNamespace

import pytest

from hawkshot.modeling.search_results import format_grid_search_results


def test_grid_search_results_metric_signs():
    search = SimpleNamespace(
        cv_results_={
            "params": [{"alpha": 0.1}],
            "mean_test_MAE": [-10.0],
            "mean_test_RMSE": [-12.0],
            "mean_test_overestimation_rate": [-0.25],
            "mean_test_mean_overestimation": [-5.0],
            "mean_test_mean_signed_error": [2.0],
        }
    )

    results = format_grid_search_results(search)

    assert results.loc[0, "MAE"] == pytest.approx(10.0)
    assert results.loc[0, "RMSE"] == pytest.approx(12.0)
    assert results.loc[0, "overestimation_rate"] == pytest.approx(0.25)
    assert results.loc[0, "mean_overestimation"] == pytest.approx(5.0)
    assert results.loc[0, "mean_signed_error"] == pytest.approx(2.0)

    # Original GridSearchCV scores must remain available.
    assert results.loc[0, "mean_test_MAE"] == -10.0


def test_grid_search_results_handles_missing_optional_metrics():
    search = SimpleNamespace(
        cv_results_={
            "params": [{"alpha": 0.1}],
            "mean_test_MAE": [-10.0],
        }
    )

    results = format_grid_search_results(search)

    assert results.loc[0, "MAE"] == 10.0
    assert "RMSE" not in results.columns
    assert "mean_overestimation" not in results.columns
