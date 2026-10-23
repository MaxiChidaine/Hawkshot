import pandas as pd
import pytest

from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    ParameterGrid,
)

from hawkshot.experiments.checkpoints import (
    inspect_checkpoint,
    write_checkpoint,
    CheckpointConflictError,
)

from hawkshot.experiments.provenance import (
    build_experiment_manifest,
    materialize_cv_splits,
)

from hawkshot.experiments.runner import run_or_reuse_grid_search


def test_ridge_checkpoint_end_to_end(tmp_path):
    X = pd.DataFrame(
        {
            "sensor_2": [10.0, 12.0] * 4,
        }
    )
    y = pd.Series([20.0, 18.0] * 4)
    groups = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])

    model = Ridge()
    param_grid = {"alpha": [0.1, 1.0]}
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
    }

    splits = materialize_cv_splits(
        GroupKFold(n_splits=2),
        X,
        y,
        groups,
    )

    manifest = build_experiment_manifest(
        model=model,
        param_grid=param_grid,
        X=X,
        y=y,
        groups=groups,
        cv=splits,
        scoring_version="test_mae_rmse_v1",
        refit="MAE",
        search_n_jobs=1,
    )

    checkpoint_path = tmp_path / "ridge_test.csv"
    expected_rows = len(ParameterGrid(param_grid))

    fit_calls = 0

    def run_or_reuse():
        nonlocal fit_calls

        existing = inspect_checkpoint(
            checkpoint_path,
            experiment=manifest,
            expected_rows=expected_rows,
        )

        if existing is not None:
            return existing

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            refit="MAE",
            cv=splits,
            n_jobs=1,
        )

        fit_calls += 1
        search.fit(X, y, groups=groups)

        results = pd.DataFrame(search.cv_results_)

        write_checkpoint(
            checkpoint_path,
            experiment=manifest,
            results=results,
            expected_rows=expected_rows,
        )

        return results

    first_results = run_or_reuse()

    assert fit_calls == 1
    assert checkpoint_path.is_file()
    assert checkpoint_path.with_suffix(".meta.json").is_file()
    assert len(first_results) == expected_rows

    second_results = run_or_reuse()

    assert fit_calls == 1
    assert len(second_results) == expected_rows

    assert second_results["mean_test_MAE"].to_list() == (
        pytest.approx(first_results["mean_test_MAE"].to_list())
    )


def test_runner_reuses_checkpoint_and_rejects_conflicts(tmp_path, monkeypatch):
    X = pd.DataFrame(
        {
            "sensor_2": [10.0, 12.0] * 4,
        }
    )
    y = pd.Series([20.0, 18.0] * 4)
    groups = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])

    splits = materialize_cv_splits(
        GroupKFold(n_splits=2),
        X,
        y,
        groups,
    )

    inputs = {
        "model": Ridge(),
        "param_grid": {"alpha": [0.1, 1.0]},
        "X": X,
        "y": y,
        "groups": groups,
        "cv_splits": splits,
        "scoring": {"MAE": "neg_mean_absolute_error"},
        "scoring_version": "test-mae-v1",
        "checkpoint_path": tmp_path / "ridge.csv",
        "n_jobs": 1,
    }

    original_fit = GridSearchCV.fit
    fit_calls = 0

    def counting_fit(self, *args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(GridSearchCV, "fit", counting_fit)

    first = run_or_reuse_grid_search(**inputs)

    assert fit_calls == 1
    assert "MAE" in first.columns

    second = run_or_reuse_grid_search(**inputs)

    assert fit_calls == 1
    assert second["MAE"].tolist() == pytest.approx(first["MAE"].tolist())

    conflicting_inputs = {
        **inputs,
        "param_grid": {"alpha": [0.5, 2.0]},
    }

    with pytest.raises(CheckpointConflictError):
        run_or_reuse_grid_search(**conflicting_inputs)

    assert fit_calls == 1
