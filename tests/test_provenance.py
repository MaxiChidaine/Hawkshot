import pandas as pd
import pytest
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.base import BaseEstimator


from hawkshot.experiments.provenance import (
    fingerprint_cv_splits,
    fingerprint_pandas,
    build_experiment_manifest,
    materialize_cv_splits,
)

from hawkshot.experiments.checkpoints import experiment_fingerprint
from hawkshot.modeling.objectives import make_asymmetric_mse


def test_identical_data_produces_same_fingerprints():
    X = pd.DataFrame(
        {
            "sensor_2": [10.0, 12.0, 14.0],
            "sensor_3": [20.0, 21.0, 22.0],
        }
    )

    assert fingerprint_pandas(X) == fingerprint_pandas(X.copy())


def test_modified_sensor_changes_fingerprint():
    X = pd.DataFrame(
        {
            "sensor_2": [10.0, 12.0, 14.0],
        }
    )

    X_modified = X.copy()
    X_modified.loc[1, "sensor_2"] = 12.1

    assert fingerprint_pandas(X) != fingerprint_pandas(X_modified)


def test_modified_feature_name_changes_fingerprint():
    X = pd.DataFrame(
        {
            "sensor_2": [10.0, 12.0, 14.0],
        }
    )

    X_renamed = X.rename(columns={"sensor_2": "sensor_3"})

    assert fingerprint_pandas(X) != fingerprint_pandas(X_renamed)


def test_modified_target_changes_fingerprint():
    y = pd.Series([125.0, 100.0, 75.0])

    y_modified = y.copy()
    y_modified.iloc[0] = 120

    assert fingerprint_pandas(y) != fingerprint_pandas(y_modified)


def test_cv_fingerprint_detect_different_splits():
    groups = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    X = pd.DataFrame({"sensor_2": range(12)})
    y = pd.Series(range(12))

    cv_3 = GroupKFold(n_splits=3)
    cv_2 = GroupKFold(n_splits=2)

    hash_3 = fingerprint_cv_splits(cv_3, X, y, groups)
    hash_2 = fingerprint_cv_splits(cv_2, X, y, groups)

    assert hash_3 != hash_2

    assert hash_3 == fingerprint_cv_splits(cv_3, X, y, groups)


@pytest.fixture
def experiment_input():
    X = pd.DataFrame(
        {
            "sensor_2": [10.0, 12.0] * 4,
        }
    )
    y = pd.Series([20.0, 18.0] * 4)
    groups = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])

    return {
        "model": Ridge(alpha=1.0),
        "param_grid": {"alpha": [0.1, 1.0]},
        "X": X,
        "y": y,
        "groups": groups,
        "cv": GroupKFold(n_splits=2),
        "scoring_version": "hawkshot-scoring-v1",
    }


def fingerprint_experiment(**kwargs):
    experiment = build_experiment_manifest(**kwargs)
    return experiment_fingerprint(experiment)


def test_identical_experiment_have_same_fingerprint(experiment_input):
    first = fingerprint_experiment(**experiment_input)
    second = fingerprint_experiment(**experiment_input)

    assert first == second


def test_model_parameters_change_fingerprint(experiment_input):
    original = fingerprint_experiment(**experiment_input)

    modified = fingerprint_experiment(**{**experiment_input, "model": Ridge(alpha=2.0)})

    assert original != modified


def test_same_size_different_grid_changes_fingerprint(experiment_input):
    original = fingerprint_experiment(**experiment_input)

    modified = fingerprint_experiment(
        **{
            **experiment_input,
            "param_grid": {"alpha": [0.5, 2.0]},
        }
    )

    assert original != modified


def test_modified_data_changes_experiment_fingerprint(experiment_input):
    original = fingerprint_experiment(**experiment_input)

    X_modified = experiment_input["X"].copy()
    X_modified.loc[0, "sensor_2"] = 99.0

    modified = fingerprint_experiment(
        **{
            **experiment_input,
            "X": X_modified,
        }
    )

    assert original != modified


def test_scoring_version_changes_fingerprint(experiment_input):
    original = fingerprint_experiment(**experiment_input)

    modified = fingerprint_experiment(
        **{**experiment_input, "scoring_version": "hawkshot_scoring-v2"},
    )

    assert original != modified


def test_custom_callable_requires_explicit_identity(experiment_input):
    modified_input = {
        **experiment_input,
        "param_grid": {"alpha": [lambda x: x]},
    }

    with pytest.raises(ValueError, match="JSON-serializable"):
        build_experiment_manifest(**modified_input)


class ObjectiveEstimator(BaseEstimator):
    def __init__(self, objective):
        self.objective = objective


def test_custom_objective_weight_changes_fingerprint(
    experiment_input,
):
    objective_a = make_asymmetric_mse(3.5)
    objective_b = make_asymmetric_mse(5.0)

    # Verify metadata immediately after objective creation.
    assert isinstance(
        getattr(objective_a, "__hawkshot_objective_spec__", None),
        dict,
    )

    model_a = ObjectiveEstimator(objective_a)
    model_b = ObjectiveEstimator(objective_b)

    # Verify that sklearn returns the original function.
    recovered_objective = model_a.get_params(deep=False)["objective"]

    assert recovered_objective is objective_a

    # Verify that metadata survived the estimator.
    assert isinstance(
        getattr(
            recovered_objective,
            "__hawkshot_objective_spec__",
            None,
        ),
        dict,
    )

    hash_a = fingerprint_experiment(**{**experiment_input, "model": model_a})
    hash_b = fingerprint_experiment(**{**experiment_input, "model": model_b})

    assert hash_a != hash_b


def test_custom_objective_in_grid_changes_fingerprint(experiment_input):
    objective_a = make_asymmetric_mse(3.5)
    objective_b = make_asymmetric_mse(5.0)

    model = ObjectiveEstimator("reg:squarederror")

    input_a = {
        **experiment_input,
        "model": model,
        "param_grid": {"objective": [objective_a]},
    }

    input_b = {
        **experiment_input,
        "model": model,
        "param_grid": {"objective": [objective_b]},
    }

    manifest_a = build_experiment_manifest(**input_a)
    manifest_b = build_experiment_manifest(**input_b)

    assert input_a["param_grid"]["objective"][0] is objective_a

    assert manifest_a["param_grid"]["objective"][0] == {
        "custom_objective": objective_a.__hawkshot_objective_spec__
    }

    assert experiment_fingerprint(manifest_a) != experiment_fingerprint(manifest_b)


def test_multiple_grid_parameters_change_fingerprint(experiment_input):
    grid_a = {
        "alpha": [0.1, 1.0],
        "solver": ["auto", "svd"],
    }

    grid_b = {
        "alpha": [0.5, 1.2],
        "solver": ["auto", "cholesky"],
    }

    hash_a = fingerprint_experiment(**{**experiment_input, "param_grid": grid_a})

    hash_b = fingerprint_experiment(**{**experiment_input, "param_grid": grid_b})

    assert hash_a != hash_b


def test_list_of_parameter_grids_changes_fingerprint(experiment_input):
    grids_a = [
        {
            "alpha": [0.1, 1.0],
        },
        {
            "alpha": [2.0],
            "solver": ["svd", "cholesky"],
        },
    ]

    grids_b = [
        {
            "alpha": [0.1, 1.0],
        },
        {
            "alpha": [2.0],
            "solver": ["svd", "lsqr"],
        },
    ]

    hash_a = fingerprint_experiment(**{**experiment_input, "param_grid": grids_a})

    hash_b = fingerprint_experiment(**{**experiment_input, "param_grid": grids_b})

    assert hash_a != hash_b


def test_materialized_splits_match_original_cv(experiment_input):
    X = experiment_input["X"]
    y = experiment_input["y"]
    groups = experiment_input["groups"]

    cv = GroupKFold(n_splits=2)

    splits = materialize_cv_splits(cv, X, y, groups)

    original_hash = fingerprint_cv_splits(cv, X, y, groups)
    materialized_hash = fingerprint_cv_splits(splits, X, y, groups)

    assert original_hash == materialized_hash
    assert len(splits) == 2


def test_materialized_splits_are_reused(experiment_input):
    class CountingCV:
        def __init__(self):
            self.calls = 0

        def split(self, X, y, groups):
            self.calls += 1
            yield from GroupKFold(n_splits=2).split(X, y, groups=groups)

    X = experiment_input["X"]
    y = experiment_input["y"]
    groups = experiment_input["groups"]

    cv = CountingCV()
    splits = materialize_cv_splits(cv, X, y, groups)

    manifest = build_experiment_manifest(**{**experiment_input, "cv": splits})

    search = GridSearchCV(
        estimator=Ridge(),
        param_grid={"alpha": [0.1, 1.0]},
        scoring="neg_mean_absolute_error",
        cv=splits,
    )

    search.fit(X, y)

    assert isinstance(manifest["folds_fingerprint"], str)
    assert len(search.cv_results_["params"]) == 2

    assert cv.calls == 1


def test_xgboost_manifest_without_training(experiment_input):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )

    manifest = build_experiment_manifest(
        **{
            **experiment_input,
            "model": model,
            "param_grid": {
                "max_depth": [5, 7],
            },
        }
    )

    fingerprint = experiment_fingerprint(manifest)

    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64


def test_xgboost_nan_and_none_have_different_fingerprints(
    experiment_input,
):
    from xgboost import XGBRegressor

    model_nan = XGBRegressor(missing=float("nan"))
    model_none = XGBRegressor(missing=None)

    manifest_nan = build_experiment_manifest(**{**experiment_input, "model": model_nan})

    manifest_none = build_experiment_manifest(
        **{**experiment_input, "model": model_none}
    )

    assert manifest_nan["fixed_params"]["missing"] == {
        "__hawkshot_nonfinite_float__": "nan"
    }

    assert experiment_fingerprint(manifest_nan) != experiment_fingerprint(manifest_none)


def test_numpy_scalar_in_manifest(experiment_input):
    manifest = build_experiment_manifest(
        **{
            **experiment_input,
            "param_grid": {"alpha": [np.int64(2)]},
        }
    )

    assert manifest["param_grid"]["alpha"] == [2]
