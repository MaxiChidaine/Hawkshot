import pandas as pd
import pytest


from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from importlib.metadata import version


from hawkshot.experiments.checkpoints import experiment_fingerprint
from hawkshot.experiments.provenance import build_oof_manifest


@pytest.fixture
def oof_inputs():
    X = pd.DataFrame({"sensor_2": [10.0, 20.0] * 6})
    y = pd.Series([2.0, 4.0] * 6)
    groups = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    cycles = pd.Series([1, 2] * 6)

    folds = list(GroupKFold(n_splits=3).split(X, y, groups))

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "cycles": cycles,
        "folds": folds,
    }


def test_oof_manifest_changes_with_model_parameters():
    X = pd.DataFrame({"sensor_2": [10.0, 20.0] * 6})
    y = pd.Series([2.0, 4.0] * 6)
    groups = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    cycles = pd.Series([1, 2] * 6)

    folds = list(GroupKFold(n_splits=3).split(X, y, groups))

    common_inputs = {
        "X": X,
        "y": y,
        "groups": groups,
        "cycles": cycles,
        "folds": folds,
    }

    manifest_median = build_oof_manifest(
        model=DummyRegressor(strategy="median"), **common_inputs
    )

    manifest_mean = build_oof_manifest(
        model=DummyRegressor(strategy="mean"),
        **common_inputs,
    )

    assert experiment_fingerprint(manifest_median) != experiment_fingerprint(
        manifest_mean
    )


def test_manifest_tracks_pipeline_parameters(oof_inputs):
    pipeline_a = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True)),
            ("regressor", LinearRegression()),
        ]
    )

    pipeline_b = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("regressor", LinearRegression()),
        ]
    )

    manifest_a = build_oof_manifest(
        model=pipeline_a,
        **oof_inputs,
    )

    manifest_b = build_oof_manifest(model=pipeline_b, **oof_inputs)

    assert experiment_fingerprint(manifest_a) != experiment_fingerprint(manifest_b)


def test_oof_manifest_tracks_data_and_folds(oof_inputs):
    model = DummyRegressor(strategy="median")

    manifest_original = build_oof_manifest(
        model=model,
        **oof_inputs,
    )

    modified_X = oof_inputs["X"].copy()
    modified_X.loc[0, "sensor_2"] += 1.0

    manifest_modified_data = build_oof_manifest(
        model=model,
        **{**oof_inputs, "X": modified_X},
    )

    modified_folds = oof_inputs["folds"][::-1]

    manifest_modified_folds = build_oof_manifest(
        model=model,
        **{**oof_inputs, "folds": modified_folds},
    )

    original_fingerprint = experiment_fingerprint(manifest_original)

    assert original_fingerprint != experiment_fingerprint(manifest_modified_data)

    assert original_fingerprint != experiment_fingerprint(manifest_modified_folds)


def test_oof_manifest_tracks_estimator_versions(oof_inputs):
    gb_manifest = build_oof_manifest(
        model=GradientBoostingRegressor(),
        **oof_inputs,
    )

    xgb_manifest = build_oof_manifest(
        model=XGBRegressor(),
        **oof_inputs,
    )

    assert gb_manifest["versions"]["sklearn"] == version("scikit-learn")
    assert xgb_manifest["versions"]["sklearn"] == version("scikit-learn")

    assert xgb_manifest["versions"]["xgboost"] == version("xgboost")

    assert "xgboost" not in gb_manifest["versions"]
