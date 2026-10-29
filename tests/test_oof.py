import pytest
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyRegressor

from hawkshot.modeling.oof import generate_oof_predictions


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
    folds = list(cv.split(X, y, groups))

    cycles = pd.Series([1, 2] * 6)

    return X, y, groups, folds, cycles


def test_oof_covers_all_observations(sample_data):
    X, y, groups, folds, cycles = sample_data

    model = DummyRegressor(strategy="median")

    oof_df = generate_oof_predictions(
        model=model,
        X=X,
        y=y,
        groups=groups,
        folds=folds,
        cycles=cycles,
    )

    assert oof_df["row_position"].tolist() == list(range(len(X)))

    assert oof_df["engine_id"].tolist() == groups.tolist()
    assert oof_df["cycle"].tolist() == cycles.tolist()
    assert oof_df["y_true"].tolist() == y.tolist()


def test_oof_rejects_group_leakage(sample_data):
    X, y, groups, cycles, folds = sample_data

    bad_folds = [
        ([0, 2], [1, 3]),
    ]

    model = DummyRegressor(strategy="median")

    with pytest.raises(ValueError, match="Group leakage"):
        generate_oof_predictions(
            model=model,
            X=X,
            y=y,
            groups=groups,
            cycles=cycles,
            folds=bad_folds,
        )


def test_oof_rejects_missing_observation(sample_data):
    X, y, groups, folds, cycles = sample_data

    bad_folds = [
        (train_idx, val_idx[:-1] if i == 0 else val_idx)
        for i, (train_idx, val_idx) in enumerate(folds)
    ]

    model = DummyRegressor(strategy="median")

    with pytest.raises(AssertionError):
        generate_oof_predictions(
            model=model,
            X=X,
            y=y,
            groups=groups,
            cycles=cycles,
            folds=bad_folds,
        )


def test_oof_rejects_duplacted_observation(sample_data):
    X, y, groups, folds, cycles = sample_data

    bad_folds = []

    for i, (train_idx, val_idx) in enumerate(folds):
        bad_val_idx = val_idx.copy()

        if i == 0:
            bad_val_idx[-1] = bad_val_idx[0]

    bad_folds.append((train_idx, bad_val_idx))

    model = DummyRegressor(strategy="median")

    with pytest.raises(AssertionError):
        generate_oof_predictions(
            model=model,
            X=X,
            y=y,
            groups=groups,
            cycles=cycles,
            folds=bad_folds,
        )
