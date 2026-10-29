import pandas as pd
import pytest

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold

import hawkshot.experiments.runner as runner

from hawkshot.experiments.checkpoints import CheckpointConflictError


def test_oof_runner_reuses_checkpoint(tmp_path, monkeypatch):
    X = pd.DataFrame({"sensor_2": [10.0, 20.0] * 6})
    y = pd.Series([2.0, 4.0] * 6)
    groups = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    cycles = pd.Series([1, 2] * 6)

    folds = list(GroupKFold(n_splits=3).split(X, y, groups))

    checkpoint_path = tmp_path / "dummy_oof.csv"

    experiment = {
        "model": DummyRegressor(strategy="median"),
        "X": X,
        "y": y,
        "groups": groups,
        "cycles": cycles,
        "cv_splits": folds,
        "checkpoint_path": checkpoint_path,
    }

    first_results = runner.run_or_reuse_oof(**experiment)

    assert checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".meta.json").exists()

    def unexpected_training(**kwargs):
        raise AssertionError("OOF predictions were recomputed")

    monkeypatch.setattr(
        runner,
        "generate_oof_predictions",
        unexpected_training,
    )

    second_results = runner.run_or_reuse_oof(**experiment)

    pd.testing.assert_frame_equal(first_results, second_results)


def test_oof_runner_refjects_conflicting_experiment(tmp_path):
    X = pd.DataFrame({"sensor_2": [10.0, 20.0] * 4})
    y = pd.Series([2.0, 4.0] * 4)
    groups = pd.Series([1, 1, 2, 2, 3, 3, 4, 4])
    cycles = pd.Series([1, 2] * 4)

    folds = list(GroupKFold(n_splits=2).split(X, y, groups))

    checkpoint_path = tmp_path / "dummy_oof.csv"

    experiment = {
        "X": X,
        "y": y,
        "groups": groups,
        "cycles": cycles,
        "cv_splits": folds,
        "checkpoint_path": checkpoint_path,
    }

    runner.run_or_reuse_oof(
        model=DummyRegressor(strategy="median"),
        **experiment,
    )

    original_csv = checkpoint_path.read_bytes()

    with pytest.raises(CheckpointConflictError):
        runner.run_or_reuse_oof(
            model=DummyRegressor(strategy="mean"),
            **experiment,
        )

    assert checkpoint_path.read_bytes() == original_csv
