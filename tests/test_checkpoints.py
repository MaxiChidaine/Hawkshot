import hashlib
import json
import os

import pandas as pd
import pytest

import hawkshot.experiments.checkpoints as checkpoints


@pytest.fixture
def experiment():
    return {
        "model": "XGBRegressor",
        "fixed_params": {"learning_rate": 0.1, "n_estimators": 100},
        "param_grid": {"max_depth": [3, 5], "subsample": [0.8, 1.0]},
        "dataset_fingerprint": "abc123",
        "folds_fingerprint": "def456",
        "scoring_id": "synthetic_scorer_v1",
    }


def create_test_checkpoint(path, experiment, rows=2):
    results = pd.DataFrame(
        {
            "params": [f"candidate-{i}" for i in range(rows)],
            "mean_test_MAE": [-float(i + 1) for i in range(rows)],
        }
    )
    results.to_csv(path, index=False)

    metadata = {
        "schema_version": 1,
        "experiment": experiment,
        "experiment_fingerprint": checkpoints.experiment_fingerprint(experiment),
        "csv_sha256": hashlib.sha256(
            results.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
    }

    path.with_suffix(".meta.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_missing_missing_checkpoint_returns_none(tmp_path, experiment):
    path = tmp_path / "run.csv"

    assert (
        checkpoints.inspect_checkpoint(
            checkpoint_path=path,
            experiment=experiment,
            expected_rows=2,
        )
        is None
    )


def test_valid_checkpoint_can_be_reused(tmp_path, experiment):
    path = tmp_path / "run.csv"
    create_test_checkpoint(path, experiment, rows=2)

    assert path.is_file(), f"CSV missing. Files: {list(tmp_path.iterdir())}"

    metadata_path = path.with_suffix(".meta.json")

    assert metadata_path.is_file(), (
        f"Metadata missing. Files: {list(tmp_path.iterdir())}"
    )

    results = checkpoints.inspect_checkpoint(
        checkpoint_path=path,
        experiment=experiment,
        expected_rows=2,
    )

    assert len(results) == 2


def test_same_row_count_different_grid_is_rejected(tmp_path, experiment):
    path = tmp_path / "run.csv"
    create_test_checkpoint(path, experiment, rows=2)

    changed_experiment = {
        **experiment,
        "param_grid": {"max_depth": [3, 5], "subsample": [0.8, 0.9]},
    }

    with pytest.raises(checkpoints.CheckpointConflictError):
        checkpoints.inspect_checkpoint(
            checkpoint_path=path,
            experiment=changed_experiment,
            expected_rows=2,
        )


def test_missing_metadata_is_rejected(tmp_path, experiment):
    path = tmp_path / "run.csv"
    pd.DataFrame({"params": ["a", "b"]}).to_csv(path, index=False)

    with pytest.raises(checkpoints.CheckpointConflictError):
        checkpoints.inspect_checkpoint(
            checkpoint_path=path,
            experiment=experiment,
            expected_rows=2,
        )


def test_incomplete_checkpoint_is_rejected(tmp_path, experiment):
    path = tmp_path / "run.csv"
    create_test_checkpoint(path, experiment, rows=1)

    with pytest.raises(checkpoints.CheckpointConflictError):
        checkpoints.inspect_checkpoint(
            checkpoint_path=path,
            experiment=experiment,
            expected_rows=2,
        )


def test_modified_csv_is_rejected(tmp_path, experiment):
    path = tmp_path / "run.csv"
    create_test_checkpoint(path, experiment, rows=2)

    path.write_text(
        path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(checkpoints.CheckpointConflictError):
        checkpoints.inspect_checkpoint(
            checkpoint_path=path,
            experiment=experiment,
            expected_rows=2,
        )


def test_write_creates_valid_checkpoint(tmp_path, experiment):
    path = tmp_path / "run.csv"

    results = pd.DataFrame(
        {
            "params": ["candidate-a", "candidate-b"],
            "mean_test_MAE": [-10.0, -12.0],
        }
    )

    checkpoints.write_checkpoint(
        path,
        experiment=experiment,
        results=results,
        expected_rows=2,
    )

    recovered = checkpoints.inspect_checkpoint(
        path,
        experiment=experiment,
        expected_rows=2,
    )

    pd.testing.assert_frame_equal(recovered, results)
    assert path.with_suffix(".meta.json").is_file()


def test_write_refuses_existing_checkpoint(tmp_path, experiment):
    path = tmp_path / "run.csv"

    results = pd.DataFrame(
        {
            "params": ["a", "b"],
            "mean_test_MAE": [-1.0, -2.0],
        }
    )

    checkpoints.write_checkpoint(
        path, experiment=experiment, results=results, expected_rows=2
    )

    original_csv = path.read_bytes()
    metadata_path = path.with_suffix(".meta.json")
    original_metadata = metadata_path.read_bytes()

    with pytest.raises(checkpoints.CheckpointConflictError):
        checkpoints.write_checkpoint(
            path, experiment=experiment, results=results, expected_rows=2
        )

    assert path.read_bytes() == original_csv
    assert metadata_path.read_bytes() == original_metadata


def test_write_rejects_incomplete_results(tmp_path, experiment):
    path = tmp_path / "run.csv"

    results = pd.DataFrame(
        {
            "params": ["candidate-a"],
            "mean_test_MAE": [-10.0],
        }
    )

    with pytest.raises(ValueError):
        checkpoints.write_checkpoint(
            path,
            experiment=experiment,
            results=results,
            expected_rows=2,
        )

        assert not path.exists()
        assert not path.with_suffix(".meta.json").exists()


def test_write_preserves_legacy_csv(tmp_path, experiment):
    path = tmp_path / "run.csv"
    path.write_text("historical results", encoding="utf-8")

    results = pd.DataFrame(
        {
            "params": ["a", "b"],
            "mean_test_MAE": [-1.0, -2.0],
        }
    )

    with pytest.raises(checkpoints.CheckpointConflictError):
        checkpoints.write_checkpoint(
            path,
            experiment=experiment,
            results=results,
            expected_rows=2,
        )

    assert path.read_text(encoding="utf-8") == "historical results"
    assert not path.with_suffix(".meta.json").exists()


def test_interrupted_publication_keeps_checkpoint_invalid(
    tmp_path, experiment, monkeypatch
):
    path = tmp_path / "run.csv"

    results = pd.DataFrame(
        {
            "params": ["candidate-a", "candidate-b"],
            "mean_test_MAE": [-10.0, -12.0],
        }
    )

    original_link = os.link

    def interrupted_link(source, destination):
        if str(destination).endswith(".meta.json"):
            raise OSError("Simulated interruption")

        return original_link(source, destination)

    monkeypatch.setattr(checkpoints.os, "link", interrupted_link)

    with pytest.raises(OSError, match="Simulated interruption"):
        checkpoints.write_checkpoint(
            path, experiment=experiment, results=results, expected_rows=2
        )
    assert path.exists()
    assert not path.with_suffix(".meta.json").exists()

    with pytest.raises(checkpoints.CheckpointConflictError):
        checkpoints.inspect_checkpoint(
            path,
            experiment=experiment,
            expected_rows=2,
        )

    assert not any(file.name.endswith(".tmp") for file in tmp_path.iterdir())
