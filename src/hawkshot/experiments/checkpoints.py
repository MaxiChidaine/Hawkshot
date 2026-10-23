import hashlib
import json
import os

from tempfile import NamedTemporaryFile
from io import BytesIO
from pathlib import Path

import pandas as pd


class CheckpointConflictError(RuntimeError):
    """Raised when a checkpoint conflict is detected."""


def experiment_fingerprint(experiment: dict) -> str:
    """Generate a unique fingerprint for the experiment configuration.

    Args:
        experiment (dict): The experiment configuration.

    Returns:
        str: The unique fingerprint for the experiment configuration.
    """
    try:
        payload = json.dumps(
            experiment,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Experiment configuration must be JSON-serializable.") from exc

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inspect_checkpoint(
    checkpoint_path: Path,
    *,
    experiment: dict,
    expected_rows: int,
) -> pd.DataFrame | None:
    """Read a checkpoint only if its metadata and contents are valid."""
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".meta.json")

    if expected_rows < 1:
        raise ValueError("Expected rows must be 1 or more for checkpoint inspection.")

    if not checkpoint_path.exists() and not metadata_path.exists():
        return None

    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise CheckpointConflictError(
            f"Checkpoint path {checkpoint_path} or metadata path {metadata_path} is missing."
        )

    try:
        metadata = json.load(metadata_path.open("r", encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointConflictError(
            f"Failed to read metadata from {metadata_path}: {exc}"
        ) from exc

    expected_fingerprint = experiment_fingerprint(experiment)

    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or not isinstance(metadata.get("experiment"), dict)
        or metadata.get("experiment_fingerprint") != expected_fingerprint
        or experiment_fingerprint(metadata["experiment"]) != expected_fingerprint
    ):
        raise CheckpointConflictError(
            f"Checkpoint metadata in {metadata_path} does not match the expected experiment configuration."
        )
    csv_bytes = checkpoint_path.read_bytes()
    actual_csv_hash = hashlib.sha256(csv_bytes).hexdigest()

    if metadata.get("csv_sha256") != actual_csv_hash:
        raise CheckpointConflictError(
            f"Checkpoint CSV hash in {checkpoint_path} does not match the expected hash in metadata."
        )

    try:
        results = pd.read_csv(BytesIO(csv_bytes))
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise CheckpointConflictError(
            f"Failed to read checkpoint CSV from {checkpoint_path}: {exc}"
        ) from exc

    if len(results) != expected_rows:
        raise CheckpointConflictError(
            f"Checkpoint CSV in {checkpoint_path} has {len(results)} rows, expected {expected_rows}."
        )

    required_columns = {"params", "mean_test_MAE"}

    if not required_columns.issubset(results.columns):
        raise CheckpointConflictError(
            f"Checkpoint CSV in {checkpoint_path} is missing required columns: {required_columns - set(results.columns)}."
        )

    return results


def _stage_bytes(directory: Path, prefix: str, content: bytes) -> Path:
    """Write complete content to a temporary file."""
    with NamedTemporaryFile(
        mode="wb",
        dir=directory,
        prefix=f".{prefix}",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())

        return Path(temporary_file.name)


def _publish_new(staged_path: Path, destination: Path) -> None:
    """Publish a staged file without replacing an existing destination."""
    try:
        os.link(staged_path, destination)
    except FileExistsError as exc:
        raise CheckpointConflictError(
            f"Refusing to overwrite file: {destination}"
        ) from exc


def write_checkpoint(
    checkpoint_path: Path,
    *,
    experiment: dict,
    results: pd.DataFrame,
    expected_rows: int,
) -> Path:
    """Write checkpoint results and metadata without overwritting files.

    Parameters
    ----------
    checkpoint_path : Path
        Destination CVS path.
    experiment : dict
        JSON-serializable experiment manifest.
    results : pandas.DataFrame
        Grid-search results to save.
    expected_rows : int
        Expected number of hyperparameter combinations.

    Returns
    -------
    Path
        Path of the written checkpoint CSV

    Raises
    ------
    CheckpointConflictError
        If a checkpoint or its metadata already exists.
    ValueError
        If the results are incompletee or invalid.
    """

    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".meta.json")

    if expected_rows < 1 or len(results) != expected_rows:
        raise ValueError("Unexpected number of checkpoint rows.")

    required_columns = {"params", "mean_test_MAE"}

    if not required_columns.issubset(results.columns):
        raise ValueError("Required result columns are missing")

    if results["mean_test_MAE"].isna().any():
        raise ValueError("Checkpoint contains missing MAE scores")

    csv_bytes = results.to_csv(index=False).encode("utf-8")

    metadata = {
        "schema_version": 1,
        "experiment": experiment,
        "experiment_fingerprint": experiment_fingerprint(experiment),
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }

    metadata_bytes = json.dumps(
        metadata,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() or metadata_path.exists():
        raise CheckpointConflictError(
            "Checkpoint already exists. Refusing to overwrite"
        )

    staged_csv = None
    staged_metadata = None

    try:
        staged_csv = _stage_bytes(
            checkpoint_path.parent,
            checkpoint_path.stem,
            csv_bytes,
        )

        staged_metadata = _stage_bytes(
            checkpoint_path.parent,
            checkpoint_path.stem + ".meta",
            metadata_bytes,
        )

        _publish_new(staged_csv, checkpoint_path)
        _publish_new(staged_metadata, metadata_path)

    finally:
        if staged_csv is not None:
            staged_csv.unlink(missing_ok=True)

        if staged_metadata is not None:
            staged_metadata.unlink(missing_ok=True)

    return checkpoint_path
