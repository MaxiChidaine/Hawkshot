import hashlib
import json
import sklearn
import math

from importlib.metadata import version
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from hawkshot.experiments.checkpoints import experiment_fingerprint


def fingerprint_pandas(data: pd.DataFrame | pd.Series) -> str:
    """Compute a SHA-256 fingerprint of pandas data and its structure.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Data to identify, including values, index, and structure.

    Returns
    -------
    str
        SHA-256 hexadecimal fingerprint.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Input must be a pandas DataFrame or Series.")

    if isinstance(data, pd.DataFrame):
        metadata = {
            "type": "DataFrame",
            "shape": data.shape,
            "columns": [repr(col) for col in data.columns],
            "dtypes": [str(dtype) for dtype in data.dtypes],
            "index_names": [repr(name) for name in data.index.names],
        }
    else:
        metadata = {
            "type": "Series",
            "shape": data.shape,
            "name": repr(data.name),
            "dtype": str(data.dtype),
            "index_names": [repr(name) for name in data.index.names],
        }

    digest = hashlib.sha256()

    digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))

    value_hashes = pd.util.hash_pandas_object(
        data,
        index=True,
    )

    digest.update(value_hashes.to_numpy(dtype="<u8").tobytes())

    return digest.hexdigest()


def fingerprint_cv_splits(cv, X, y, groups) -> str:
    """Compute a fingerprint of the actual cross-validation splits/

    Parameters
    ----------
    cv : cross-validation splitter
        Splitter implementing the split method.
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Target Value.
    groups : pandas.Series
        Engine indentifiers.

    Returns
    -------
    str
        SHA-256 fingerprints of the train and validation indices.
    """
    splits = []

    split_iterator = cv.split(X, y, groups=groups) if hasattr(cv, "split") else iter(cv)
    for train_indices, validation_indices in split_iterator:
        splits.append(
            {
                "train": train_indices.tolist(),
                "validation": validation_indices.tolist(),
            }
        )

    if not splits:
        raise ValueError("Cross-validation produced no splits.")

    payload = json.dumps(
        splits,
        separators=(",", ":"),
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_experiment_manifest(
    *,
    model,
    param_grid,
    X,
    y,
    groups,
    cv,
    scoring_version,
    refit="MAE",
    search_n_jobs=-1,
):
    """Build a reproductible description of a grid search experiment.

    Parameters
    ----------

    model : sklearn-compatible estimator
        Base estimator used by GridSearchCV.
    param_grid : dict or list of dict
        Hyperparameters combinations to evaluate.
    X : pandas.DataFrame
        Training features.
    y : pandas.Series
        Target Value.
    groups : pandas.Serie
        Engine identifiers used for grouped cross-validation.
    cv : cross-validation splitter
        Splitter defining the train and validation folds.
    scoring_version : str
        Explicit version identifier for the scoring protocol.
    search_n_jobs : int, default=-1
        Parallelism setting of GridSearchCV.

    Returns
    -------

    dict
        JSON-serializable experiment description.
    """

    if not isinstance(scoring_version, str) or not scoring_version.strip():
        raise ValueError("A non-empty scoring_version is required.")

    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y and groups must have the same length.")

    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }

    model_class = type(model)
    model_module = model_class.__module__

    if model_module.startswith("xgboost"):
        versions["xgboost"] = version("xgboost")

    fixed_params = model.get_params(deep=False).copy()

    objective = fixed_params.get("objective")

    if callable(objective):
        objective_spec = getattr(
            objective,
            "__hawkshot_objective_spec__",
            None,
        )

        print("Model:", type(model).__name__)
        print("Objective:", repr(objective))
        print("Objective spec:", repr(objective_spec))
        print("Spec type:", type(objective_spec))

        if not isinstance(objective_spec, dict):
            raise ValueError("Custom objective requires explicit provenance metadata")

        fixed_params["objective"] = {
            "custom_objective": objective_spec,
        }

    experiment = {
        "manifest_schema_version": 1,
        "model": f"{model_module}.{model_class.__qualname__}",
        "fixed_params": _serialize_grid_value(fixed_params),
        "param_grid": _serialize_grid_value(param_grid),
        "X_fingerprint": fingerprint_pandas(X),
        "y_fingerprint": fingerprint_pandas(y),
        "groups_fingerprint": fingerprint_pandas(groups),
        "folds_fingerprint": fingerprint_cv_splits(cv, X, y, groups),
        "scoring_version": scoring_version,
        "refit": refit,
        "search_n_jobs": search_n_jobs,
        "versions": versions,
    }

    try:
        experiment_fingerprint(experiment)
    except ValueError as exc:
        raise ValueError(
            "Experiment parameters must be JSON-serializable. "
            "Custom objectives require an explicit stable representation."
        ) from exc

    return experiment


def _serialize_grid_value(value):
    """Convert custom objectives into JSON-compatible descriptions."""
    if isinstance(value, dict):
        return {key: _serialize_grid_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_serialize_grid_value(item) for item in value]

    if isinstance(value, np.generic):
        return _serialize_grid_value(value.item())

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            kind = "nan"
        elif value > 0:
            kind = "+inf"
        else:
            kind = "-inf"

        return {"__hawkshot_nonfinite_float__": kind}

    if callable(value):
        objective_spec = getattr(value, "__hawkshot_objective_spec__", None)

        if not isinstance(objective_spec, dict):
            raise ValueError(
                "Experiment parameters must be JSON-serializable. "
                "Callable has nop explicit provenance metadata."
            )

        return {"custom_objective": objective_spec.copy()}

    return value


def materialize_cv_splits(cv, X, y, groups):
    """Generate reusable cross-validation splits.

    Parameters
    ----------
    cv : cross-validation splitter
        Splitter implementing the split method.
    X : pandas.DataFrame
        Inpute features.
    y : pandas.Series
        Target values.
    groups : pandas.Series
        Engine identifiers.

    Returns
    -------
    list of tuple
        Train and validation index arrays for each fold.
    """

    splits = [
        (
            np.asarray(train_idx, dtype=int).copy(),
            np.asarray(validation_idx, dtype=int).copy(),
        )
        for train_idx, validation_idx in cv.split(X, y, groups=groups)
    ]

    if not splits:
        raise ValueError("Cross-validation produced no splits")

    return splits


def _describe_estimator(model):
    """return a JSON-compatible description of an estimator."""

    model_class = type(model)
    model_name = f"{model_class.__module__}.{model_class.__qualname__}"

    if isinstance(model, Pipeline):
        return {
            "class": model_name,
            "params": _serialize_grid_value(
                {
                    key: value
                    for key, value in model.get_params(deep=False).items()
                    if key != "steps"
                }
            ),
            "steps": [
                {
                    "name": name,
                    "estimator": _describe_estimator(estimator),
                }
                for name, estimator in model.steps
            ],
        }

    return {
        "class": model_name,
        "params": _serialize_grid_value(model.get_params(deep=False)),
    }


def build_oof_manifest(
    *,
    model,
    X,
    y,
    groups,
    cycles,
    folds,
    protocol_version="oof_v1",
):
    """Build a reproductible manifest for OOF predictions."""

    if not (len(X) == len(y) == len(groups) == len(cycles)):
        raise ValueError("X, y, groups and cycles must have the same length.")

    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        **_get_estimator_version(model),
    }

    experiment = {
        "manifest_schema_version": 1,
        "experiment_type": "oof",
        "protocol_version": protocol_version,
        "estimator": _describe_estimator(model),
        "X_fingerprint": fingerprint_pandas(X),
        "y_fingerprint": fingerprint_pandas(y),
        "groups_fingerprint": fingerprint_pandas(groups),
        "cycles_fingerprint": fingerprint_pandas(cycles),
        "folds_fingerprint": fingerprint_cv_splits(folds, X, y, groups),
        "versions": versions,
    }

    experiment_fingerprint(experiment)

    return experiment


def _get_estimator_version(model):
    """Identify the library versions used by an estimator."""

    distributions = {
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
    }

    if isinstance(model, Pipeline):
        estimators = [
            estimator
            for _, estimator in model.steps
            if estimator is not None and estimator != "passthrough"
        ]

    else:
        estimators = [model]

    packages = {"sklearn"}

    packages.update(
        type(estimator).__module__.split(".")[0] for estimator in estimators
    )

    return {
        package: version(distribution)
        for package, distribution in distributions.items()
        if package in packages
    }
