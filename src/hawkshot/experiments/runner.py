from pathlib import Path

from sklearn.model_selection import GridSearchCV, ParameterGrid

from hawkshot.experiments.checkpoints import (
    inspect_checkpoint,
    write_checkpoint,
)

from hawkshot.experiments.provenance import (
    build_experiment_manifest,
)

from hawkshot.modeling.search_results import format_grid_search_results


def run_or_reuse_grid_search(
    *,
    model,
    param_grid,
    X,
    y,
    groups,
    cv_splits,
    scoring,
    scoring_version,
    checkpoint_path,
    refit="MAE",
    n_jobs=-1,
):
    """Run a grid search or safely reuse a matching checkpoint.

    Parameters
    ----------
    model : sklearn_compatible estimator
        Base model to optimize.
    param_grid : dict or list of dict
        Hyperparameter search space.
    X : pandans.DataFrame
        Training features.
    y : pandas.Series
        Target values.
    groups : pandas.Series
        Engine identifiers.
    cv_splits : list of tuple
        Precomputed train and validation indices.
    scoring : dict
        Scoring configuration passed to GridSearchCV.
    scoring_version : str
        Explicit identifier of the scoring protocol.
    checkpoint_path : pathlib.Path
        Destination of the experiment results.
    refit : str, default="MAE"
        Metric used to select the best configuration.
    n_jobs : int, default=-1
        Number of parallel jobs.

    Returns
    -------
    pandas.DataFrame
        Formatted results from the search or its checkpoint.
    """

    checkpoint_path = Path(checkpoint_path)

    manifest = build_experiment_manifest(
        model=model,
        param_grid=param_grid,
        X=X,
        y=y,
        groups=groups,
        cv=cv_splits,
        scoring_version=scoring_version,
        refit=refit,
        search_n_jobs=n_jobs,
    )

    expected_rows = len(ParameterGrid(param_grid))

    existing = inspect_checkpoint(
        checkpoint_path, experiment=manifest, expected_rows=expected_rows
    )

    if existing is not None:
        return existing

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit,
        cv=cv_splits,
        n_jobs=n_jobs,
    )

    grid_search.fit(X, y, groups=groups)

    results = format_grid_search_results(grid_search)

    write_checkpoint(
        checkpoint_path,
        experiment=manifest,
        results=results,
        expected_rows=expected_rows,
    )

    return results
