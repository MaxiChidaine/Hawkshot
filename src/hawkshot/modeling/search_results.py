import pandas as pd


def format_grid_search_results(grid_search):
    """Convert GridSearchCV results into a readable DataFrame.

    Preserve the original scikit-learn scores while adding
    metrics with their original interpretation.
    """
    results_df = pd.DataFrame(grid_search.cv_results_)

    metric_signs = {
        "MAE": -1,
        "RMSE": -1,
        "overestimation_rate": -1,
        "mean_overestimation": -1,
        "mean_signed_error": 1,
    }

    for metric, sign in metric_signs.items():
        source_column = f"mean_test_{metric}"

        if source_column in results_df.columns:
            results_df[metric] = sign * results_df[source_column]

    return results_df
