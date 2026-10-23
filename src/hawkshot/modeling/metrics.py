import numpy as np


def overestimation_rate(y_true, y_pred):
    """
    Calculate the overestimation rate.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        The overestimation rate, which is the proportion of predictions that are greater than the true values.
    """
    error = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.mean(error > 0))


def mean_signed_error(y_true, y_pred):
    """
    Calculate the mean signed error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        Mean prediction error (y_pred - y_true).
        Positive values indicate overestimation, while negative values
        indicate underestimation.
    """
    error = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.mean(error))


def mean_overestimation(y_true, y_pred):
    """
    Calculate the mean overestimation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        Mean of strictly positive prediction errors (y_pred - y_true).
        Only overestimated predictions are included in the average.
        Returns 0.0 when no overestimation occurs.
    """
    error = np.asarray(y_pred) - np.asarray(y_true)
    positive_errors = error[error > 0]

    if len(positive_errors) == 0:
        return 0.0

    return float(np.mean(positive_errors))


def mean_signed_error_scorer(estimator, X, y_true):
    """
    Scorer function for mean signed error, compatible with scikit-learn's scoring API.

    Parameters
    ----------
    estimator : object
        The fitted estimator to use for predictions.
    X : array-like of shape (n_samples, n_features)
        The input data.
    y_true : array-like of shape (n_samples,)
        The true target values.

    Returns
    -------
    float
        The mean signed error.
    """
    y_pred = estimator.predict(X)
    return mean_signed_error(y_true, y_pred)
