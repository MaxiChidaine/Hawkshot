import numpy as np


def make_asymmetric_mse(overestimation_weight):
    """Create an asymmetric squared-error objective for XGBoost.

    Parameters
    ----------
    overestimation_weight : float
        Weight applied when predicted RUL exceeds true RUL.
        A value greater than 1 penalizes overestimation more heavily.

    Returns
    -------
    callable
        XGBoost-compatible objective returning gradient and Hessian
    """

    def asymmetric_mse(y_true, y_pred):
        error = y_pred - y_true

        weights = np.where(
            error > 0,
            overestimation_weight,
            1.0,
        )

        grad = weights * error
        hess = weights

        return grad, hess

    asymmetric_mse.__hawkshot_objective_spec__ = {
        "name": "hawkshot.asymmetric_mse",
        "version": 1,
        "overestimation_weight": float(overestimation_weight),
    }

    return asymmetric_mse
