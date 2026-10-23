import numpy as np

from hawkshot.modeling.objectives import make_asymmetric_mse


def test_asymmetric_mse_gradient_and_hessian():
    objective = make_asymmetric_mse(3.5)

    y_true = np.array([10.0] * 3)
    y_pred = np.array([12.0, 8.0, 10.0])

    grad, hess = objective(y_true, y_pred)

    np.testing.assert_allclose(grad, [7.0, -2.0, 0.0])
    np.testing.assert_allclose(hess, [3.5, 1.0, 1.0])


def test_objective_metadata_identifies_weight():
    objective_a = make_asymmetric_mse(3.5)
    objective_b = make_asymmetric_mse(5.0)

    spec_a = objective_a.__hawkshot_objective_spec__
    spec_b = objective_b.__hawkshot_objective_spec__

    assert spec_a["version"] == 1
    assert spec_a["overestimation_weight"] == 3.5
    assert spec_b["overestimation_weight"] == 5.0
    assert spec_a != spec_b
