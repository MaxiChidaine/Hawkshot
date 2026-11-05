from sklearn.ensemble import (
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)

from xgboost import XGBRegressor

from hawkshot.modeling.objectives import make_asymmetric_mse

MODEL_CLASSES = {
    "GradientBoosting": GradientBoostingRegressor,
    "ExtraTrees": ExtraTreesRegressor,
    "RandomForest": RandomForestRegressor,
    "XGBoost": XGBRegressor,
}


def build_model_from_config(config):
    """Reconstruct an estimator from a finalist configuration."""

    model_name = config["model"]

    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model family: {model_name}")

    params = config["params"].copy()

    objective = params.get("objective")

    if isinstance(objective, dict):
        spec = objective.get("custom_objective")

        if (
            not isinstance(spec, dict)
            or spec.get("name") != "hawkshot.asymmetric_mse"
            or spec.get("version") != 1
        ):
            raise ValueError("Unsupported custom objective configuration.")

        params["objective"] = make_asymmetric_mse(
            overestimation_weight=spec["overestimation_weight"]
        )

    model_class = MODEL_CLASSES[model_name]

    return model_class(**params)
