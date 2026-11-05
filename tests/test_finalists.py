import json
from pathlib import Path

import pytest

from hawkshot.experiments.finalists import build_model_from_config


def test_all_finalists_reconstruct_parameters():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "oof_finalists.json"

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    finalists = payload["finalists"]

    assert len(finalists) == 12

    for config in finalists:
        model = build_model_from_config(config)
        restored_params = model.get_params()

        for param_name, expected_value in config["params"].items():
            actual_value = restored_params[param_name]

            if param_name == "objective" and isinstance(expected_value, dict):
                assert (
                    actual_value.__hawkshot_objective_spec__
                    == expected_value["custom_objective"]
                )

            else:
                assert actual_value == expected_value


@pytest.mark.parametrize(
    "config, error_message",
    [
        (
            {
                "model": "UnknownModel",
                "params": {},
            },
            "Unsupported model family",
        ),
        (
            {
                "model": "XGBoost",
                "params": {
                    "objective": {
                        "custom_objective": {
                            "name": "unknown_objective",
                            "version": 1,
                            "overestimation_weight": 3.5,
                        }
                    }
                },
            },
            "Unsupported custom objective",
        ),
    ],
)
def test_rejects_unsupported_finalist_config(config, error_message):
    with pytest.raises(ValueError, match=error_message):
        build_model_from_config(config)
