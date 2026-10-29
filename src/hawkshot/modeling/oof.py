import pandas as pd

from sklearn.base import clone


def generate_oof_predictions(model, X, y, groups, cycles, folds):
    fold_predictions = []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])

        if train_groups & val_groups:
            raise ValueError("Group leakage detected")

        fold_model = clone(model)

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_val = X.iloc[val_idx]

        fold_model.fit(X_train, y_train)

        y_pred = fold_model.predict(X_val)

        fold_df = pd.DataFrame(
            {
                "row_position": val_idx,
                "engine_id": groups.iloc[val_idx].to_numpy(),
                "cycle": cycles.iloc[val_idx].to_numpy(),
                "fold": fold_id,
                "y_true": y.iloc[val_idx].to_numpy(),
                "y_pred": y_pred,
            }
        )
        fold_predictions.append(fold_df)

    oof_df = pd.concat(fold_predictions, ignore_index=True)

    assert len(oof_df) == len(X)
    assert oof_df["row_position"].is_unique

    oof_df = oof_df.sort_values("row_position").reset_index(drop=True)

    return oof_df
