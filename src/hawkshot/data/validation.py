import pandas as pd


def validate_engine_trajectories(df: pd.DataFrame) -> None:
    """Validate engine cycle sequences before feature engineering."""

    required_columns = {"engine_id", "cycle"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("The DataFrame is empty.")

    if not pd.api.types.is_integer_dtype(df["cycle"]):
        raise ValueError("The 'cycle' column must be of integer type.")

    if (df["cycle"] < 1).any():
        raise ValueError("The 'cycle' column must contain positive integers.")

    duplicates = df.duplicated(subset=["engine_id", "cycle"])

    if duplicates.any():
        raise ValueError("Duplicate (engine_id, cycle) pairs found in the DataFrame.")

    df_sorted = df.sort_values(["engine_id", "cycle"])

    cycle_diffs = df_sorted.groupby("engine_id")["cycle"].diff()

    if cycle_diffs.dropna().ne(1).any():
        raise ValueError("Missing cycles detected in engine trajectories.")
