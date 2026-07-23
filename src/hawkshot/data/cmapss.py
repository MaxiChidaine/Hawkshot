from pathlib import Path
import pandas as pd

ID_COLUMNS = ["engine_id", "cycle"]

OPERATIONAL_COLUMNS = [f"operational_setting_{i}" for i in range(1, 4)]

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

COLUMN_NAMES = ID_COLUMNS + OPERATIONAL_COLUMNS + SENSOR_COLUMNS

EFFECTIVELY_CONSTANT_SENSORS = ["sensor_6"]


def load_fd001(data_dir: str | Path) -> pd.DataFrame:
    """Load the raw CMAPSS FD001 dataset.

    Args:
        data_dir : Directory containing the C-MAPSS FD001dataset files.

    Returns:
        The FD001 training dataset with descriptive column names.

    Raises:
        FileNotFoundError: If the dataset file is not found in the specified directory.
        ValueError: If the file does not contain the expected number of columns.

    """
    data_file = Path(data_dir) / "train_FD001.txt"

    if not data_file.exists():
        raise FileNotFoundError(f"FD001 training file not found : {data_file}")

    df = pd.read_csv(
        data_file,
        sep=r"\s+",
        header=None,
    )

    expected_columns = len(COLUMN_NAMES)

    if df.shape[1] != expected_columns:
        raise ValueError(
            "Unexpected FD001 structure. : "
            f"expected {expected_columns} columns, got {df.shape[1]}."
        )

    df.columns = COLUMN_NAMES

    return df


def filter_constant_sensors(
    df: pd.DataFrame,
    effectively_constant_sensors: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove strctly and effectively constant sensor variables.

    Args:
        df : C-MAPSS DataFrame containing the sensor columns.
        effective_constant_sensors: Sensors manually classified as effectvely constant after exploratory analysis. by default, sensor_6 is removed for FD001.


    Returns:
        A tuple containing :
            - a filtered copy of the DataFrame,
            - the names of the removed sensors.

    Raises:
        ValueError: If expected sensor columns are missing.
    """
    missing_sensors = [sensor for sensor in SENSOR_COLUMNS if sensor not in df.columns]

    if missing_sensors:
        raise ValueError(
            f"Missing expected sensor columns in the DataFrame: {missing_sensors}"
        )

    if effectively_constant_sensors is None:
        effectively_constant_sensors = EFFECTIVELY_CONSTANT_SENSORS

    strictly_constant_sensors = [
        sensor for sensor in SENSOR_COLUMNS if df[sensor].nunique() == 1
    ]

    sensors_to_remove = set(strictly_constant_sensors + effectively_constant_sensors)

    # Preserve the original C-MAPSS sensor order.
    removed_sensors = [
        sensor for sensor in SENSOR_COLUMNS if sensor in sensors_to_remove
    ]

    filtered_df = df.drop(columns=removed_sensors)

    return filtered_df, removed_sensors
