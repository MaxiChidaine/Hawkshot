import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FD001 Dataset Overview

    This notebook provide an exploratory analysis of the FD001 training dataset from the NASA C-MAPSS turbofan engine degradation datset.

    The target is to understand teh structure of the data, identify uninformative sensor variables, and investigate how sensor measurements evolve throughout the lifetime of individual engines.

    The analysis focuses on the following questions :

    - How is the FD001 dataset organised ?
    - Which sensors are constant or nearly constant ?
    - Which sensors show visible degradation trends ?
    - Are the trends consistent across the engine fleet ?
    - how stringly is each sensor associated with engine age ?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Objective

    FD001 contains multiple simulated turbofan engines monitored from initial operating state until failure.

    Each engine has its own lifetime, representd as a sequence of operating cycles. At every cycle, the dataset provides :

    - the engine identifier,
    - the current operating cycle,
    - three operational setting,
    - 21 sensor measurements.

    The objective of this notebook is not yet to build a Remaining useful Life prediction model. The current goal is to understand the dataset and identify the sensor variables that may contain useful information about engine degradation

    the analysis is therfore organised into three main stages :

    1. Inspect the structure and quality of the dataset,
    2. Remove constant or nearly constant sensor variable.
    3. Assess sensor degradation trends visually and statistically.

    The result of the exploration will guide the future feature-selection and modelling steps.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import marimo as mo

    return mo, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Data Loading

    The raw FD001 training file is stored as a whitespace-separated text file without column headers.

    The columns are assigned manually according to the C-MAPSS dataset structure :

    - 'engine_id' : unique identifier of the engine,
    - 'cycle' : operating cycle of the engine,
    - 'setting_1' to 'setting_3' : operational settings,
    - 'sensor_1' to 'sensor_21' : simulated sensor measurements.

    Each row represents one observation of the engine at one operating cycle.

    The dataset is loaded without modyfying the orginal source file. Any cleaning or filtering is applied to a separate DataFram in order to preserve the orginal dataset
    """)
    return


@app.cell
def _(pd):
    df_raw = pd.read_csv(
        "data/raw/cmapss/train_FD001.txt",
        sep=r"\s+",
        header=None,
    )
    return (df_raw,)


@app.cell
def _():
    id_columns = ["engine_id", "cycle"]

    operational_columns = [f"operational_setting_{i}" for i in range(1, 4)]

    sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

    column_names = id_columns + operational_columns + sensor_columns
    return column_names, sensor_columns


@app.cell
def _(column_names, df_raw):
    df_raw.columns = column_names
    df_raw.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Dataset structure

    Before analysing individual sensors, the general structure of the dataset is examined.

    The main points of interest are :

    - The number of engine trajectories,
    - The total number of obervations,
    - The number and type of variables,
    - The lifetime of each engine,
    - The presence of missing or duplicated values,
    - The distribution of engine lifetimes.

    The FD001 training datset contains one complete run-to-failure trajectory for each engine. Consequently, the highest cycle recorded for an engine correspond to its observed lifetime in a training data.

    The number of observations differs between engines because they do not all fail after the same number of cycles. The dataset must therefore be interpreted as a collection of independent time series rather than as a conventional table of unrelated observations.

    This distinction will be important during modelling. Data from the same engine must remain grouped together when constructing training and validation sets.
    """)
    return


@app.cell
def _(df_filtered, mo):
    mo.md(
        f"""
    The loaded dataset contains:

    - **{df_filtered.shape[0]:,} observations**
    - **{df_filtered["engine_id"].nunique()} engines**
    - **{df_filtered.shape[1]} columns**
    - Engine lifetimes ranging from **{
            df_filtered.groupby("engine_id")["cycle"].max().min()
        }** to **{df_filtered.groupby("engine_id")["cycle"].max().max()} cycles**
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Constant and near-constant sensors

    A sensor that remains constant throughout the dataset cannot provide information about engine degradation.

    Near-constant sensors may alors have limited predictive value because their variation is very small compared with the other signals. however, a small numerical range does not automatically mean that a sensor is useless. Its variation may still be systematic and related to engine ageing.

    For this reason, two complementary indicators are examined :

    - The number of unique values,
    - The variance of each sensor.

    Sensors with a single unique value are classified as constant and removed from the subsequent trend analysis.

    Near-constant sensors are identified separatly and retained temporarly. Their usefulness will be assessed by examining their evolution across cycles and across engines.

    This step reduces unnecessary variables while avoiding the premature removal of sensors whose degradation signal may be small but meaningful.
    """)
    return


@app.cell
def _(df_raw, sensor_columns):
    sensor_variations = (
        df_raw[sensor_columns]
        .nunique()
        .sort_values()
        .rename("number_of_unique_values")
        .to_frame()
    )

    sensor_variations
    return


@app.cell
def _(df_raw, sensor_columns):
    constant_sensors = []

    for sensor in df_raw[sensor_columns]:
        if df_raw[sensor].nunique() == 1:
            constant_sensors.append(sensor)

    near_constant_sensors = ["sensor_6"]

    removed_sensors = constant_sensors + near_constant_sensors
    return (removed_sensors,)


@app.cell
def _(df_raw, removed_sensors):
    df_filtered = df_raw.drop(columns=removed_sensors)
    df_filtered.head()

    return (df_filtered,)


@app.cell
def _(mo, removed_sensors):
    mo.md(
        f"""
        The analysis identified {len(removed_sensors)} constant sensors**:
    
        '{", ".join(removed_sensors)}'

        'Sensor_6' despite having more that 1 variable is considered a constant variable as its value barely changes from 26.61 to 26.60 throughout the 20000+ observations.
    
        These variables are excluded from the following analysis because they contain the variation in FD001"""
    )
    return


@app.cell
def _(df_filtered):
    available_sensors = [
        column for column in df_filtered.columns if column.startswith("sensor_")
    ]
    return (available_sensors,)


@app.cell
def _(available_sensors, mo):
    sensor_selector = mo.ui.multiselect(
        options=available_sensors,
        value=["sensor_2", "sensor_4", "sensor_11"],
        label="Capteurs à afficher",
        max_selections=15,
    )

    sensor_selector
    return (sensor_selector,)


@app.cell
def _(mo):
    smoothing_window = mo.ui.slider(
        start=3, stop=50, step=1, value=20, label="Fenêtre de lissage"
    )

    smoothing_window
    return (smoothing_window,)


@app.cell
def _(df_filtered, plt, sensor_selector, smoothing_window):
    selected_engine_data = df_filtered[df_filtered["engine_id"] == 4].copy()

    selected_sensors = sensor_selector.value

    normalized_sensors = (
        selected_engine_data[selected_sensors]
        - selected_engine_data[selected_sensors].mean()
    ) / selected_engine_data[selected_sensors].std()

    smoothed_sensors = normalized_sensors.rolling(
        window=smoothing_window.value,
        center=True,
    ).mean()

    plt.figure(figsize=(15, 8))

    for sensors in selected_sensors:
        plt.plot(
            selected_engine_data["cycle"], smoothed_sensors[sensors], label=sensors
        )

    plt.axhline(0, linewidth=1)
    plt.grid(True)
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
