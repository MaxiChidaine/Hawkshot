import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FD001 Dataset Overview

    This notebook provides an exploratory overview of the FD001 training dataset
    from the NASA C-MAPSS turbofan engine degradation dataset.

    The objective is to verify the structure and integrity of the data, describe
    the engine run-to-failure trajectories, and identify sensor variables with no
    meaningful variation before conducting a more detailed sensor trend analysis.

    The analysis focuses on the following questions:

    - How is the FD001 dataset organised?
    - How do engine lifetimes vary across the fleet?
    - Does the dataset contain missing or duplicated observations?
    - Which sensors are strictly or effectively constant?
    - Which sensor variables should be retained for further analysis?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Objective

    The objective of this notebook is to better understand how degradation is represented in the FD001 dataset before beginning any Remaining Useful Life modelling.

    The analysis focuses on three main questions:

    1. **What is the structure of the run-to-failure trajectories?**

    The notebook examines how engine lifetimes are organised, how many cycles are recorded for each engine, and how the different trajectories compare across the fleet.

    2. **Which variables contain little or almost no information?**

       Constant and near-constant variables are identified in order to remove sensors that do not vary enough to contribute meaningfully to the analysis.

    3. **Which sensor variables should be retained for further analysis?**

    The notebook identifies strictly and effectively constant sensors and
    prepares a consistent set of variables for the subsequent sensor trend
    analysis.

    The purpose of this exploration is to remove uninformative variables and
    prepare a consistent set of sensor measurements for the subsequent trend
    analysis and Remaining Useful Life prediction steps.
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

    The dataset is loaded without modifying the original source file. Any cleaning or filtering is applied to a separate DataFrame in order to preserve the original dataset
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
    return column_names, operational_columns, sensor_columns


@app.cell
def _(column_names, df_raw):
    df_raw.columns = column_names
    df_raw.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Dataset Structure and Integrity

    Before analysing the sensor measurements, the general structure and integrity
    of the dataset are verified.

    The FD001 training dataset contains one complete run-to-failure trajectory for
    each engine. The final cycle recorded for an engine therefore corresponds to
    its observed lifetime.

    Because engine lifetimes differ, the dataset must be interpreted as a
    collection of independent time series rather than as a conventional table of
    unrelated observations.

    This distinction will be important during modelling: observations from the
    same engine must remain grouped when constructing training and validation
    sets.
    """)
    return


@app.cell
def _(df_raw):
    engine_lifetimes = df_raw.groupby("engine_id")["cycle"].max().rename("lifetime")
    return (engine_lifetimes,)


@app.cell
def _(df_raw, engine_lifetimes, operational_columns, pd, sensor_columns):
    dataset_summary = pd.Series(
        {
            "observations": len(df_raw),
            "engines": df_raw["engine_id"].nunique(),
            "columns": df_raw.shape[1],
            "operational_settings": len(operational_columns),
            "sensors": len(sensor_columns),
            "missing_values": df_raw.isna().sum().sum(),
            "duplicated_rows": df_raw.duplicated().sum(),
            "duplicated_engine_cycles": df_raw[["engine_id", "cycle"]]
            .duplicated()
            .sum(),
            "minimum_lifetime": engine_lifetimes.min(),
            "maximum_lifetime": engine_lifetimes.max(),
        },
        name="value",
    )

    dataset_summary
    return


@app.cell
def _(engine_lifetimes, plt):
    plt.figure(figsize=(10, 5))
    plt.hist(engine_lifetimes, bins=15, edgecolor="black")

    plt.xlabel("Engine lifetime in cycles")
    plt.ylabel("Number of engines")
    plt.title("Distribution of engine lifetimes in FD001")
    plt.grid(axis="y", alpha=0.3)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Most engine trajectories contain approximately 150 to 240 cycles, while a smaller number of engines operate for more than 300 cycles. This confirms that trajectory lengths vary substantially across the fleet and must be handled independently during future modelling steps.
    """)
    return


@app.cell
def _(engine_lifetimes):
    engine_lifetimes.describe().round(2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Constant and near-constant sensors

    A sensor that remains constant throughout the dataset cannot provide information about engine degradation.

    Near-constant sensors may also have limited predictive value because their variation is very small compared with the other signals. However, a small numerical range does not automatically mean that a sensor is useless. Its variation may still be systematic and related to engine ageing.

    For this reason, two complementary indicators are examined :

    - The number of unique values,
    - The variance of each sensor.

    Sensors with a single unique value are classified as constant and removed from the subsequent trend analysis.

    Near-constant sensors are identified separately and retained temporarily. Their usefulness will be assessed by examining their evolution across cycles and across engines.

    This step reduces unnecessary variables while avoiding the premature removal of sensors whose degradation signal may be small but meaningful.
    """)
    return


@app.cell
def _(df_raw, pd, sensor_columns):
    sensor_profile = pd.DataFrame(
        {
            "unique_values": df_raw[sensor_columns].nunique(),
            "minimum": df_raw[sensor_columns].min(),
            "maximum": df_raw[sensor_columns].max(),
            "mean": df_raw[sensor_columns].mean(),
            "std": df_raw[sensor_columns].std(),
            "variance": df_raw[sensor_columns].var(),
        }
    )

    sensor_profile["range"] = sensor_profile["maximum"] - sensor_profile["minimum"]

    sensor_profile = sensor_profile.sort_values(["unique_values", "variance"])

    sensor_profile_display = sensor_profile.round(2)
    sensor_profile_display
    return (sensor_profile,)


@app.cell
def _(sensor_profile):
    constant_sensors = sensor_profile.index[
        sensor_profile["unique_values"] == 1
    ].tolist()

    effectively_constant_sensors = ["sensor_6"]

    removed_sensors = constant_sensors + effectively_constant_sensors

    removed_sensors
    return (removed_sensors,)


@app.cell
def _(df_raw, removed_sensors):
    df_filtered = df_raw.drop(columns=removed_sensors)
    df_filtered.head()
    return (df_filtered,)


@app.cell
def _(df_filtered):
    available_sensors = [
        column for column in df_filtered.columns if column.startswith("sensor_")
    ]
    return (available_sensors,)


@app.cell(hide_code=True)
def _(available_sensors, df_filtered, mo):
    mo.md(f"""

    The analysis identified six strictly constant sensors:

    `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, and `sensor_19`.

    In addition, `sensor_6` is classified as effectively constant.

    Although 'sensor_6' is not strictly constant,  it only takes two
    values, 26.60 and 26.61, across more than 20,000 observations. This extremely limited variation is considered insufficient for the present analysis. The sensor is therefore treated as effectively constant and removed together with the strictly constant sensors.

     Other low-variability sensors, such as 'sensor_17', are retained because their wider range of values may still contain a systematic degradation signal.

     The strictly and effectively constant sensors are excluded from the following
    analysis because they contain no meaningful variation for the present
    degradation study.

    After removing the constant sensors, the analysis dataset contains:
    - **{df_filtered.shape[0]:,} observations**
    - **{df_filtered.shape[1]} columns**
    - **{len(available_sensors)} remaining sensors**

    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Variables Retained for further analysis

    The variables retained at this stage are those that contain at least some variation in the FD001 training dataset.

    This filtering step does not imply that every retained sensor is informative for Remaining Useful Life (RUL) prediction. It only removes variables that are demonstrably unable to describe changes in engine condition.

    The retained sensor variables will be examined in the next notebook through individual engine trajectories, comparisons between engines, per-engine Spearman correlations and fleet level consistency measures.

    The table below lists the sensors retained for the subsequent analysis, in which their trends across individual engines will be examined in greater detail.
    """)
    return


@app.cell
def _(available_sensors, sensor_profile):
    retained_sensor_profile = sensor_profile.loc[available_sensors].copy()

    retained_sensor_profile[
        [
            "unique_values",
            "minimum",
            "maximum",
            "range",
            "variance",
        ]
    ].round(2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Main findings and next steps

    This first exploration established the general structure and quality of the FD001 training dataset. The data contain 20,631 observations corresponding to 100 independent engine trajectories monitored until failure. Each observation represents the state of one engine at a specific operating cycle and includes three operational settings and 21 sensor measurements.

    The engine trajectories do not all have the same durations. Their observed lifetimes range from 128 to 362 cycles, confirming that FD001 must be treated as a collection of independent time series. This structure will need to be preserved during the future construction of the training and validation dataset in order to avoid mixing observations from the same engine between different subsets.

    The initial variable inspection also identified several sensor measurements that provide no meaningful variation. Six sensors are strictly constant across the complete dataset. 'sensor_6' is not strictly constant, but only takes two values separated by 0.01 across more than 20,000 observations. It is therefore treated as effectively constant and removed from the following analyses. Other sensors with limited variability are retained because their changes may still be related to engine ageing.

    At this stage, the retained sensors have not yet been classified according to their predictive relevance. The current filtering only removes variables whose lack of variation makes them unsuitable for degradation analysis. The next notebook will examine how the remaining sensor measurements evolve throughout engine life. Individual trajectories will first be explored visually, before extending the analysis to the complete fleet through per-engine Spearman correlations and fleet level consistency measures.
    """)
    return


if __name__ == "__main__":
    app.run()
