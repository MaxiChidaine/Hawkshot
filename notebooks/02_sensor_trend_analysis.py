import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from hawkshot.data.cmapss import (
        SENSOR_COLUMNS,
        filter_constant_sensors,
        load_fd001,
    )

    df_raw = load_fd001("data/raw/cmapss")

    df_filtered, removed_sensors = filter_constant_sensors(df_raw)
    return SENSOR_COLUMNS, df_filtered, mo


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Visual analysis of sensor trends

    The remaining sensors are plotted against the operating cycle for selected engines.

    This visual analysis aims to determine whether a sensor :

    - Increases or decreases as the engine approches failure,
    - Remains stable throughout the engine lifetime,
    - Contains a trend that is partially hidden by noise,
    - Behave similarly across several engines,
    - Shows engine-specific behaviour.

    The raw sensor measurements are displayed to preserve the actual variability of the data. A smoothed curve may also be added to make the underlying trend easier to interpret.

    Smoothing is used only as a visual aid. It does not replace the original measurements and is not yet used as an input transformation for machine learning models.

    A sensor may appear informative for one engine but not for the rest of the fleet. Therefore, conclusions should not be based on a single trajectory. Several engines with different lifetime must be compared before considering a sensor representative of degradation.
    """)
    return


@app.cell
def _(SENSOR_COLUMNS, df_filtered, mo):
    available_sensors = [
        sensor for sensor in SENSOR_COLUMNS if sensor in df_filtered.columns
    ]

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
        start=3, stop=50, step=1, value=20, label="Smooth window"
    )

    smoothing_window
    return (smoothing_window,)


@app.cell
def _(df_filtered, plt, sensor_selector, smoothing_window):
    selected_engine_data = df_filtered[df_filtered["engine_id"] == 7].copy()

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
