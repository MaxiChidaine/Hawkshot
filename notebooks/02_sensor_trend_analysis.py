import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from hawkshot.data.cmapss import (
        SENSOR_COLUMNS,
        filter_constant_sensors,
        load_fd001,
    )

    df_raw = load_fd001("data/raw/cmapss")

    df_filtered, removed_sensors = filter_constant_sensors(df_raw)

    available_sensors = [
        sensor for sensor in SENSOR_COLUMNS if sensor in df_filtered.columns
    ]
    return available_sensors, df_filtered, mo, plt


@app.cell
def _(mo):
    mo.md(r"""
    # FD001 Sensor trend analysis

    This notebooks investigates how the sensor measurements retained from the FD001 dataset evolve throughout engine life.

    The previous dataset overview identified and removed striclty and effectively constant sensor variables. The present analysis focuses on the remaining sensors and examines wether their evolution is associated with engine ageing.

    The analysis follows a progressive approach :

    1. Explore sensor trajectories for an individual engine.
    2. Compare the same sensor across several engines.
    3. Calculate per-engine Spearman correlations.
    4. Assess the strength and consistency of sensor trends across the fleet.

    The initial visual exploration is used to formulate hypotheses. These hypotheses will then be evaluated statistically across all 100 engines.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Individual Engine exploration

    The first step is to examine how several sensor measurements evolve throughout the lifetime of a single engine.

    Because the sensors use different units and numerical scales, their values are standardised within the selected engine before being displayed on the same graph. This trandformation preserves the direction and shape of each trajectories while allowing direct visual comparison between sensors.

    Both the standardised raw measurements and a rolling mean are displayed. The rolling mean is used only as a visual aid to reveal trends that may be partially hidden by short-term variability.

    This exploration can indicate whether a sensor increases, decreases, remains stable, or changes mainly near the end of engine life. However, observations from a single engine cannot be generalised to the complete fleet.
    """)
    return


@app.cell
def _(df_filtered, mo):
    engine_selector = mo.ui.dropdown(
        options=df_filtered["engine_id"].unique().tolist(),
        value=1,
        label="Engine to display",
    )
    return (engine_selector,)


@app.cell
def _(available_sensors, mo):
    multiple_sensor_selector = mo.ui.multiselect(
        options=available_sensors,
        value=["sensor_2", "sensor_4", "sensor_11"],
        label="Sensors to display",
        max_selections=5,
    )
    return (multiple_sensor_selector,)


@app.cell
def _(mo):
    smoothing_window = mo.ui.slider(
        start=3, stop=50, step=1, value=20, label="Smooth window"
    )
    return (smoothing_window,)


@app.cell
def _(
    df_filtered,
    engine_selector,
    mo,
    multiple_sensor_selector,
    plt,
    smoothing_window,
):
    selected_engine_data = df_filtered[
        df_filtered["engine_id"] == engine_selector.value
    ].copy()

    selected_sensors = multiple_sensor_selector.value

    mo.stop(
        not multiple_sensor_selector.value,
        mo.md("Select at least one sensor to display"),
    )

    normalized_sensors = (
        selected_engine_data[selected_sensors]
        - selected_engine_data[selected_sensors].mean()
    ) / selected_engine_data[selected_sensors].std()

    smoothed_sensors = normalized_sensors.rolling(
        window=smoothing_window.value,
        center=True,
        min_periods=1,
    ).mean()

    _fig, _ax = plt.subplots(figsize=(15, 8))

    for sensor in selected_sensors:
        _ax.plot(
            selected_engine_data["cycle"],
            normalized_sensors[sensor],
            alpha=0.25,
            linewidth=1,
        )

        _ax.plot(
            selected_engine_data["cycle"],
            smoothed_sensors[sensor],
            linewidth=2,
            label=sensor,
        )

    _ax.axhline(0, linewidth=1)
    _ax.set_xlabel("Operating cycle")
    _ax.set_ylabel("Standardised sensor value")
    _ax.set_title(f"Sensor trajectories for engine {engine_selector.value}")
    _ax.grid(alpha=0.3)
    _ax.legend()

    mo.vstack(
        [
            mo.hstack(
                [
                    engine_selector,
                    multiple_sensor_selector,
                    smoothing_window,
                ],
                justify="start",
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    For the selected engine, several sensors show a visible monotonic evolution, while other appear noisier or more stable. The direction and intensity of theses trends must now be compared across several engines before they can be considered representative of fleet-wide degradation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Comparison across engines
    """)
    return


@app.cell
def _(df_filtered, mo):
    multiple_engine_selector = mo.ui.multiselect(
        options=df_filtered["engine_id"].unique().tolist(),
        value=[1, 50, 80],
        label="Engines to display",
        max_selections=5,
    )

    return (multiple_engine_selector,)


@app.cell
def _(available_sensors, mo):
    sensor_selector = mo.ui.dropdown(
        options=available_sensors,
        value="sensor_2",
        label="Sensor to display",
    )

    return (sensor_selector,)


@app.cell
def _(
    df_filtered,
    mo,
    multiple_engine_selector,
    plt,
    sensor_selector,
    smoothing_window,
):
    _selected_engines = multiple_engine_selector.value
    _selected_sensor = sensor_selector.value

    _selected_sensor_data = df_filtered[
        df_filtered["engine_id"].isin(_selected_engines)
    ].copy()

    _selected_sensor_data = _selected_sensor_data[
        ["engine_id", "cycle", _selected_sensor]
    ]

    _fig, _ax = plt.subplots(figsize=(15, 8))

    for _engine in _selected_engines:
        _engine_data = _selected_sensor_data[
            _selected_sensor_data["engine_id"] == _engine
        ]

        _sensor_values = _engine_data[_selected_sensor]
        _sensor_std = _sensor_values.std()

        if _sensor_std == 0:
            _normalized_values = _sensor_values * 0
        else:
            _normalized_values = (_sensor_values - _sensor_values.mean()) / _sensor_std

        _smoothed_values = _normalized_values.rolling(
            window=smoothing_window.value,
            center=True,
            min_periods=1,
        ).mean()

        _ax.plot(
            _engine_data["cycle"],
            _smoothed_values,
            linewidth=2,
            label=f"Engine {_engine}",
        )

        _ax.plot(
            _engine_data["cycle"],
            _normalized_values,
            alpha=0.25,
            linewidth=1,
        )

    _ax.axhline(0, linewidth=1)
    _ax.set_xlabel("Operating cycle")
    _ax.set_ylabel("Standardised sensor value")
    _ax.set_title(f"Different engine trajectories for {_selected_sensor}")
    _ax.grid(alpha=0.3)

    if _selected_engines:
        _ax.legend()

    mo.vstack(
        [
            mo.hstack(
                [
                    multiple_engine_selector,
                    sensor_selector,
                    smoothing_window,
                ],
                justify="start",
            ),
            _fig,
        ]
    )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visual findings across selected engines

    The comparison of several engines reveals that many retained sensors show a similar evolution throughout engin life. Sensors such as 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_8', 'sensor_11', 'sensor_13', 'sensor_15' and 'sensor_17' generally increase as the engines approach failure. In contrast, 'sensor_12', 'sensor_20', 'sensor_21' show predominantly decreasing trajectories. For these sensors, short_lived engines also appear to complete the same general evolution over fewer operating cycles, resulting in visually steeper trajectories.

    The behaviour of 'sensor_17', is particularly noteworthy. Although it contains only limited number of distinct values in the raw dataset, its trajectory still appears to evolve systematically with engine age. This observation supports the decision to retain low-variability sensors when their measurements may still contain a structured degradation signal.

    Other sensors show less consistent behavious across engines. 'sensor_14' appears to be decerasing, but its trend is weaker or less regular for some trajectories. 'sensor_9' is the clearest example of engine-specific behaviour. Depending on the selected engine, its measurements may increase, decrease, remain relatively stable, or change direction during engine lifetime. Repeating the visual comparison with different group of engines produces the same heterogeneous patter, suggesting that the relationship between the sensor and engine age may not be consistent across the fleet.

    These graphs use values that are standardised separately for each engine and smoothed with a rolling mean. Standardisation improves the comparison of trajectories shapes but removes differences in absolute sensor levels and amplitudes. Smoothing makes the underlying trends easier to observe but may also make then appear more regular than the original measurements. The visual results should therefore be interpreted as exploratory evidence rather than as definitive assessment of sensor relevance.

    The next step is to quantify these observations acorss all 100 engines. Per-engine Spearman correlations will be calculated from the original, unsmoothed sensor measurements to determine the direction, strength, and fleet-level consistency of the relationship between each sensor and the operating cycle.
    """)
    return


if __name__ == "__main__":
    app.run()
