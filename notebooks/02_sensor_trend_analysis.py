import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

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
    return available_sensors, df_filtered, mo, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # FD001 Sensor trend analysis

    This notebook investigates how the sensor measurements retained from the FD001 dataset evolve throughout engine life.

    The previous dataset overview identified and removed strictly and effectively constant sensor variables. The present analysis focuses on the remaining sensors and examines whether their evolution is associated with engine ageing.

    The analysis follows a progressive approach :

    1. Explore sensor trajectories for an individual engine.
    2. Compare the same sensor across several engines.
    3. Calculate a Spearman correlation for every engine-sensor pair.
    4. Assess the strength, direction, and fleet-level consistency of the observed sensor trends.

    The visual exploration is first used to formulate hypotheses. These hypotheses are then evaluated statistically across all 100 engine trajectories.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Individual engine exploration

    The first step is to examine how several sensor measurements evolve throughout the lifetime of a single engine.

    Because the sensors use different units and numerical scales, their values are standardised within the selected engine before being displayed on the same graph. This transformation preserves the direction and shape of each trajectory while allowing direct visual comparison between sensors.

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
    ### Initial visual observations

    For the selected engine, several sensors show a visible monotonic evolution, while others appear noisier or more stable. Some trajectories remain relatively flat during the early operating cycles and change more strongly near the end of engine life.

    These obersations suggest that some sensor measurements may reflect the progression of degradation. However, the direction, intensity, and timing of the trends must be compared across multiple engines before they can be considered representative of fleet-wide behaviour
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Comparison across engines

    The previous graph compares several sensors wothing one engine. The analysis is now reversed, on sensor is selected and its trajectory is compared across several engines.

    Each engine trajectory is standardised independently. This removes differences in absolute sensor levels and amplitudes and makes the shape of the trajectories easier to compare. The rolling mean is also calculated separalety for each engine to avoid maxing measurements belonging to different trajectories.

    The horizontal axis still represents the absolute operating cycle. Engines with shorter lifetimes therefore and earlier on the graph, while longer-lived engines extend further along the cycle axis.
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
    ### Visual Findings Across Selected Engines

    The comparison of several engines reveals that many retained sensors show a similar evolution throughout engine life. Sensors such as `sensor_2`, `sensor_3`, `sensor_4`, `sensor_8`, `sensor_11`, `sensor_13`, `sensor_15`, and `sensor_17` generally increase as the engines approach failure. In contrast, `sensor_7`, `sensor_12`, `sensor_20`, and `sensor_21` generally follow decreasing trajectories.

    For many of these sensors, shorter-lived engines appear to complete a similar overall evolution over fewer operating cycles, resulting in visually steeper trajectories. This observation may indicate differences in degradation rate, although it cannot be confirmed from a small selection of engines alone.

    The behaviour of `sensor_17` is particularly noteworthy. Although it contains only a limited number of distinct values in the raw dataset, its trajectory still appears to evolve systematically with engine age. This supports the decision to retain low-variability sensors when their measurements may contain a structured degradation signal.

    Other sensors show less consistent behaviour. `sensor_9` is the clearest example of engine-specific evolution: depending on the selected engine, its measurements may increase, decrease, remain relatively stable, or change direction during the trajectory. `sensor_14` also appears less regular than most of the other retained sensors. Repeating the comparison with a different groups of engines suggests that these heterogeneous patterns are not limited to a single selection.

    These graphs use values that are standardised separately for each engine and smoothed with a rolling mean. Standardisation improves the comparison of trajectory shapes but removes differences in absolute sensor levels and amplitudes. Smoothing makes the underlying trends easier to observe, but may also make them appear more regular than the original measurements.

    The visual results should therefore be interpreted as exploratory evidence rather than as a definitive assessment of sensor relevance. The next step is to quantify these observations across all 100 engines using the original, unsmoothed sensor measurements.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Per-engine Spearman Correlation

    The visual comparisons suggest that several sensors follow monotonic trends, but thet only display a small subset of the engine fleet. A quatitative measure is therefore required to evalutate every retained sensor across all 100 engines.

    For each engine-sensor pari, the Spearman rank correlation coefficient is calculated between the operating cycles and the orginal sensor measurements. This produces 100 coeficients for each of the 14 retained sensors, resulting in a total of 1,400 engine-sensor observations.

    Spearman correlation measures the strength and direction of a monotonic relationship:

    - a coefficient close to 1 indicates that the sensor generally increases with the operating cycle,
    - a coefficient close to -1 inficates that the sensor generally decreases,
    - a coefficient close to 0 indicates that no clear monotonic relationship is present.

    The calculation is performed on the original, unsmoothed measurements. Standardisation is unnecessary because it does not change the ordering of the values, while smoothing could artificially make a trajectory appear more monotonic.
    """)
    return


@app.cell
def _(available_sensors, mo):
    sensor_selector_for_spearman = mo.ui.dropdown(
        options=available_sensors,
        value="sensor_2",
        label="Sensor to display",
    )

    return (sensor_selector_for_spearman,)


@app.cell
def _(available_sensors, df_filtered, pd):
    _spearman_results = []

    for _sensor in available_sensors:
        for _engine_id in df_filtered["engine_id"].unique():
            _engine_data = df_filtered[df_filtered["engine_id"] == _engine_id]

            _rho = _engine_data["cycle"].corr(
                _engine_data[_sensor],
                method="spearman",
            )

            _spearman_results.append(
                {
                    "engine_id": _engine_id,
                    "sensor": _sensor,
                    "spearman_rho": float(_rho),
                }
            )

    spearman_by_engine = pd.DataFrame(_spearman_results)

    return (spearman_by_engine,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Fleet-level sensor summary

    The 100 coefficients obtained for each sensor must be summarised without hiding differences between engines. A simple arithmetic mean would be insufficient, therefore, several indicators and calculated for each sensor :

    - **Median rho** describes the typical directin and strength of the correlation.
    - **Median absolute rho** desbribes the typical correlation strength independently of direction.
    - ** First and Third quartiles** delimit the central 50% of the engine-level coefficients.
    - **Interquartil range (IQR)** measures the dispersion of these central coefficients. A small IQR indicates similar behaviour across engines, whereas a large IQR indicates heterogenous trajectories.
    - **Positive and negative shares** indicate the proportion of engines for which the sensor increases or decreases with the operating cycle.
    - **Direction consistency** is the larger consistent direction across the fleet, while a value close to 0.5 indicates an approximately even split between positive and negative relationships.

    Together, these indicators distinguish sensors that are strongly and consistently associated with engine age from sensors whose behaviour depends on the individual engine.
    """)
    return


@app.cell
def _(mo, sensor_selector_for_spearman, spearman_by_engine):
    selected_sensor_spearman = (
        spearman_by_engine[
            spearman_by_engine["sensor"] == sensor_selector_for_spearman.value
        ]
        .copy()
        .round(3)
    )

    mo.vstack(
        [
            sensor_selector_for_spearman,
            selected_sensor_spearman.round(3),
        ]
    )
    return


@app.cell
def _(spearman_by_engine):
    spearman_summary = (
        spearman_by_engine.assign(absolute_rho=lambda df: df["spearman_rho"].abs())
        .groupby("sensor")
        .agg(
            median_rho=("spearman_rho", "median"),
            median_absolute_rho=("absolute_rho", "median"),
            q1_rho=("spearman_rho", lambda values: values.quantile(0.25)),
            q3_rho=("spearman_rho", lambda values: values.quantile(0.75)),
            positive_share=("spearman_rho", lambda values: (values > 0).mean()),
            negative_share=("spearman_rho", lambda values: (values < 0).mean()),
        )
        .reset_index()
    )

    spearman_summary["iqr_rho"] = (
        spearman_summary["q3_rho"] - spearman_summary["q1_rho"]
    )

    spearman_summary["direction_consistency"] = spearman_summary[
        ["positive_share", "negative_share"]
    ].max(axis=1)

    spearman_summary = spearman_summary.sort_values(
        by=["direction_consistency", "median_absolute_rho"],
        ascending=False,
    )

    spearman_summary.round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Main findings from the spearman analysis

    The fleet-level analysis shows that 12 of the 14 retained sensors follow a consistent monotonic relationship with engine age. For these sensors, the direction of the relationship is identical across all 100 engines, some measurements consistently increase with the operating cycle, while others consistently decrease.

    The strongest and most stable relationships are observed for sensors such a `sensor_11`, `sensor_12`, `sensor_4`, and `sensor_7`. Their high median absolute correlations, perfect direction consistency, and small interquartile ranges indicate that their behaviour is highly repeatable across the fleet.

    `sensor_9` and `sensor_14` differ from the other sensors. Their correlations are often strong for individual engines, but the direction varies across the fleet. They should therefore not be interpreted as having no relationship with engine age, but rather as displaying engine-dependent behaviour that cannot be represented by one consistent monotonic trend.

    Overall, the analysis identifies a large group of sensors that reliably track engine ageing and two sensors whose behaviour requires a different or more context-dependent interpretation.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
