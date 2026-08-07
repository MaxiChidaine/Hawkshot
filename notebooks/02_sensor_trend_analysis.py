import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # FD001 Sensor trend analysis

    This notebook investigates how the sensor measurements retained from the FD001 dataset evolve throughout engine life.

    The previous dataset overview identified and removed strictly and effectively constant sensor variables. The present analysis focuses on the remaining sensors and examines whether their evolution is associated with engine ageing.

    The analysis follows a progressive approach:

    1. Explore several sensor trajectories for an individual engine.
    2. Compare the same sensor across multiple engines.
    3. Calculate a Spearman correlation for every engine-sensor pair.
    4. Assess the strength, direction, and fleet-level consistency of the observed
       sensor trends.
    5. Investigate whether the retained sensors carry distinct or partly redundant
       degradation information.
    6. Examine how sensor trend strength and consistency evolve across different
       stages of engine life.

    The visual exploration is first used to formulate hypotheses. These hypotheses
    are then evaluated statistically across all 100 engine trajectories, and each
    result is used to motivate the next analytical question.
    """)
    return


@app.cell
def _():
    from itertools import combinations

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
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
    return available_sensors, combinations, df_filtered, mo, np, pd, plt


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

    These observations suggest that some sensor measurements may reflect the progression of degradation. However, the direction, intensity, and timing of the trends must be compared across multiple engines before they can be considered representative of fleet-wide behaviour
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Comparison Across Engines

    The previous graph compares several sensors within one engine. The analysis is now reversed, one sensor is selected and its trajectory is compared across several engines.

    Each engine trajectory is standardised independently. This removes differences in absolute sensor levels and amplitudes and makes the shapes of the trajectories easier to compare. The rolling mean is also calculated separately for each engine to avoid mixing measurements belonging to different trajectories.

    The horizontal axis still represents the absolute operating cycle. Engines with shorter lifetimes therefore end earlier on the graph, while longer-lived engines extend further along the cycle axis.
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

    The comparison of several engines reveals that many retained sensors show a similar evolution throughout engine life. Sensors such as `sensor_2`, `sensor_3`, `sensor_4`, `sensor_8`, `sensor_11`, `sensor_13`, `sensor_15`, and `sensor_17` generally increase as the engines approach failure. In contrast, `sensor_7`, `sensor_12`, `sensor_20`, and `sensor_21` generally follow decreasing trajectories.

    For many of these sensors, shorter-lived engines appear to complete a similar overall evolution over fewer operating cycles, resulting in visually steeper trajectories. This observation may indicate differences in degradation rate, although it cannot be confirmed from a small selection of engines alone.

    The behaviour of `sensor_17` is particularly noteworthy. Although it contains only a limited number of distinct values in the raw dataset, its trajectory still appears to evolve systematically with engine age. This supports the decision to retain low-variability sensors when their measurements may contain a structured degradation signal.

    Other sensors show less consistent behaviour. `sensor_9` is the clearest example of engine-specific evolution: depending on the selected engine, its measurements may increase, decrease, remain relatively stable, or change direction during the trajectory. `sensor_14` also appears less regular than most of the other retained sensors. Repeating the comparison with different groups of engines suggests that these heterogeneous patterns are not limited to a single selection.

    These graphs use values that are standardised separately for each engine and smoothed with a rolling mean. Standardisation improves the comparison of trajectory shapes but removes differences in absolute sensor levels and amplitudes. Smoothing makes the underlying trends easier to observe, but may also make them appear more regular than the original measurements.

    The visual results should therefore be interpreted as exploratory evidence rather than as a definitive assessment of sensor relevance. The next step is to quantify these observations across all 100 engines using the original, unsmoothed sensor measurements.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Per-Engine Spearman Correlations

    The visual comparisons suggest that several sensors follow monotonic trends,
    but they display only a small subset of the engine fleet. A quantitative
    measure is therefore required to evaluate every retained sensor across all
    100 engines.

    For each engine-sensor pair, the Spearman rank correlation coefficient is
    calculated between the operating cycle and the original sensor measurements.
    This produces 100 coefficients for each of the 14 retained sensors, resulting
    in a total of 1,400 engine-sensor observations.

    Spearman correlation measures the strength and direction of a monotonic
    relationship:

    - a coefficient close to `1` indicates that the sensor generally increases
      with the operating cycle;
    - a coefficient close to `-1` indicates that the sensor generally decreases;
    - a coefficient close to `0` indicates that no clear monotonic relationship
      is present.

    The calculation is performed on the original, unsmoothed measurements.
    Standardisation is unnecessary because it does not change the ordering of the
    values, while smoothing could artificially make a trajectory appear more
    monotonic.
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
            selected_sensor_spearman,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Fleet-Level Sensor Summary

    The 100 coefficients obtained for each sensor must be summarised without
    hiding differences between engines. A simple arithmetic mean could be
    misleading because strong positive and negative correlations may cancel one
    another. Several complementary indicators are therefore calculated:

    - **Median rho** describes the typical direction and strength of the
      relationship.
    - **Median absolute rho** describes its typical strength independently of
      direction.
    - **First and third quartiles** delimit the central 50% of the engine-level
      coefficients.
    - **Interquartile range (IQR)** measures the dispersion of the central
      coefficients. A small IQR indicates similar behaviour across engines,
      whereas a large IQR indicates heterogeneous trajectories.
    - **Positive and negative shares** indicate the proportions of engines for
      which the sensor increases or decreases with the operating cycle.
    - **Direction consistency** is defined as the larger of the positive and
      negative shares. A value close to `1` indicates a consistent direction
      across the fleet, while a value close to `0.5` indicates an approximately
      even split between positive and negative relationships.

    Together, these indicators distinguish sensors that are strongly and
    consistently associated with engine age from sensors whose behaviour depends
    on the individual engine.
    """)
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

    spearman_summary_display = spearman_summary[
        [
            "sensor",
            "median_rho",
            "median_absolute_rho",
            "iqr_rho",
            "positive_share",
            "negative_share",
            "direction_consistency",
        ]
    ]

    spearman_summary_display.round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Main findings from the Spearman analysis

    The fleet-level analysis shows that 12 of the 14 retained sensors follow a consistent monotonic relationship with engine age. For these sensors, the direction of the relationship is identical across all 100 engines, some measurements consistently increase with the operating cycle, while others consistently decrease.

    The strongest and most stable relationships are observed for sensors such as `sensor_11`, `sensor_12`, `sensor_4`, and `sensor_7`. Their high median absolute correlations, perfect direction consistency, and small interquartile ranges indicate that their behaviour is highly repeatable across the fleet.

    `sensor_9` and `sensor_14` differ from the other sensors. Their correlations are often strong for individual engines, but the direction varies across the fleet. They should therefore not be interpreted as having no relationship with engine age, but rather as displaying engine-dependent behaviour that cannot be represented by one consistent monotonic trend.

    Overall, the analysis identifies a large group of sensors that reliably track engine ageing and two sensors whose behaviour requires a different or more context-dependent interpretation.

    These results identify which sensors track engine age consistently. However, they do not show whether the 12 coherent sensors provide distinct information or whether several of them reflect the same underlying age-related signal.

    This result motivates the next question, how strongly are the retained sensors related to one another within individual engine trajectories?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Inter-sensor relationships and potential redundancy

    The previous analysis identified several sensors that follow engine age through strong and consistent monotonic trends. However, this does not necessarily mean that each sensor provides distinct information.

    Sensors that evolve similarly throughout engine life may be strongly correlated with one another and may therefore contain partly redundant degradation signals. Conversely, sensors with weaker inter-sensor relationships may provide more complementary information.

    To investigate this question, Spearman correlations are calculated between every pair of retained sensors within each engine trajectory. Computing the relationships separately for each engine preserves the time-series structure of the dataset and avoids treating all 20,631 observations as independent.

    With 14 retained sensors, 91 unique sensor pairs are evaluated across all 100 engines, producing 9,100 pairwise correlation coefficients.

    This analysis is exploratory. A strong correlation between two sensors may reflect a shared response to engine ageing rather than identical physical information, and it does not automatically justify removing either variable.
    """)
    return


@app.cell
def _(available_sensors, combinations, df_filtered, pd):
    sensor_pairs = list(combinations(available_sensors, 2))

    _inter_sensor_results = []

    for _sensor_a, _sensor_b in sensor_pairs:
        for _engine_id in df_filtered["engine_id"].unique():
            _engine_data = df_filtered[df_filtered["engine_id"] == _engine_id]

            _rho = _engine_data[_sensor_a].corr(
                _engine_data[_sensor_b],
                method="spearman",
            )

            _inter_sensor_results.append(
                {
                    "engine_id": _engine_id,
                    "sensor_a": _sensor_a,
                    "sensor_b": _sensor_b,
                    "spearman_rho": _rho,
                }
            )

    inter_sensor_spearman_by_engine = pd.DataFrame(_inter_sensor_results)
    return (inter_sensor_spearman_by_engine,)


@app.cell
def _(inter_sensor_spearman_by_engine):
    inter_sensor_summary = (
        inter_sensor_spearman_by_engine.assign(
            absolute_rho=lambda df: df["spearman_rho"].abs()
        )
        .groupby(["sensor_a", "sensor_b"])
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

    inter_sensor_summary["iqr_rho"] = (
        inter_sensor_summary["q3_rho"] - inter_sensor_summary["q1_rho"]
    )

    inter_sensor_summary["direction_consistency"] = inter_sensor_summary[
        ["positive_share", "negative_share"]
    ].max(axis=1)

    inter_sensor_summary = inter_sensor_summary.sort_values(
        by=[
            "median_absolute_rho",
            "direction_consistency",
        ],
        ascending=False,
    )
    return (inter_sensor_summary,)


@app.cell
def _(available_sensors, inter_sensor_summary, pd):
    heatmap_matrix = pd.DataFrame(
        1.0,
        index=available_sensors,
        columns=available_sensors,
    )

    for _, _row in inter_sensor_summary.iterrows():
        _sensor_a = _row["sensor_a"]
        _sensor_b = _row["sensor_b"]
        _rho = _row["median_rho"]

        heatmap_matrix.loc[_sensor_a, _sensor_b] = _rho
        heatmap_matrix.loc[_sensor_b, _sensor_a] = _rho
    return (heatmap_matrix,)


@app.cell
def _():
    sensor_order = [
        # Consistent increasing sensors
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_8",
        "sensor_11",
        "sensor_13",
        "sensor_15",
        "sensor_17",
        # Consistent decreasing sensors
        "sensor_7",
        "sensor_12",
        "sensor_20",
        "sensor_21",
        # Heterogeneous sensors
        "sensor_9",
        "sensor_14",
    ]
    return (sensor_order,)


@app.cell
def _(heatmap_matrix, np, pd, plt, sensor_order):
    _heatmap_data = heatmap_matrix.loc[
        sensor_order,
        sensor_order,
    ].copy()

    # Hide the diagonal and the redundant upper triangle

    _mask = np.triu(np.ones_like(_heatmap_data, dtype=bool))

    _heatmap_display = _heatmap_data.mask(_mask)

    _cmap = plt.get_cmap("RdBu_r").copy()
    _cmap.set_bad("white")

    _fig, _ax = plt.subplots(figsize=(11, 9))

    _image = _ax.imshow(_heatmap_display, vmin=-1, vmax=1, cmap=_cmap)

    _ax.set_xticks(range(len(sensor_order)))
    _ax.set_xticklabels(
        sensor_order,
        rotation=45,
        ha="right",
    )

    _ax.set_yticks(range(len(sensor_order)))
    _ax.set_yticklabels(sensor_order)

    _ax.set_title(
        "Median Spearman correlation between sensor pairs",
        pad=18,
    )

    # Display the numerical value in each visible cell

    for _row_index in range(len(sensor_order)):
        for _column_index in range(len(sensor_order)):
            _value = _heatmap_display.iloc[
                _row_index,
                _column_index,
            ]
            if pd.notna(_value):
                _text_color = "white" if abs(_value) >= 0.6 else "black"

                _ax.text(
                    _column_index,
                    _row_index,
                    f"{_value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=_text_color,
                )

    for _boundary in [7.5, 11.5]:
        _ax.axhline(_boundary, color="black", linewidth=0.8)
        _ax.axvline(_boundary, color="black", linewidth=0.8)

    _colorbar = _fig.colorbar(_image, ax=_ax, shrink=0.8)

    _colorbar.set_label("Median Spearman rho")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interim conclusion and next question

    The heatmap reveals two broad directional groups. Sensors that increase with engine age are positively correlated with one another, as are sensors that decrease with age. Relationships between the two groups are predominantly negative, which is consistent with a shared age-related pattern expressed in opposite directions.

    The sensors are ordered according to the direction and consistency identified in the previous analysis. The visible block structure is therefore a descriptive organisation rather than the result of an independent clustering algorithm.

    The maximum median absolute pairwise correlation is approximately 0.71. This indicates substantial shared information, but no near-perfect redundancy between any pair of sensors. The current analysis therefore provides no clear justification for removing a sensor solely because it duplicates another one.

    `sensor_9` and `sensor_14` must still be interpreted cautiously. Their median pairwise relationships appear moderate to strong, but the previous engine-level results showed considerable variation in sign and magnitude across the fleet.

    The next question is therefore not only whether the sensors are related to engine age, but whether they become informative at the same stage of engine life.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Sensor trends across engine life phases

    The previous analyses evaluated sensor trends over complete engine trajectories. However, a strong overall relationship with engine age does not indicate when the degradation signal becomes visible.

    Some sensors may evolve progressively throughout engine life, whereas others may remain relatively stable before changing more strongly near failure. These behaviours could have different implications for Remaining Useful Life prediction.

    - **Early life:** 0% to 50%
    - **Intermediate life:** 50% to 80%
    - **Late life:** 80% to 100%

    Relative life position is used only for exploratory analysis because the final lifetime of an operating engine would not be known in a real predictive maintenance application.
    """)
    return


@app.cell
def _(df_filtered, pd):
    df_life_phases = df_filtered.copy()

    df_life_phases["max_cycle"] = df_life_phases.groupby("engine_id")[
        "cycle"
    ].transform("max")

    df_life_phases["relative_cycle"] = (
        df_life_phases["cycle"] / df_life_phases["max_cycle"]
    )

    df_life_phases["life_phase"] = pd.cut(
        df_life_phases["relative_cycle"],
        bins=[0, 0.5, 0.8, 1.0],
        labels=["early", "intermediate", "late"],
        include_lowest=True,
    )

    df_life_phases[
        ["engine_id", "cycle", "max_cycle", "relative_cycle", "life_phase"]
    ].head()
    return (df_life_phases,)


@app.cell
def _(available_sensors, df_life_phases, pd):
    _phase_spearman_results = []

    for _sensor in available_sensors:
        for _engine_id in df_life_phases["engine_id"].unique():
            _engine_data = df_life_phases[df_life_phases["engine_id"] == _engine_id]

            for _phase in ["early", "intermediate", "late"]:
                _phase_data = _engine_data[_engine_data["life_phase"] == _phase]

                _rho = _phase_data["cycle"].corr(
                    _phase_data[_sensor],
                    method="spearman",
                )

                _phase_spearman_results.append(
                    {
                        "engine_id": _engine_id,
                        "sensor": _sensor,
                        "life_phase": _phase,
                        "spearman_rho": _rho,
                    }
                )

    spearman_by_life_phase = pd.DataFrame(_phase_spearman_results)
    return (spearman_by_life_phase,)


@app.cell
def _(spearman_by_life_phase):
    life_phase_summary = (
        spearman_by_life_phase.assign(absolute_rho=lambda df: df["spearman_rho"].abs())
        .groupby(
            ["sensor", "life_phase"],
            observed=True,
        )
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

    life_phase_summary["iqr_rho"] = (
        life_phase_summary["q3_rho"] - life_phase_summary["q1_rho"]
    )

    life_phase_summary["direction_consistency"] = life_phase_summary[
        ["positive_share", "negative_share"]
    ].max(axis=1)

    life_phase_summary.round(3)
    return (life_phase_summary,)


@app.cell
def _(available_sensors, life_phase_summary):
    phase_order = ["early", "intermediate", "late"]

    phase_strength_matrix = life_phase_summary.pivot(
        index="sensor",
        columns="life_phase",
        values="median_absolute_rho",
    ).reindex(
        index=available_sensors,
        columns=phase_order,
    )
    return phase_order, phase_strength_matrix


@app.cell
def _(phase_order, phase_strength_matrix, plt):
    _fig, _ax = plt.subplots(figsize=(7, 8))

    _image = _ax.imshow(
        phase_strength_matrix,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        aspect="auto",
    )

    _ax.set_xticks(range(len(phase_order)))
    _ax.set_xticklabels(
        ["Early", "Intermediate", "Late"],
    )

    _ax.set_yticks(range(len(phase_strength_matrix)))
    _ax.set_yticklabels(phase_strength_matrix.index)

    _ax.set_title(
        "Median absolute Spearman correlation across life phases",
        pad=16,
    )

    for _row_index in range(len(phase_strength_matrix.index)):
        for _column_index in range(len(phase_order)):
            _value = phase_strength_matrix.iloc[
                _row_index,
                _column_index,
            ]

            _text_color = "white" if _value >= 0.6 else "black"

            _ax.text(
                _column_index,
                _row_index,
                f"{_value:.2f}",
                ha="center",
                va="center",
                color=_text_color,
                fontsize=8,
            )
    _colorbar = _fig.colorbar(
        _image,
        ax=_ax,
        shrink=0.8,
    )

    _colorbar.set_label("Median absolute Spearman rho")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Strength of sensor trends across engine life

    The median absolute correlations reveal a common pattern across all retained sensors: the monotonic relationship with the operating cycle becomes progressively stronger as engines approach failure.

    During early life, the median absolute correlations remain relatively weak, ranging from approximately 0.13 to 0.28. They increase during the intermediate phase, reaching values between 0.32 and 0.57, and become substantially stronger during late life, where they range from 0.54 to 0.85.

    This progression suggests that the degradation signal is not distributed uniformly throughout engine life. Sensor measurements contain relatively limited monotonic information during the first half of the trajectories, while their evolution becomes increasingly structured during the intermediate and late phases.

    `sensor_11, `sensor_12`, `sensor_4`, and `sensor_7` already show some of the strongest relationships during early and intermediate life and remain highly informative during the late phase. In contrast, several sensors such as `sensor_2`, `sensor_3` and `sensor_17` show particularly weak early-life relationships before becoming more strongly associated with operating cycle near failure.

    `sensor_9` and `sensor_14` present the strongest late-life absolute correlations, with the median values of approximately 0.80 and 0.85 respectively. However, these results must be interpreted cautiously. Absolute correlation measures the strength of a relationship independently of its direction. A high value may therefore describe either a strong increasing trend or a strong decreasing trend, and it does not indicate whether the same direction is observed across all engines.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Local trend consistency across engines

    The signed median correlations preserve the same overall direction identifies over complete engine trajectories, increasing sensors remain positive, while decreasing sensors remain negative across the three life phases.

    Because this result largely confirms the previous full-life analysis, an additional signed-correlation heatmap is not required. The remaining question is whether the typical direction is shared consistently across all engines.

    Direction consistency is therefore examined at the phase level. This analysis reveals whether a strong local trend represents a common fleet-wide behaviour or whether different engines evolve in opposite directions during the same life phase.
    """)
    return


@app.cell
def _(life_phase_summary):
    unstable_phase_trends = life_phase_summary[
        life_phase_summary["direction_consistency"] < 0.9
    ][
        [
            "sensor",
            "life_phase",
            "median_rho",
            "median_absolute_rho",
            "direction_consistency",
        ]
    ].sort_values(
        by="direction_consistency",
        ascending=True,
    )

    unstable_phase_trends
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Strengthening of sensor trends between life phases

    The phase level consistency analysis shows whether the dominant direction of a sensor trend is shared across engines. However, it does not indicate when the relationship with the engine age becomes substantially stronger.

    A sensor may present a relatively stable direction throughout engine life while its monotonic signal emerges progressively or becomes pronounced only near failure. Distinguishing these temporal behaviours may help to identify sensors that act as gradual ageing indicators and sensors that are primarily sensitive to advanced degradation.

    The phase-to-phase differences in median absolute Spearman correlation are therefore calculated :

    - **Intermediate - early** measures how strongly the signal develops after the first half of engine life.
    - **Late - intermediate** measures the additional strengthening occurring near failure.
    - **Late - early** represents the total increase in trend strength across the analysed trajectory.

    These differences quantify the timing of signal emergence. They do not, by themselves, establish whether the underlying physical mechanism corresponds to linear wear, a threshold effect, controller response, or an abrupt anomaly.
    """)
    return


@app.cell
def _(phase_strength_matrix):
    phase_delta_summary = phase_strength_matrix.copy()

    phase_delta_summary["delta_intermediate_early"] = (
        phase_delta_summary["intermediate"] - phase_delta_summary["early"]
    )
    phase_delta_summary["delta_late_intermediate"] = (
        phase_delta_summary["late"] - phase_delta_summary["intermediate"]
    )
    phase_delta_summary["delta_late_early"] = (
        phase_delta_summary["late"] - phase_delta_summary["early"]
    )

    phase_delta_display = phase_delta_summary[
        [
            "delta_intermediate_early",
            "delta_late_intermediate",
            "delta_late_early",
        ]
    ].sort_values(
        by="delta_late_early",
        ascending=False,
    )

    phase_delta_display.round(3)
    return (phase_delta_summary,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Exploratory temporal sensor profiles

    The phase-to-phase deltas indicate when the monotonic relationship between each sensor and engine age becomes stronger. Direction consistency indicates whether the dominant local trend is shared across engines.

    These two dimensions are now combined to provide a compact description of each sensor`s temporal behaviour:

    - The **timing dimension** distinguishes signals that strengthen mainly before the late phase, mainly during the transition to late life, or relatively evenly across both transitions.
    - The **consistency dimension** distinguishes behaviours that are broadly shared across the fleet from behaviours that depend more strongly on the individual engine.

    The timing profile is derived from the proportion of the total strengthening that occurs between the intermediate and the late phases. The consistency profile is derived from the lowest direction consistency observed across the three life phases.

    The resulting profiles are exploratory labels rather than data-driven clusters or validated physical categories. Their boundaries are defined using heuristic thresholds chosen to support interpretation, and sensors close to a boundary could change category if the thresholds were modified. The underlying continuous values therefore remain the primary results, while the labels are used only as concise descriptive summaries.
    """)
    return


@app.cell
def _(available_sensors, life_phase_summary, phase_delta_summary, phase_order):
    phase_consistency_matrix = (
        life_phase_summary.pivot(
            index="sensor",
            columns="life_phase",
            values="direction_consistency",
        )
        .reindex(
            index=available_sensors,
            columns=phase_order,
        )
        .rename(
            columns={
                "early": "consistency_early",
                "intermediate": "consistency_intermediate",
                "late": "consistency_late",
            }
        )
    )

    sensor_temporal_profiles = phase_delta_summary[
        [
            "delta_intermediate_early",
            "delta_late_intermediate",
            "delta_late_early",
        ]
    ].join(phase_consistency_matrix)

    sensor_temporal_profiles["late_strengthening_share"] = (
        sensor_temporal_profiles["delta_late_intermediate"]
        / sensor_temporal_profiles["delta_late_early"]
    )

    sensor_temporal_profiles["minimum_direction_consistency"] = (
        sensor_temporal_profiles[
            [
                "consistency_early",
                "consistency_intermediate",
                "consistency_late",
            ]
        ].min(axis=1)
    )
    return (sensor_temporal_profiles,)


@app.cell
def _(mo):
    mo.md(r"""
    For the timing profile, a late-strengthening share below 0.45 is labelled **early-to-intermediate dominant**, a value above 0.55 is labelled **late-dominant**, and values between these limits are labelled **balanced progression**. The interval from 0.45 to 0.55 provides a small tolerance around an equal distribution of strengthening between both transitions.

    For the consistency profile, a minimum direction consistency of at least 0.80 is labelled **stable across engines**, a value between 0.65 and 0.80 is labelled **moderately heterogeneous**, and a value below 0.65 is labelled **engine-dependent**.

    These thresholds are not physical constants and should not be interpreted as formal decision boundaries.
    """)
    return


@app.cell
def _():
    def classify_timing(_row):
        _late_share = _row["late_strengthening_share"]

        if _late_share > 0.55:
            return "Late-dominant"
        elif _late_share < 0.45:
            return "Early-to-intermediate dominant"

        return "Balanced progression"

    def classify_consistency(_values):
        if _values >= 0.8:
            return "Stable across engines"

        elif _values >= 0.65:
            return "Moderately heterogeneous"

        return "Engine-dependent"

    return classify_consistency, classify_timing


@app.cell
def _(classify_consistency, classify_timing, sensor_temporal_profiles):
    sensor_temporal_profiles["timing_profile"] = sensor_temporal_profiles.apply(
        classify_timing,
        axis=1,
    )

    sensor_temporal_profiles["consistency_profile"] = sensor_temporal_profiles[
        "minimum_direction_consistency"
    ].apply(classify_consistency)
    return


@app.cell
def _(sensor_temporal_profiles):
    sensor_temporal_profile_display = sensor_temporal_profiles[
        [
            "timing_profile",
            "late_strengthening_share",
            "delta_intermediate_early",
            "delta_late_intermediate",
            "consistency_profile",
            "minimum_direction_consistency",
            "consistency_late",
        ]
    ].rename(
        columns={
            "late_strengthening_share": "late_share",
            "delta_intermediate_early": "delta_mid_early",
            "delta_late_intermediate": "delta_late_mid",
            "minimum_direction_consistency": "min_consistency",
            "consistency_late": "late_consistency",
        }
    )
    sensor_temporal_profile_display.round(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interpretation of the exploratory temporal profiles

    The phase-level consistency shows that most retained sensors follow highly repeatable temporal behaviours across the fleet. Twelve of the fourteen sensors are classified as **stable across engines** according to the exploratory consistency thresholds. Only `sensor_9` is classified as moderately heterogeneous, while `sensor_14` remains clearly engine-dependent.

    The timing profiles nevertheless reveal several distinct ways in which the degradation signal develops throughout engine life.

    A first timing group follows a **balanced progression**. `sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_8`, `sensor_13`, `sensor_14`, and `sensor_17` gain comparable amounts of monotonic strength during both phase transitions. Their signal therefore appears to strengthen progressively rather than emerging primarily during one specific stage.

    Most sensors in this group also show highly consistent directions across the fleet. `sensor_14` is the notable exception and is discussed separately below.

    A second group, composed of `sensor_11`, `sensor_12` and `sensor_15`, is classified as **early-to-intermediate dominant**. These sensors gain a larger share of their total trend strength before the late phase. In particular, `sensor_11` and `sensor_12` already present some of the strongest early-life relationships observed in the dataset. They may therefore be especially valuable as candidates for tracking degradation before the engine reaches an advanced stage.

    A third group is **late-dominant**. `sensor_20`, `sensor_21` and `sensor_9` experience a larger proportion of their strengthening during the transition from intermediate to late life. These sensors may therefore be more sensitive to advanced degradation or to changes that become increasingly visible as failure approaches.

    `sensor_9` is particularly noteworthy. Its signal becomes very strong during late life, but its direction is less consistent across engines than for most other sensors. This suggests that it may contain useful late-stage degradation information while responding differently across individual engine trajectories.

    `sensor_14` presents a different profile. Its strengthening is relatively balanced across life phases and its late-life absolute correlation is the strongest of all retained sensors, yet its direction remains strongly engine-dependent. This indicates that `sensor_14` carries a pronounced degradation-related signal at the individual-engine level without following one common fleet-wide direction.

    Overall, the combined timing and consistency analysis distinguishes several temporal roles among the retained sensors, progressive and repeatable ageing signals, signals that strengthen relatively early, signals that become particularly informative near failure, and a small number of engine-dependent responses.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Conclusion and modelling implications

    This notebook progressively investigated how the 14 sensor variables retained from FD001 evolve throughout engine life and whether their behaviour provides meaningful degradation information.

    The initial visual exploration revealed that most sensors exhibit structured trajectories as engines approach failure. This observation was confirmed across the complete fleet using per-engine Spearman correlations. Twelve of the fourteen retained sensors show a consistent full-life monotonic direction across all 100 engines, while `sensor_9` and `sensor_14` exhibit more engine-dependent behaviour despite often presenting strong individual relationships with engine age.

    The inter-sensor analysis showed that several variables share a common age-related signal. Increasing sensors are generally positively correlated with one another, decreasing sensors form a similar group, and relationships between the two groups are predominantly negative. However, the strongest median absolute pairwise correlation remains around 0.71. No sensor pair therefore appears sufficiently redundant to justify removing one variable solely on the basis of pairwise correlation.

    The life-phase analysis provides an additional temporal perspective. For every retained sensor, the monotonic relationship with engine age becomes substantially stronger as failure approaches. Early-life relationships remain relatively weak, strengthen during the intermediate phase, and become strongest during late life. The phase-to-phase differences further show that this strengthening does not occur at the same rate for every sensor. Some signals develop progressively, whereas others become much more pronounced during advanced degradation.

    Together, these findings suggest that the retained sensors do not all play the same temporal role. Some provide gradual ageing information, while others may be particularly informative near failure. The exploratory temporal profiles help describe these differences, but their phase boundaries and classification thresholds are heuristic rather than physically validated categories.

    No additional sensor is removed at this stage. Correlation with engine age, pairwise redundancy, and temporal behaviour describe the structure of the sensor signals, but they do not establish their actual predictive contribution to Remaining Useful Life estimation. A sensor that appears redundant may still provide complementary information to a model, while an individually strong sensor may add little once other variables are available.

    The next stage will therefore shift from exploratory analysis to target and feature preparation. The RUL target will be constructed, data will be separated by engine to prevent information leakage, and the retained sensors will initially be evaluated together in a baseline model. Subsequent model comparisons and ablation experiments will then determine which sensor combinations provide complementary predictive information.

    Finally, the relative life position and life-phase labels introduced in this notebook are strictly exploratory variables. They depend on the final lifetime of each engine and must not be used as predictors in a real RUL model, where future failure time is unknown. The phase boundaries themselves are also exploratory. Because the early, intermediate, and late phases cover different proportions of engine life, their local Spearman correlations are estimated from different numbers of operating cycles.
    """)
    return


if __name__ == "__main__":
    app.run()
