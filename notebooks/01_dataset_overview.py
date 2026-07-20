import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    return (pd,)


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
    id_colomns = ["engine_id", "cycle"]

    operational_columns = [f"operational_setting_{i}" for i in range(1, 4)]

    sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

    column_names = id_colomns + operational_columns + sensor_columns
    return (column_names,)


@app.cell
def _(column_names, df_raw):
    df_raw.columns = column_names
    df_raw.head()
    return


@app.cell
def _(df_raw):
    df_raw
    return


@app.cell
def _(df_raw):
    df_raw.shape
    return


if __name__ == "__main__":
    app.run()
