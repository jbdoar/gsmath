"""Presentation-oriented tabular reports built from export DataFrames."""

def report(
    value,
    *,
    precision=4,
    include_mode=False,
    include_uncertainty=True,
    include_units=False,
):
    """Create a compact rounded report from a summary or comparison."""
    from .export import to_dataframe

    dataframe = to_dataframe(value, include_units=include_units)

    if not include_mode:
        dataframe = dataframe.loc[
            :,
            ~dataframe.columns.str.endswith(".mode"),
        ]

    if not include_uncertainty:
        dataframe = dataframe.loc[
            :,
            ~dataframe.columns.str.endswith(".uncertainty"),
        ]

    dataframe = dataframe.dropna(axis="columns", how="all")

    return _format_report_dataframe(
        dataframe,
        precision=precision,
    )


def _format_report_dataframe(dataframe, *, precision=4):
    """Round numeric columns without mutating the caller's DataFrame."""
    dataframe = dataframe.copy()

    numeric_columns = dataframe.select_dtypes(include="number").columns
    dataframe[numeric_columns] = dataframe[numeric_columns].round(precision)

    return dataframe
