"""Generic two-column CSV loading for :class:`gsmath.Spectrum`."""

from pathlib import Path

import numpy as np

from ..spectrum import Spectrum


def from_csv(
    filename: str | Path,
    *,
    encoding: str | None = None,
    skip_header: int = 0,
    wavelength_column: int = 0,
    values_column: int = 1,
    delimiter: str = ",",
    label: str | None = None,
    **spectrum_kwargs,
) -> Spectrum:
    """Read configurable wavelength/value columns into a spectrum."""
    path = Path(filename)

    if encoding is None:
        try:
            path.read_text(encoding="utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            encoding = "cp1252"

    if skip_header < 0:
        raise ValueError("skip_header must be >= 0")

    if wavelength_column < 0 or values_column < 0:
        raise ValueError("Column indices must be >= 0")

    if label is None:
        label = path.stem

    header = []

    with path.open(encoding=encoding) as file:
        for _ in range(skip_header):
            line = file.readline()
            if not line:
                break
            header.append(line.rstrip("\n"))

    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=skip_header,
        dtype=float,
        encoding=encoding,
    )

    if data.size == 0:
        raise ValueError("CSV contains no numeric data.")

    data = np.atleast_2d(data)

    if data.shape[1] <= max(wavelength_column, values_column):
        raise ValueError(
            f"CSV contains {data.shape[1]} columns, but requested columns "
            f"{wavelength_column} and {values_column}."
        )

    metadata = dict(spectrum_kwargs.pop("metadata", {}) or {})

    if header and "header" not in metadata:
        metadata["header"] = header

    return Spectrum(
        wavelength=data[:, wavelength_column],
        values=data[:, values_column],
        label=label,
        metadata=metadata,
        **spectrum_kwargs,
    )
