"""Generic wavelength/value CSV output."""

import csv
from pathlib import Path


def to_csv(
    spectrum,
    path,
    *,
    encoding="utf-8",
    delimiter=",",
    header=False,
):
    """Write wavelength and spectral value columns to a generic CSV file.

    CSV does not carry gsmath units or metadata. Set ``header=True`` to add
    column names; when reading that output, pass ``skip_header=1``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding=encoding, newline="") as file:
        writer = csv.writer(file, delimiter=delimiter)
        if header:
            writer.writerow(("wavelength", "values"))
        writer.writerows(zip(spectrum.wavelength, spectrum.values))

    return path
