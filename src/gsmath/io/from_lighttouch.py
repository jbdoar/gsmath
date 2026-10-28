"""Reader for Gamma Scientific LightTouch spectral exports."""

from pathlib import Path

from .._lighttouch import lighttouch_unit_from_header
from ..spectrum import Spectrum


def from_lighttouch(path):
    """Parse LightTouch metadata and spectral rows into a unit-aware spectrum.

    A ``.cal`` contains output-quantity-per-count factors; other LightTouch
    exports contain the spectral quantity named by their third header line.
    """
    path = Path(path)

    try:
        path.read_text(encoding="utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        encoding = "cp1252"
        
    lines = path.read_text(encoding=encoding).splitlines()

    header_length = int(lines[3])
    data_start = 4 + header_length + 3

    header = lines[:data_start]
    integration_time = None

    for line in header:
        if "[Integration Time ms]" in line:
            integration_time = float(line.split()[0])
            break

    value_unit = lighttouch_unit_from_header(header)
    if value_unit is not None and path.suffix.lower() == ".cal":
        value_unit = f"({value_unit}) / count"
    
    metadata = {
        "header": header,
        "integration_time": integration_time,
    }
    
    return Spectrum.from_csv(
        path,
        encoding=encoding,
        skip_header=data_start,
        metadata=metadata,
        value_unit=value_unit,
    )
