"""Reader for Gamma Scientific LightTouch spectral exports."""

from pathlib import Path

from gsmath import Spectrum


def from_lighttouch(path):
    """Parse LightTouch metadata and spectral rows into a spectrum."""
    path = Path(path)

    try:
        path.read_text(encoding="utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        encoding = "cp1252"
        
    lines = path.read_text(encoding=encoding).splitlines()

    # A file saved from a spreadsheet ends each header line with a comma.
    header_length = int(lines[3].strip().rstrip(","))
    data_start = 4 + header_length + 3

    header = [line.strip().rstrip(",").rstrip() for line in lines[:data_start]]
    integration_time = None

    for line in header:
        if "[Integration Time ms]" in line:
            integration_time = float(line.split()[0])
            break
    
    metadata = {
        "header": header,
        "integration_time": integration_time,
    }
    
    return Spectrum.from_csv(
        path,
        encoding=encoding,
        skip_header=data_start,
        metadata=metadata,
    )
