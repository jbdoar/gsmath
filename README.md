# gsmath

`gsmath` is a Python toolkit for unit-aware optical measurement. It works with
sampled spectra and images, performs spectral and colorimetric analysis, and
provides reproducible calibration and comparison workflows.

The package is alpha software and currently installs from GitHub:

```bash
python -m pip install "git+https://github.com/jbdoar/gsmath.git"
```

## Quick start

```python
from gsmath import Spectrum

spectrum = Spectrum(
    wavelength=[400, 500, 600, 700],
    values=[0.1, 0.8, 0.6, 0.05],
    wavelength_unit="nm",
    value_unit="uW / cm^2 / sr / nm",
    label="Example LED",
)

processed = (
    spectrum
    .interpolate(step=1)
    .smooth("savgol", window_length=21, polyorder=3)
    .normalize()
)

print(processed.peaks().peak_wavelength_polyfit)
print(processed.power().integral)
print(processed.color(observer="2deg").xy)
processed.plot().show()
```

Transformations return new objects, so the original measurement remains
available. Units are handled with Pint and analysis results expose both numeric
values and units.

## What it covers

- Spectra, multi-channel series, images, regions of interest, and collections
- CSV, LightTouch, image, raw-camera, and native `.gsmath` I/O
- Interpolation, smoothing, cleaning, normalization, arithmetic, and plotting
- Peak, power, photon-flux, color, dominant-wavelength, uniformity, and
  slanted-edge MTF analysis
- Measurement records, calibration pipelines, exposure bracketing, and HDR
- Camera colorimetry, dark/flat/linearity correction, mosaics, and perspective
- Declarative profiles, tolerances, comparisons, reports, and DataFrames
- Stray-light and experimental wavelength-calibration workflows

## Documentation

- [Documentation index](docs/README.md)
- [User guide](docs/user-guide.md)
- [API reference](docs/api.md)
- [Runnable examples](examples/README.md)
- [Reference-data sources and licensing](NOTICE.md)

## Development

```bash
git clone git@github.com:jbdoar/gsmath.git
cd gsmath
python -m pip install -e ".[dev]"
pytest
```

`gsmath` requires Python 3.10 or newer. The source code is MIT licensed;
bundled reference datasets have separate terms recorded in [NOTICE.md](NOTICE.md).

Public exports are loaded on demand. Import the names you need, such as
`from gsmath import Spectrum`, to keep unrelated analysis, plotting, and
export modules out of startup. The first use of a feature loads its
dependencies; subsequent accesses reuse them. `from gsmath import *` is
supported, but loads all names in `__all__`.
