# Examples

These are runnable workflows built from the public `gsmath` API. Reusable
command-line tools live in `src/gsmath/scripts/`; case studies and compositions
of library features live here. Small synthetic inputs are checked in so an
example can run from a fresh checkout. Instrument scans and generated results
are passed by path and kept under ignored `artifacts/`.

| Example | Demonstrates | Inputs |
| --- | --- | --- |
| `compare_spectral_collections.py` | `Spectrum`, `Collection`, `Profile`, `Field`, `compare`, `report`, and plotting | A `label,left,right` CSV manifest; bundled synthetic pair below |
| `measurement_pipeline.py` | `Measurement`, ordered `Calibration` steps, units, native persistence, and color | Self-contained synthetic measurement |
| `camera_colorimeter.py` | `DarkModel`, `Linearity`, `build_flat`, `SpectralSensitivity`, `fit_color_matrix`, `uniformity`: a Bayer camera calibrated as an imaging colorimeter, end to end | Self-contained simulated camera |
| `wavelength_calibration.py` | Lamp peak identification and wavelength solution | Lamp scan path |
| `joint_wavelength_calibration.py` | Joint line-profile fitting and factory calibration | Narrow and working-slit scan paths |
| `validate_wavelength_calibration.py` | Held-out line validation | Narrow and working-slit scan paths plus reviewed `--holdout` wavelengths |

Run the synthetic examples from the repository root:

```bash
python examples/measurement_pipeline.py
python examples/camera_colorimeter.py
python examples/compare_spectral_collections.py \
    examples/data/paired_spectra/pairs.csv --format csv
```

Manifest paths resolve relative to the manifest. The two included spectra
are synthetic and deliberately differ slightly in center wavelength and
amplitude. Pass `--output` to redirect the report and plots, or `--no-plots`
when only the CSV report is needed.

Still useful to add: a `bracket()` example and a cleaning/analysis example.
Action spectra need a defined reference-data API before they can have a
meaningful example.
