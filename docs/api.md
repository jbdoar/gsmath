# API reference

This page covers every public name exported by `gsmath`. Signatures emphasize
the useful arguments; the [user guide](user-guide.md) explains behavior,
constraints, and complete workflows.

## Loading and core data

### `read(path, *, format=None, **kwargs)`

Load any supported native, spectral, or image format. Usually the file suffix
selects the reader; `format` overrides detection.

### `Sampled`

Shared immutable-style base for sampled data. It supplies unit-aware
arithmetic, value-unit conversion, metadata copying, and cached analyses.

### `Series(x, values, ..., x_unit=None, value_unit=None, channels=None)`

One-dimensional, optionally multi-channel sampled data. Supports channel
selection, trimming, interpolation, arithmetic, peaks, statistics, and plots.

### `Spectrum(wavelength, values, ..., wavelength_unit="nm", value_unit=None)`

A single-channel series whose axis is wavelength. Adds spectral power, color,
normalization, wavelength calibration, shifting, smoothing, and spectral I/O.

### `Image(values, ..., value_unit=None, channels=None, pixel_map=None)`

Two-dimensional single- or multi-channel data. Provides ROI selection, line
cuts, image transforms, color/statistical analysis, uniformity, and plotting.

### `PixelMap(x, y, unit=None)`

Maps each image sample to physical x/y coordinates.

### `LineCut(distance, values, ..., x=None, y=None, channels=None)`

A `Series` sampled along a path through an image; retains source coordinates.

### `Collection(items=(), label="", metadata=None)`

An ordered, labeled group. Apply compatible operations to every item and use
collections for batch summary or reference and one-to-one comparisons.

## Regions and geometry

- `ROI`: base interface for objects that produce image masks.
- `Rectangle(x0, y0, x1, y1, coords=False)`: rectangular ROI.
- `Circle(cx, cy, radius, coords=False)`: circular ROI.
- `Polygon(vertices, coords=False)`: polygonal ROI.
- `Mask(array)`: ROI defined by a Boolean array.
- `Quad(vertices, coords=False, edge_rms=None)`: four-corner display region.
- `find_display(image, *, channel=None, threshold=None)`: detect a bright
  quadrilateral display in an image.
- `homography(source, target)`: calculate a projective transform between two
  quadrilaterals.

With `coords=False`, geometry uses pixel coordinates. Set `coords=True` for
the image's physical coordinate map.

## Measurement and calibration

### Records and execution

- `Measurement(raw, dark=None, exposure=None, ..., calibration={})`: raw data
  plus acquisition context, settings, instrument identity, and a calibration
  reference. `process()` applies a `Calibration`.
- `ExposureSeries(measurements, saturation="auto", ...)`: combine multiple
  exposure times while rejecting saturated elements, then finish calibration.
- `Calibration(name, steps, *, metadata=None)`: ordered, serializable pipeline.
  `process`, `write`, `read`, `reference`, `fingerprint`, and `with_steps`
  support reproducible execution.
- `Step(*, optional=False)`: base class for calibration steps.

### Built-in steps

- `SubtractDark`: subtract `Measurement.dark`.
- `SubtractDarkModel`: subtract a fitted exposure/temperature dark model.
- `MaskSaturated`: replace saturated values with NaN.
- `BlackLevel`: subtract mosaic black levels.
- `Linearize`: map measured detector response to an ideal response.
- `FlatField`: divide by a flat-field image, optionally normalized.
- `Demosaic`: convert a mosaiced image to named color channels.
- `ColorMatrix`: apply a channel-mixing matrix.
- `Despike`: reject isolated spectral spikes.
- `AdaptiveSmooth`: smooth a spectrum according to local noise.
- `PerExposure`: divide values by exposure time.
- `Responsivity`: apply a sampled responsivity calibration.
- `StrayLight`: apply a `StrayLightCorrection`.
- `WavelengthAxis`: replace pixel/sample position with a wavelength solution.

All step constructors accept `optional=True` where shown by their runtime
signature. A skipped optional step marks processing incomplete instead of
raising for missing input.

### Bracketing and detector characterization

- `Bracketed(before, during, after)` and `bracket(...)`: interpolate a
  calibration or dark measurement around an acquisition.
- `DarkModel`: fitted bias, dark-current rate, noise, and optional temperature
  behavior.
- `Linearity`: detector linearity characterization and diagnostics.
- `build_flat(frames, *, dark=None, exposures=None, temperatures=None,
  sigma=None, label="flat")`: construct a flat field from image frames.

## Mosaics and camera colorimetry

- `Mosaic(pattern, black_level=None, white_level=None)`: CFA layout and sensor
  levels attached to raw images.
- `demosaic(image, method="bilinear")`: demosaic immediately outside a pipeline.
- `site_means(image, roi=None)`: mean signal at each CFA site.
- `SpectralSensitivity(wavelength, values, channels=("R", "G", "B"), ...)`:
  multi-channel camera response versus wavelength.
- `fit_color_matrix(measured, reference, *, method="xyz", ...)`: fit a camera
  to reference tristimulus/color values.
- `ColorMatrixFit`: fitted matrix plus training data and diagnostics.

## Spectrum analyses

Call these through `spectrum.peaks()`, `spectrum.power()`,
`spectrum.stats()`, and `spectrum.color()` in ordinary use.

- `PeakAnalysis`: detected peak position, polynomial/Gaussian refinements,
  centroid, prominence, and FWHM.
- `PowerAnalysis`: integrated power and spectral centroid.
- `StatisticsAnalysis`: descriptive value statistics.
- `ColorAnalysis`: XYZ, xy, CIE 1960 uv, CIE 1976 u'v', CIELAB, CIELUV,
  correlated color temperature, Duv, and color differences.
- `xyz_to_lab(XYZ, white_XYZ)`, `xyz_to_luv(XYZ, white_XYZ)`, `xyz_to_xy(XYZ)`,
  `xyz_to_uv(XYZ)`, `xyz_to_uv_prime(XYZ)`, and `xyz_to_cct(XYZ)`: standalone
  conversions for scalar or broadcast-compatible arrays.
- `reference_white_XYZ(white, observer="2deg")`: resolve an illuminant name or
  explicit white point.
- `delta_e_2000(Lab1, Lab2, *, k_L=1, k_C=1, k_H=1)`: CIEDE2000 difference.

## Image analyses

- `ImageStatistics(image, roi=None)`: per-channel descriptive statistics.
- `Histogram`: histogram counts, edges, channels, and unit.
- `ImageColor(image, white="D65", observer="2deg")`: pixel-wise tristimulus
  color conversions and differences.
- `TristimulusColor`: XYZ image values with reference white and unit.
- `edge_mtf(image, roi=None, *, oversampling=4)`: slanted-edge analysis.
- `EdgeMTF`: MTF, ESF, LSF, geometry, and curve results.
- `grid(rows, cols, *, inset=0.1)`: normalized sampling-point grid.
- `Uniformity(image, area=None, points=..., spot=0.04)`: sampled display
  uniformity.
- `FullFieldUniformity(sampled, sigma=None, inset=0, percentiles=None)`:
  interpolate and summarize full-field uniformity.

## Profiles, comparison, and output

- `Field(getter, mode="difference", tolerance=None, uncertainty=None, ...,
  unit=None, minimum=None, maximum=None)`: define one extracted result and its
  acceptance rule.
- `from_analysis(method, fields, *, options=None, relative=())`: build fields
  from named analysis properties.
- `Profile(name="", fields={})`: reusable nested summary/comparison schema;
  `with_limits()` applies configuration-driven tolerances.
- `compare(measured, reference, *, profile)`: compare objects or compatible
  collections using a profile.
- `Comparison`: nested result with `passed`, `failed`, `leaves`, `records`, and
  `summary` views.
- `ComparisonRecord`: one leaf's values, difference, unit, rule, and pass state.
- `to_dict(value, *, flat=False, include_units=False)`: export result objects.
- `to_dataframe(value, *, include_units=False)`: tabular pandas export.
- `print_result(value, *, flat=False, include_units=False)`: console output.
- `report(value, *, precision=4, include_mode=False,
  include_uncertainty=True, include_units=False)`: compact report table.

Field comparison modes are `difference`, `relative`, `percent`, and `ratio`.
Minimum/maximum limits can be used without a separate reference value.

## Reference data

- `ColorMatchingFunction`: wavelength and x-bar/y-bar/z-bar arrays.
- `CMFS`: bundled `"2deg"` and `"10deg"` color-matching functions.
- `Illuminant`: named spectral power distribution.
- `ILLUMINANTS`: bundled standard illuminants.

See the [reference-data notice](../NOTICE.md) before redistributing these data.

## Stray light

- `StrayLightDistribution`: wavelength-indexed distribution matrix; supports
  validation, correction construction, persistence, and regridding.
- `StrayLightCorrection`: correction matrix applied directly or by the
  `StrayLight` step; `distribution()` recovers the underlying distribution.

Both matrix types provide `regrid(new_wavelength, bounds="raise")`. Review the
approximation limits in the user guide before using a regridded correction.

## Wavelength calibration

These APIs are experimental. Validate reference lines, correspondence, and
coverage for the instrument before applying a result.

- `HGAR_LINES_NM`: provisional default Hg/Ar reference line tuple.
- `CalibrationPeak`: detected pixel, height, prominence, and width.
- `LineMatch`: one assigned peak/reference pair and residual.
- `WavelengthCalibration`: polynomial, matches, peaks, domain, and diagnostics;
  `wavelength()` evaluates it and `to_step()` creates `WavelengthAxis`.
- `detect_calibration_peaks(intensity, **options)`: locate candidate peaks.
- `identify_calibration_lines(peaks, reference_lines=..., **options)`: robustly
  assign candidates and fit a constrained wavelength polynomial.
- `calibrate_wavelength(spectrum, reference_lines=..., **options)`: detection
  and identification convenience function.

### Joint line-profile fitting

- `LineSpreadFunction`: asymmetric Gaussian/effective-slit shape parameters.
- `JointLineFit`: calibrations, shapes, components, residuals, masks, errors,
  and warnings from a profile fit.
- `FactoryLineModel`: persisted narrow/working-slit characterization with
  `fit()` for later field scans.
- `fit_lamp_lines(spectrum, *, initial=None, factory=None, ...)`: fit one lamp
  spectrum, optionally using factory priors.
- `fit_factory_lines(narrow, working, *, initial=None, ...)`: jointly fit a
  factory scan pair.

### Held-out validation

- `HeldOutLine`: prediction errors and diagnostics for one held-out feature.
- `WavelengthValidation`: records, failures, summaries, and `assess()` policy.
- `validate_factory_lines(narrow, working, heldout_lines, **options)`: rebuild
  and test peak-only and joint solutions while withholding each requested line.

## Plotting

`SpectrumGraph(*, ax=None, title=None)` wraps a Matplotlib axes for spectrum and
series plots. Most users obtain one from `spectrum.plot()` or `series.plot()`,
then call `show()` or `save(path)`.

## Command-line tools

Installing the package provides:

- `gsmath-straylight`: generate and evaluate stray-light corrections.
- `gsmath-spectral-color-error`: evaluate spectral color error.
- `gsmath-generate-amplitude-cal`: generate amplitude calibration artifacts.

Run any command with `--help` for current arguments. Larger compositional
workflows live in the repository's `examples/` directory.
