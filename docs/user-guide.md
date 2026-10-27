# gsmath user guide

`gsmath` is a Python library for loading, transforming, analyzing, comparing,
and plotting sampled optical spectra. Its central `Spectrum` object keeps the
wavelength axis, measured values, physical units, label, and instrument
metadata together as data moves through an analysis.

The project is intended for reusable spectral calculations and command-line
workflows. Experiment-specific analyses can import the library without adding
instrument paths or generated results to the package itself.

## Installation

Until the project is published on PyPI, install it directly from the Git
repository. SSH is usually the simplest option for machines that already have
GitHub access configured:

```bash
python -m pip install "git+ssh://git@github.com/jbdoar/gsmath.git"
```

HTTPS works as well when the repository is public or Git credentials are
configured:

```bash
python -m pip install "git+https://github.com/jbdoar/gsmath.git"
```

To work from a clone instead:

```bash
git clone git@github.com:jbdoar/gsmath.git
cd gsmath
python -m pip install .
```

## Features

- Unit-aware spectral data using Pint, including conversions and arithmetic
  between compatible spectra.
- Generic CSV and Gamma Scientific LightTouch input, generic CSV output, and a
  versioned native `.gsmath` format for faithful round trips.
- Trimming, interpolation, regridding, padding, normalization, wavelength
  calibration and shifting, plus Savitzky-Golay, Gaussian, moving-average, and
  median smoothing.
- Peak detection with FWHM, center wavelength, centroid, polynomial-refined
  peak position, and Gaussian-fit peak position.
- Integrated power, spectral centroid, and common descriptive statistics.
- CIE 1931/1964 tristimulus and chromaticity calculations, CIE 1960/1976 UCS
  coordinates, CIELAB and CIELUV, CIEDE2000 color difference, correlated
  color temperature, and Duv.
- Bundled CIE color-matching functions, standard illuminants, and Robertson
  data used by the colorimetric calculations.
- `Measurement` records raw acquisitions and their context; ordered `Calibration`
  steps reproduce dark, stray-light, exposure, wavelength, and responsivity
  corrections. An `ExposureSeries` combines readings at several exposures into
  one high dynamic range result.
- Declarative summary and comparison profiles with difference, relative,
  percent, and ratio modes; tolerances can produce pass/fail results.
- Labeled collections for applying operations and comparisons across groups of
  spectra.
- Matplotlib plotting and export to nested dictionaries, flattened records,
  pandas DataFrames, and compact reports.
- Stray-light distribution and correction matrices, plus a command-line
  workflow for generating and evaluating corrections.

## Reference data and citations

The package includes CIE colour-matching functions and standard-illuminant
tables, plus the isotemperature-line table used by the Robertson correlated
colour-temperature calculation. These reference datasets are not covered by
the software's MIT license. Their sources, citations, transformations, and
applicable terms are recorded in [NOTICE.md](../NOTICE.md).

## Usage

```python
from gsmath import Spectrum

spectrum = Spectrum(
    wavelength=[360, 500, 650, 830],
    values=[0.1, 0.8, 1.0, 0.2],
    label="Example LED",
    wavelength_unit="nm",
    value_unit="uW / cm^2 / sr / nm",
    metadata={"integration_time_ms": 100},
)

# Operations return new spectra; the original is not modified.
processed = (
    spectrum
    .interpolate(startwl=360, stopwl=830, step=1)
    .smooth(method="savgol", window_length=21, polyorder=3)
    .normalize(by="peak")
)

print(processed.peaks().peak_wavelength_polyfit)
print(processed.power().integral)
print(processed.color(observer="2deg").xy)

color = processed.color(observer="2deg", white="D65")
print(color.Lab)  # L*, a*, b*
print(color.Luv)  # L*, u*, v*

reference_color = spectrum.color(observer="2deg", white="D65")
delta_E_00 = color.delta_e_2000(reference_color)

graph = processed.plot(label="Processed")
graph.show()
```

## Spectrum objects

A `Spectrum` keeps the sampled data and the context needed to interpret it:

- `wavelength`: a one-dimensional NumPy array of sample coordinates, finite and
  strictly increasing (a spectrum checks this when it is made).
- `values`: a NumPy array of measured values on that wavelength grid, one
  per wavelength. NaN marks a missing value.
- `wavelength_unit` and `value_unit`: parsed Pint units, or `None` for an
  intentionally unitless quantity.
- `label`: a human-readable name propagated into derived spectra and plots.
- `metadata`: a copied dictionary for instrument settings, timestamps, or
  other application-specific information.

Transformation and arithmetic methods return a new `Spectrum`; they do not
modify the source object. This makes processing chains explicit and leaves the
raw measurement available:

```python
visible = spectrum.trim(380, 780)
sampled = visible.interpolate(step=1)
smoothed = sampled.smooth("gaussian", sigma=1.5)
normalized = smoothed.normalize(by="peak")

assert spectrum is not visible
```

Use `copy()` when only selected fields need to change:

```python
renamed = spectrum.copy(label="Calibrated LED")
with_context = spectrum.copy(
    metadata={**spectrum.metadata, "operator": "lab-1"},
)
```

Leaving a field out keeps it. For an optional field such as `value_unit`,
passing `None` removes it, so `spectrum.copy(value_unit=None)` is unitless.
This works the same for every kind of data.

### Spectra, series and images

A `Spectrum` is one kind of `Series`: values along a single axis, in one or
more named channels. A plain `Series(x, values, x_unit=..., channels=...)`
holds any other curve, such as a signal against time, temperature or source
level. The other kinds name their axis for what it is:

| Kind | Axis | Channels |
|---|---|---|
| `Spectrum` | `wavelength` | one |
| `SpectralSensitivity` | `wavelength` | a camera's R, G, B |
| `LineCut` (from `image.line`) | `distance` | the image's |
| `Series` | `x` | any |

Whatever its name, `series.axis` and `series.axis_unit` reach the axis.
Every series can do the following:
- select channels (`series["G"]`);
- combine with another series on the same axis (a one-channel series
  scales every channel);
- `trim` and `interpolate`;
- give `peaks()` (position, FWHM, centroid) and `stats()` of one channel;
- `plot()` one line per channel.

`Image` shares the rest with them. That covers unit-aware arithmetic (integer
counts become float first), `to_value_unit`, cached analyses and the `copy`
rule. Characterisation results expose their curves as series too:
`EdgeMTF.curve` and `Linearity.curve`.

A spectrum is also callable as an interpolation function. A scalar wavelength
returns a scalar; an array-like input returns an array, in the order given
(`interpolate` builds a new spectrum, so its grid must increase):

```python
value_at_555_nm = spectrum(555)
values = spectrum([450, 555, 650])
```

Wavelength arguments such as `555`, trim bounds, grid spacing, calibration
coefficients, and shifts are interpreted in the spectrum's current wavelength
unit.

## Units

Units are attached to the axes while the underlying arrays remain ordinary
numeric NumPy arrays. Unit strings use Pint syntax, so both compact and
compound forms are accepted:

```python
from gsmath import Spectrum

radiance = Spectrum(
    wavelength=[400, 500, 600, 700],
    values=[0.1, 0.8, 0.6, 0.05],
    wavelength_unit="nm",
    value_unit="uW / cm^2 / sr / nm",
)

wavelength_um = radiance.to_wavelength_unit("um")
irradiance_mw = radiance.to_value_unit("mW / m^2 / sr / nm")
```

Conversions return new spectra and require dimensionally compatible units.
The library does not silently convert between a unit-bearing value and an
unlabeled (`None`) value.

Spectrum arithmetic follows these rules:

- Two spectra must have identical wavelength arrays and matching wavelength
  units. Interpolate or convert them explicitly first when necessary.
- Addition and subtraction convert the right-hand value unit to the
  left-hand value unit; incompatible dimensions raise an error.
- Multiplication and division combine value units.
- Multiplication and division by ordinary numeric scalars preserve the value
  unit.
- Adding or subtracting a plain scalar is allowed only for unitless spectra.

```python
reference = radiance.copy(
    values=[0.08, 0.75, 0.58, 0.04],
    label="Reference",
)
reference_on_grid = reference.to_wavelength_unit("nm").interpolate(
    xnew=radiance.wavelength
)
difference = radiance - reference_on_grid
scaled = radiance * 0.5
```

Analyses report units according to the quantity being calculated. For
example, widths use the wavelength unit, descriptive statistics use the value
unit, and an integral combines value and wavelength units. Use
`analysis.unit(name)` for a unit, `analysis.quantity(name)` for a Pint
quantity, or `summary(include_units=True)` for a unit-annotated result tree.

## Transforming spectra

The most common transformations are available directly on `Spectrum`:

| Method | Behavior |
| --- | --- |
| `trim(xmin, xmax)` | Keeps samples inside the inclusive wavelength interval. |
| `interpolate(xnew=...)` | Resamples onto an explicit grid. |
| `interpolate(step=..., startwl=..., stopwl=...)` | Constructs and samples a regular grid. Linear interpolation is the default; set `spline_order` for a higher-order spline. |
| `pad("zero")`, `pad("edge")`, `pad("constant", value=...)` | Replaces non-finite leading and trailing values with an explicit boundary policy. |
| `normalize(by=...)` | Scales by peak, integrated power, photometric Y, or the value at a wavelength. |
| `calibrate_wavelength(calibration)` | Applies polynomial coefficients in increasing order, or a callable, to the wavelength axis. |
| `shift_wavelength(shift)` | Adds a constant shift to every wavelength. |
| `smooth(method, ...)` | Applies Savitzky–Golay, Gaussian, moving-average, or median smoothing. |

Interpolation does not extrapolate by default. For a grid extending beyond
the measured range, choose the boundary behavior deliberately:

```python
grid = spectrum.interpolate(xnew=[350, 400, 500, 600, 700, 750])
zero_extended = grid.pad("zero")
edge_extended = grid.pad("edge")

# Alternatively, request spline extrapolation explicitly.
extrapolated = spectrum.interpolate(
    xnew=[350, 400, 500, 600, 700, 750],
    extrapolate=True,
)
```

Normalization accepts several useful forms:

```python
peak_one = spectrum.normalize()                 # maximum becomes 1
peak_100 = spectrum.normalize("peak", value=100)
area_one = spectrum.normalize("integral")
photometric_100 = spectrum.normalize("Y", value=100)
at_555_nm = spectrum.normalize(555, value=1)
```

Smoothing parameters are expressed in samples, not wavelength units. Resample
first when a consistent wavelength spacing matters:

```python
regular = spectrum.interpolate(step=1)
savgol = regular.smooth("savgol", window_length=21, polyorder=3)
gaussian = regular.smooth("gaussian", sigma=2)
boxcar = regular.smooth("moving_average", window=5)
median = regular.smooth("median", window=5)
```

Analyses are lightweight objects cached by each spectrum. Their properties can
be accessed directly, while `summary()` evaluates a consistent nested set of
power, peak, color, and statistical quantities.

Analysis values remain ordinary numeric scalars and arrays. Their units are
derived per quantity from the spectrum and are available separately when
unit-aware output is useful:

```python
from gsmath import report

power = spectrum.power()

print(power.integral)             # numeric magnitude
print(power.unit("integral"))     # value unit × wavelength unit
print(power.quantity("integral")) # Pint Quantity

summary = spectrum.summary(include_units=True)
table = report(spectrum, include_units=True)
```

For a radiometric spectral density, `power().photon_flux` integrates the
photon rate using each wavelength's photon energy. Its unit follows the input:
spectral power produces photons/s, irradiance produces photons/(m^2 s), and
radiance retains its steradian denominator. The `photon` unit converts to
moles, so photosynthetic photon flux density over a selected 400-700 nm band
can be reported directly:

```python
photon_rate = spectrum.power().quantity("photon_flux")
ppfd = (
    spectrum.trim(400, 700)
    .power()
    .quantity("photon_flux")
    .to("umol / m^2 / s")
)
```

Photon flux requires a physical wavelength axis and radiometric rate units;
unitless spectra, detector counts, and energy-only spectra are rejected.

Reports place units in column labels so their cells remain numeric. Wavelength
and width results use the wavelength unit, descriptive statistics use the
value unit, and spectral integrals combine both. Chromaticity coordinates and
Duv are dimensionless, while CCT uses kelvin.

Color analysis defaults to `mode="auto"`. Radiometric spectral-density units
select absolute photometry and produce lumens, candelas, lux, or candelas per
square meter as appropriate. Unitless, relative, and detector-count spectra
produce relative XYZ values without a claimed XYZ unit. The behavior can be
selected explicitly with `spectrum.color(mode="relative")` or
`spectrum.color(mode="absolute")`; absolute mode rejects inputs that are not
dimensionally radiometric. Wavelengths are converted to the nanometer grid of
the bundled CIE observers, and unmeasured portions of that grid contribute
zero.

CIELAB and CIELUV require a reference white. `white="D65"` is the default;
any bundled illuminant name may be used, and its XYZ white point is calculated
for the selected observer and normalized to `Y_n = 100`. An explicit XYZ
triple is also accepted with `white=(X_n, Y_n, Z_n)`. The spectrum's XYZ scale
is preserved when calculating `L*`, so relative spectra should be normalized
to the intended luminance scale rather than relying on implicit normalization.

CIEDE2000 is available as `color.delta_e_2000(other_color)` and as the
standalone `gsmath.delta_e_2000(Lab1, Lab2)` function. Color-analysis operands
must use the same observer and reference white. The standalone function also
supports broadcast-compatible arrays and the application weighting parameters
`k_L`, `k_C`, and `k_H`.

Dominant wavelength is calculated in the selected observer's `xy` diagram
from the selected reference white to the spectral-locus boundary. Purple
stimuli have no dominant wavelength, so they report a complementary wavelength
instead. Achromatic stimuli report neither and have zero excitation purity:

```python
color = spectrum.color(white="D65", observer="2deg")
print(color.dominant_wavelength)       # nm, or NaN for purple/achromatic light
print(color.complementary_wavelength)  # nm for purple light, otherwise NaN
print(color.excitation_purity)         # 0 at white, 1 on the locus boundary
```

## Comparing spectra

Comparisons are driven by a `Profile`: a reusable, declarative description of
which properties to extract and how each property should be compared. A
profile can be used both to summarize a spectrum and to compare it with a
reference, keeping routine reports and acceptance checks consistent.

Each profile contains named `Field` objects. A field's `getter` receives the
source object and returns the scalar or array to record. Its other options
control comparison behavior:

| Option | Meaning |
| --- | --- |
| `mode` | Calculation applied to measured and reference values. Defaults to `"difference"`. |
| `tolerance` | Maximum allowed absolute error. Supplying one makes the field part of the overall pass/fail result. |
| `uncertainty` | Informational uncertainty included in summaries and reports; it does not currently change pass/fail evaluation. |
| `unit` | A unit string, Pint unit, or callable that resolves a unit from the source object. Compatible reference values are converted to the measured unit before comparison. |
| `comparator` | Optional callable for comparison logic that cannot be expressed by a numeric mode. |

The built-in numeric modes are directional: the first operand is measured and
the second is the reference.

| Mode | Result | Example tolerance |
| --- | --- | --- |
| `"difference"` | `measured - reference` | `1.0` means within ±1 in the field's unit. |
| `"relative"` | `(measured - reference) / reference` | `0.02` means within ±2%. |
| `"percent"` | `100 * (measured - reference) / reference` | `2.0` means within ±2%. |
| `"ratio"` | `measured / reference` | `0.02` means the ratio must be within `1 ± 0.02`. |

Fields may be nested in dictionaries to organize the resulting summary and
comparison tree:

```python
from gsmath import Profile, compare, from_analysis, report

profile = Profile(
    name="LED acceptance",
    fields={
        "peaks": from_analysis("peaks", {"wavelength": "peak_wavelength_polyfit"}),
        "photometry": from_analysis("color", ["Y"], options={"observer": "2deg"},
                                    relative=["Y"]),
    },
).with_limits({
    "peaks": {"wavelength": {"tolerance": 1.0, "uncertainty": 0.2}},
    "photometry": {"Y": {"tolerance": 0.02}},
})

summary = processed.summary(profile=profile, include_units=True)
```

`from_analysis` takes results from one analysis by name. A mapping renames
them, and dotted names reach nested results, such as `"planck.cct"`. Each
field gets the unit the analysis declares for it. For a value no analysis
provides, write the field yourself:
`Field(lambda s: ..., unit=...)`.

### Tolerances from a configuration file

Getters belong in code, but tolerances usually belong in a configuration
file that can change without a code change. `Profile.with_limits()` takes a
nested mapping that mirrors a profile's field paths, as loaded from TOML,
JSON or YAML, and returns a new profile. Each leaf may set `tolerance`,
`mode` and `uncertainty`:

```toml
# limits.toml
[peaks.fwhm]
tolerance = 3.0              # nm

[power.integral]
mode = "relative"
tolerance = 0.10             # within ±10%

[color.2deg.x]
tolerance = 0.005
```

```python
import tomllib
from gsmath.profile import SPECTRUM_PROFILE

with open("limits.toml", "rb") as file:
    profile = SPECTRUM_PROFILE.with_limits(tomllib.load(file))
```

The new profile includes only the fields named in the file, so the file
decides what is checked as well as how tightly. An unknown field path, an
unknown setting or an unknown mode raises an error, so a typo cannot
silently drop a check. The source profile is not modified.

### One spectrum against another

Comparing two spectra produces one `Comparison`. The extracted values do not
require the two spectra to share an identical sampled wavelength grid; each
getter evaluates its spectrum independently. Any alignment requirements come
from the particular analysis used by the getter.

```python
reference = Spectrum.read("reference.gsmath")

comparison = compare(processed, reference, profile=profile)
print(comparison.passed)
print(comparison.failed)  # dotted field paths, such as ("peaks.wavelength",)
print(report(comparison, include_units=True))
```

`comparison.passed` is `True` when every field with a tolerance passes,
`False` when any evaluated field fails, and `None` when the profile contains
no tolerances. `comparison.leaves` exposes the individual
`ComparisonRecord`s, `comparison.records` maps each one's dotted field path
(such as `"peaks.fwhm"`) to its record, and `comparison.summary` provides a
nested, export-friendly dictionary.

The method form uses the built-in spectrum profile when no custom profile is
provided:

```python
default_comparison = processed.compare(reference)
custom_comparison = processed.compare(reference, profile=profile)
```

### A collection against one reference spectrum

Use a `Collection` when several measured spectra share one standard or golden
reference. Collection items must have unique labels. The result is another
`Collection`, containing one `Comparison` per measured spectrum in the same
order:

```python
from gsmath import Collection

measured = Collection(
    [
        processed.copy(label="unit-001"),
        processed.shift_wavelength(0.4).copy(label="unit-002"),
        processed.shift_wavelength(-0.3).copy(label="unit-003"),
    ],
    label="production lot",
)

batch_results = compare(measured, reference, profile=profile)

for result in batch_results:
    print(result.label, result.passed, result.failed)

batch_report = report(batch_results, include_units=True)
```

This form is intentionally `compare(measured_collection,
reference_spectrum)`. A single measured spectrum cannot be compared against a
reference collection because relative and ratio comparisons depend on which
operand is the reference.

### One collection against another

Use two collections for paired measurements—for example, before/after runs or
production units with individual reference captures. Items are paired by
label, not position, so collection order may differ. Both collections must
contain exactly the same label set:

```python
references = Collection(
    [
        reference.copy(label="unit-003"),
        reference.copy(label="unit-001"),
        reference.copy(label="unit-002"),
    ],
    label="unit references",
)

paired_results = compare(measured, references, profile=profile)

assert paired_results.labels == measured.labels
print(report(paired_results, include_units=True))
```

A missing or extra label raises an error rather than silently dropping an
item. The comparison results retain the measured collection's order, label,
and metadata.

## Reading and writing spectra

Use the native `.gsmath` format when a spectrum must round-trip with its label,
units, array dtypes, and metadata intact:

```python
spectrum.write("artifacts/measurement.gsmath")
restored = Spectrum.read("artifacts/measurement.gsmath")
```

Generic CSV is available for exchanging wavelength/value columns. CSV does not
store units or metadata, so supply that context when reading if needed:

```python
spectrum.write("artifacts/measurement.csv", header=True)
restored = Spectrum.read(
    "artifacts/measurement.csv",
    skip_header=1,
    wavelength_unit="nm",
    value_unit="uW / cm^2 / sr / nm",
)
```

The format is inferred from `.gsmath`, `.csv`, and `.cal` suffixes or selected
with the `format=` keyword. Gamma Scientific LightTouch spectral exports can
be read with `Spectrum.read(path, format="lighttouch")`; amplitude-calibration
spectra can be written as LightTouch `.cal` files with `Spectrum.write()`.

File provenance records only the input filename, not its absolute directory,
so exported native files do not disclose a user name, network share, or local
folder layout. Other metadata is preserved deliberately. Instrument exports
can contain operator names, serial numbers, timestamps, and free-form headers;
review or remove those fields before sharing a `.gsmath` or metadata-bearing
TIFF file outside their intended audience.

## Development

```bash
git clone https://github.com/jbdoar/gsmath
cd gsmath
python -m pip install -e ".[dev]"
python -m pytest
```

Generated script output defaults to the ignored `artifacts/` directory. Use
each script's output option when output needs to be written elsewhere.

```bash
gsmath-straylight --help
gsmath-spectral-color-error --help
gsmath-generate-amplitude-cal --help
```

`gsmath-straylight calibrate SCANS` builds a stray-light correction from a
directory of monochromator line scans and evaluates it on them;
`gsmath-straylight evaluate SCANS --matrix C.gsmath` evaluates an existing
one. The scans are listed by a `manifest.csv` (`file`, `set_wavelength_nm`)
or named `StrayLight_<nm>.csv`; each is a native measurement or spectrum, a
LightTouch export or a two-column CSV. The science is in
`gsmath.straylight`: `characterize_scan`, `build_distribution`,
`evaluate_correction`, and `stray_light_calibration` for all of it in one
call. `gsmath-spectral-color-error` compares two
LightTouch spectra and writes a chromaticity-error plot. These commands accept
ordinary platform-native paths and create their output directories as needed.
The stray-light generator writes grid-bearing native `D.gsmath` and
`C.gsmath` matrices along with legacy float32 `D.bin` and `C.bin` files.
Native files retain the wavelength grid and array dtype; legacy binaries need
the matching grid supplied when reading:

```python
from gsmath import StrayLightCorrection

native = StrayLightCorrection.read("C.gsmath")
legacy = StrayLightCorrection.read("C.bin", raw_scan.wavelength)
```

`gsmath-generate-amplitude-cal` takes paths to a LightTouch standard, its raw
scan, and a native `.gsmath` or legacy `.bin` stray-light correction matrix,
then writes a native `.gsmath` calibration pipeline and a LightTouch amplitude
`.cal` file. The equation is standard radiance divided by the
stray-light-corrected raw counts;
the LightTouch file stores the resulting response factors on the raw scan's
wavelength grid. For example:

```bash
gsmath-generate-amplitude-cal CAL_STANDARD.std RAW_SCAN.csv SLMAT.bin \
    --output-dir artifacts/amplitude-calibration
```

Reusable command-line tools belong in `gsmath.scripts`; runnable workflows
that illustrate library features belong in [`examples/`](../examples/README.md).
Examples use small bundled data when helpful and accept paths for instrument
datasets. Generated outputs go beneath ignored `artifacts/` by default.

## Images

An `Image` holds pixel values as a `(rows, columns)` array for one channel
(intensity, luminance, a raw sensor mosaic) or `(rows, columns, channels)`
with named channels such as `("R", "G", "B")` or `("X", "Y", "Z")`. Like a
`Spectrum`, it carries a value unit, label and metadata. Operations return new
images, and arithmetic is unit-aware. Integer raw frames are promoted to
float, so subtracting a dark frame can't wrap around.

An optional `PixelMap` gives each pixel centre a coordinate, such as field
angle or position on the object. A separable map has one value per column and
one per row; a full map has an `(x, y)` pair for every pixel (lens
distortion). Row 0 is the top of the image, so a map whose `y` increases
upward has a negative row pitch.

```python
import numpy as np
from gsmath import Circle, FlatField, Image, PixelMap, Rectangle
from gsmath.units import Q_

frame = np.random.default_rng(0).poisson(1000, (1200, 1600, 3)).astype(np.uint16)
angles = PixelMap.linear((1200, 1600), step=(0.05, -0.05), unit="deg")
image = Image(frame, label="display", value_unit="count",
              channels=("R", "G", "B"), pixel_map=angles)

centre = image[500:700, 700:900]       # slices keep the pixel map aligned
green = image["G"]                     # one channel, a 2-D image
spot = Circle(0, 0, Q_(2, "deg"), coords=True)
ring = Circle(0, 0, 10, coords=True) - spot
print(image.stats(roi=spot).mean)      # per channel; NaN pixels ignored
histogram = green.histogram(bins=256, roi=ring)
```

Regions (`Rectangle`, `Circle`, `Polygon`, `Mask`) select pixels by their
centres, either in pixel indices or with `coords=True` in the pixel map's
unit, and combine with `&`, `|`, `-` and `~`. Pixel rectangles are half-open,
like slicing.

Images work in a `Measurement` and through a `Calibration`. `SubtractDark`,
`PerExposure` and `Responsivity` (a number, quantity or per-pixel image) apply
as they do for spectra. `FlatField(flat)` divides by a flat field normalised
per channel to its mean. A single-channel flat or gain image applies to
every channel. A `Measurement` can record the sensor's temperature
(`temperature=Q_(31.5, "degC")`). For channels taken separately, such as
through a filter wheel, it can hold one exposure per channel
(`exposure=Q_([10, 40, 20], "ms")`), and `PerExposure` divides each channel by
its own.

### Transforming images

Transforms return new images, and each pixel keeps its coordinates:

```python
image.flip("horizontal")              # or "vertical"
image.rotate90(k=1)                   # counter-clockwise quarter turns
image.bin(2, method="mean")           # or (rows, cols); "sum"
image.resize((600, 800))              # same field, new pixel count
image.interpolate(regular_angle_map)  # onto another PixelMap's grid
image.smooth("gaussian", sigma=2)     # "box" or "median" with size=
image.normalize("mean", roi=centre, per_channel=True)
image.scale_channels({"R": 1.1, "B": 0.9})

across = image.line((100, 240), (500, 240), width=5)      # a LineCut
spot = image.line((-3, 0), (3, 0), coords=True)            # in map units
image.row(240), image.column(320)
```

Flipping reverses the pixel map, and a quarter turn makes a separable map
per-pixel, because x then varies down the rows. Binning averages the map, and
resampling interpolates it. `interpolate` needs a separable, monotonic source
map; target points outside it become NaN. Gaussian and box smoothing skip
NaN pixels. A raw mosaic can be sliced, flipped and rotated, and its pattern
follows. Smoothing, binning, resampling and line cuts mix neighbouring pixels,
so they ask for a demosaiced image. A `LineCut` is a series of the values
per channel against the distance along the line, and it also keeps each
sample's `x` and `y`. Distances are in the pixel map's unit, or in pixels if
the image has no map. Like any series, a cut gives a spot's width directly:
`image.line((-3, 0), (3, 0), coords=True)["Y"].peaks().fwhm` is in degrees
for an angle map.

### Finding a display and correcting keystone

A camera that isn't square-on to a display sees its active area as a
quadrilateral. `find_display` locates it in a frame of the lit display. A
full-white frame works best, and the corners it finds can be reused for every
other frame from the same setup. `rectify` then resamples an image so the
display fills a rectangle:

```python
display = white_frame.find_display()   # a Quad: corners TL, TR, BR, BL (px)
display.edge_rms                       # ~0.5 px for straight edges; more if curved
flat = frame.rectify(display, size=(Q_(344, "mm"), Q_(215, "mm")))
frame.stats(roi=display)               # a Quad is also a region
```

Detection thresholds luminance (Otsu's method), keeps the largest bright
region, and fits a straight line to each edge for sub-pixel corners. On
synthetic keystoned frames it lands within about 0.15 px. The correction is a
homography: exact for a flat display seen through a lens without distortion.
Curved edges from lens distortion raise `edge_rms`, so correct the distortion
first. The output keeps the resolution the display had in the frame unless
you give a `shape`. Its pixels keep their source coordinates, or `size` gives
a pixel map in display units with y increasing downward. Resampling suits
intensive quantities like luminance and colour. Edge pixels that were only
partly lit in the frame stay partly lit, so inset slightly when measuring at
the very edge.

### Colour of every pixel

An image with channels `("X", "Y", "Z")` has per-pixel colour. Each quantity
is itself an `Image` on the same pixels, with the same pixel map:

```python
color = xyz.color(white=display_white_XYZ)   # or "D65" for Y = 100 data
color.xy, color.uv_prime                     # channels x, y and u', v'
color.Lab, color.Luv                         # L*, a*, b* and L*, u*, v*
color.cct, color.duv                         # CCT in K, and Duv
color.delta_e_2000(reference)                # a CIEDE2000 map
color.cct.stats(roi=spot).mean               # any map has stats, cuts, plots
spot_color = color.mean(roi=spot)            # the region's mean XYZ, as numbers
spot_color.xy, spot_color.cct, spot_color.delta_e_2000(spectrum.color())
```

CIELAB, CIELUV and CIEDE2000 need the reference white on the same scale as
the XYZ. For absolute data in cd/m², pass the white's measured XYZ in those
units. A bundled illuminant name gives a white with Y = 100. A difference
reference can be another image's colour (pixel by pixel), one colour, a
spectrum's `color()`, or a Lab triple, and it must use the same white.

A camera's R, G, B become X, Y, Z through a colour matrix. Fit one from
patches measured with both the camera and a spectroradiometer, then apply it
directly or as a calibration step:

```python
from gsmath import ColorMatrix, fit_color_matrix

camera_rgb = [rgb.stats(roi=patch).mean for patch in patches]   # (n, 3)
reference = [spectrum.color().XYZ for spectrum in measured]     # (n, 3)
matrix = fit_color_matrix(camera_rgb, reference)
xyz = rgb.apply_matrix(matrix, unit="cd/m^2 / (count/ms)")

camera = Calibration("colorimeter", [
    SubtractDark(), FlatField(flat), Demosaic(), PerExposure(),
    ColorMatrix(matrix, unit="cd/m^2 / (count/ms)"),
])
```

`image.plot()` draws an image with axes in pixel-map units. Row 0 is at the
top, and a map whose y increases upward reads upward. Magnitudes use a
single-hue sequential colormap. Data with both signs (Duv, a*, a difference)
uses a blue-gray-red diverging map with gray at zero. R, G, B images are
drawn as colour (`srgb=True` encodes linear data for display). Other
multi-channel images take `channel=`.

### Calibrating a camera as a colorimeter

A camera's channel signals are its spectral sensitivities integrated against
the light. X, Y, Z are the colour-matching functions integrated against the
same light. Unless the sensitivities are an exact linear mix of the
colour-matching functions (the Luther condition), no single matrix maps one
to the other for every spectrum. So the matrix is fitted on the kind of light
the camera will measure, and checked on other samples of it.

```python
from gsmath import SpectralSensitivity, fit_color_matrix

# From a monochromator scan: dark-subtracted camera signal per channel at
# each wavelength, over the band's radiance from a calibrated detector.
sensitivity = SpectralSensitivity.from_monochromator(
    wavelength, signal, band_radiance,
    signal_unit="count/ms", reference_unit="W/sr/m^2")

# Any known spectrum then gives its camera signal without imaging it:
# a calibrated tunable source's settings, a display's primaries, ...
fit = fit_color_matrix(sensitivity.responses(training),
                       [s.color().XYZ for s in training],
                       unit="cd/m^2 / (count/ms)")
fit.validate(sensitivity.responses(held_out), [s.color().XYZ for s in held_out])
camera = Calibration("colorimeter", [SubtractDark(), FlatField(flat), Demosaic(),
                                     PerExposure(), fit.step()])
```

Instead of computing camera signals from sensitivities, you can fit on
measured pairs: each patch's camera signal (`site_means(raw, roi)` straight
from the raw mosaic), against the X, Y, Z a spectroradiometer measures.
`fit_color_matrix` methods:
- `"xyz"`: least squares, weighting each sample by 1 / (X + Y + Z) so dim
  and bright samples count alike. It can preserve a white exactly
  (`preserve=`).
- `"uv_Y"`: minimises Δu′v′ and relative ΔY, each scaled by its tolerance.
- `"delta_e2000"`: minimises CIEDE2000 against a white.

`fit.validate` reports the mean, RMS and maximum Δu′v′, Δxy, ΔY and ΔE₀₀.

What the matrix is fitted on matters more than how. On a simulated
consumer-style Bayer camera:

| Light measured | Matrix fitted on | Mean Δu′v′ | Mean ΔY |
| --- | --- | --- | --- |
| A display's colours | that display's R, G, B and white | exact | exact |
| Mixes of a 12-channel LED source | mixes of its channels | ~0.006–0.009 | ~3–4% |
| Single narrow-band LEDs, 410–690 nm | one matrix for all | ~0.08–0.10 | ~90% |

A display's colours are all mixes of its three primaries, so a matrix fitted
on that display's own primaries is exact for it. Measuring arbitrary
narrow-band light needs more channels, for example a filter-wheel camera.

### Characterising the camera: dark, linearity, flat field

These are measured once per camera and setting (gain, binning,
temperature), then applied by calibration steps:

```python
from gsmath import DarkModel, Linearity, MaskSaturated, build_flat

dark = DarkModel.fit(dark_runs)                # Measurements: bias + rate x exposure
dark.read_noise, dark.hot_pixels()             # from repeated frames; a Mask
dark.doubling, dark.plot()                     # darks at several sensor temperatures
linearity = Linearity.from_frames(series, exposures, dark=dark)
linearity.gain, linearity.max_deviation, linearity.saturation
flat = build_flat(sphere_frames, dark=dark, sigma=4)

front = [MaskSaturated(), dark.step(), linearity.step(), FlatField(flat)]
camera = Calibration("colorimeter", [*front, Demosaic(), PerExposure(), fit.step()])
```

- **`DarkModel.step()`** subtracts the modelled dark for each measurement's
  exposure, so you don't need a dark frame at every exposure.
- **Uncooled cameras:** record the sensor temperature with each
  measurement (`Measurement(..., temperature=Q_(31.5, "degC"))`) and take
  darks at several exposures at each of several temperatures (as the camera
  warms up, say). `DarkModel.fit` then also fits how the dark changes with
  temperature: per pixel, the bias drifts linearly and the dark rate grows
  as `2 ** ((T - T0) / doubling)`, with one doubling temperature (typically
  5–10 K) for the whole sensor. Its step then needs each measurement's
  temperature, and warns outside the temperature range it was fitted over.
  A model fitted at one temperature records it, and warns when used more
  than 1 K away. `build_flat` and `Linearity.from_frames` take
  `temperatures` (or the Measurements' own) for the model's dark.
- **`MaskSaturated`** marks pixels at 98% of the raw file's white level as
  NaN.
- **`Linearity.from_frames`** marks a point saturated as soon as any pixel
  in the region reaches the white level. Vignetting clips the centre while
  the mean is still rising. `step()` gives a `Linearize` step that
  maps each channel, or each colour's sites, back to proportional.
- **`build_flat`** smooths within each colour's own sites of a raw mosaic.
  On a mosaic, `FlatField` normalises each colour separately, so the flat
  source's colour doesn't shift the data's colour balance.
- **Measure the spectral sensitivity through the same front steps,** so the
  matrix is fitted on the signal it will be applied to.

`examples/camera_colorimeter.py` runs the whole procedure on a simulated
12-bit Bayer camera with bias, dark current, hot pixels, read and shot
noise, vignetting, slight nonlinearity and clipping. The camera is uncooled,
so bias and dark current drift as it warms. It then measures a non-uniform
display. The model recovers the 6.5 K doubling temperature. On a 250 ms dark
at 40 °C, it leaves a median of about 0.0 counts, where a model fitted at
24 °C leaves about 28. At 20 ms and 8 averaged frames, per-pixel luminance is
within about 0.15% on average (0.7% RMS, mostly noise), Δu′v′ averages
about 0.0013, and 9-point uniformity matches the truth to about 0.001.

### Display uniformity

`uniformity` measures luminance (Y) and chromaticity (x, y and u′v′) over
sample spots in a display's active area. The area can be the whole image, a
pixel `Rectangle`, or a `Quad` from `find_display`. Positions are fractions
of the active area, mapped through its homography, so a keystoned frame is
measured as it is, with no resampling, and each spot is a circle on the
display:

```python
from gsmath import grid

display = frame.find_display()
result = xyz.uniformity(display, points=grid(3, 3, inset=0.1), spot=0.04)
result.luminance        # min, max, mean, centre, min_max_ratio, min_mean_ratio,
                        # nonuniformity, max_deviation_from_mean/centre, cv
result.color            # x_range, y_range, max_delta_xy, max_delta_uv_prime,
                        # and both from the centre
result.points_frame     # one row per spot: u, v, pixel position, X, Y, Z, x, y, u', v'
result.plot()           # luminance with the spots, each labelled Y / centre

field = result.full_field(sigma=2, inset=0.02, percentiles=(1, 99))
field.luminance, field.color
field.relative_luminance.plot(), field.delta_uv_prime.plot()
```

Each spot's colour comes from its mean X, Y, Z, not the mean of x and y. The
centre reference is the spot nearest the middle of the area. `spot` is the
spot's diameter as a fraction of the area's width. In `full_field`, `sigma`
smooths the frame first so noise doesn't set the extremes, and
`percentiles` replaces the minimum and maximum. No standard's layout or
formula is built in: pick the points and spot size your standard or
specification calls for. A luminance-only image gets the luminance metrics.

#### Pass or fail against a specification

A specification is a mapping shaped like the result's summary. Each limited
number gets a `minimum`, a `maximum`, or both, so it can live in a TOML,
JSON or YAML file:

```python
spec = {"luminance": {"min_max_ratio": {"minimum": 0.8},
                      "mean": {"minimum": 250, "maximum": 350}},
        "color": {"max_delta_uv_prime": {"maximum": 0.01}}}
check = result.check(spec)
check.passed, check.failed          # False, ("color.max_delta_uv_prime",)
check.records["luminance.mean"]     # value, limits, passed
to_dataframe(check, include_units=True)
```

A NaN fails its limit. A number listed without limits is reported but not
judged. A misspelt name raises an error rather than skipping the check.

### Results: summaries, units, limits

Every result works the same way, whether it comes from a spectrum
(`peaks()`, `color()`, ...) or an image. That includes `stats()`,
`histogram()`, `color()`, `uniformity()`, `edge_mtf()`, and characterisation
results such as `DarkModel`, `Linearity` and `ColorMatrixFit`:

- `summary` is a nested mapping of its numbers;
- `unit("luminance.mean")` gives a number's unit, and
  `quantity("luminance.mean")` the number with it;
- `to_dict(include_units=True)` and `to_dataframe(include_units=True)`
  export it;
- `check(spec)` judges it against limits, and `compare(other, limits)`
  compares it with another result of its kind, using `tolerance` and `mode`
  per number as for spectra.

Images summarise, compare and check like spectra, by every channel's
statistics:

```python
image.summary()["Y"]["mean"]
image.compare(reference_image).records["Y.mean"].difference
image.check({"Y": {"mean": {"minimum": 100}, "std": {"maximum": 2}}})
```

`spectrum.check({"peaks": {"fwhm": {"maximum": 25}}})` applies limits to a
spectrum's profile the same way. Other series summarise each channel's peak,
FWHM, centroid and statistics, with positions in the axis unit:
`cut.check({"Y": {"fwhm": {"maximum": 0.5}}})` judges a spot's width in degrees.

A `Collection` can hold any of these, as long as each item has a unique
label: spectra, images, line cuts or results. It applies the same
operations item by item, using its first item's default profile:

```python
batch = Collection(uniformity_results)        # or spectra, images, line cuts
batch.to_dataframe(include_units=True)        # one row per item
checks = batch.check(spec)                    # a Collection of Checks
checks.passed, checks.failed                  # False, ("panel 7",)
checks["panel 7"].failed                      # ("color.max_delta_uv_prime",)
batch.compare(reference, {"luminance": {"mean": {"tolerance": 0.02,
                                                 "mode": "relative"}}})
```

`compare` takes one reference for every item, or another collection, paired
by label. Keep a collection to one kind of item, and for images or series to
the same channels.

### Sharpness: slanted-edge MTF

`edge_mtf` measures the modulation transfer function across a straight edge
tilted a few degrees from vertical or horizontal (2–10° is typical), using
the ISO 12233 e-SFR method:

```python
edge = image.edge_mtf(roi=Rectangle(400, 300, 480, 400))
edge.mtf50, edge.mtf10                  # cycles/px, per channel
edge.frequency, edge.mtf                # the curve
edge.frequency_in("1/deg")              # with a pixel map: cycles per degree
edge.angle, edge.offset                 # edge tilt; per-channel edge shift
edge.plot(per="1/deg")
```

Each row's edge position is the centroid of its derivative across the edge.
A line through those positions gives the angle, and every pixel is binned by
its distance from that line at a quarter of a pixel, giving an oversampled
edge spread function. Differentiating it gives the line spread function,
which is windowed and Fourier transformed. The result is corrected for the
derivative filter's response. Only whole phase cycles of rows are used.
Against synthetic edges with a known Gaussian blur, integrated over each
pixel, the curve agrees with the analytic MTF to about 0.01 up to
0.75 cycles/px, and MTF50 to about 1%.

Each channel is analysed separately, so `offset` also shows lateral
chromatic aberration. The data should be linear in light, so linearise
encoded (gamma) images first. With a pixel map, `frequency_in` converts
using the local pixel pitch across the edge. R, G and B curves are drawn in
their own colours and named directly at MTF50.

### Image files and raw sensor data

Reading and writing image files needs the optional dependencies:

```bash
python -m pip install "gsmath[images]"   # imagecodecs, tifffile, rawpy
```

| Format | Read | Write | Notes |
| --- | --- | --- | --- |
| `.gsmath` | yes | yes | Lossless, including pixel maps and metadata. |
| TIFF | yes | yes | Any dtype and channel count, exact. gsmath's fields round-trip, plus a separable pixel map. |
| PNG | yes | yes | 8 or 16 bit, exact (16-bit RGB included). |
| JPEG | yes | yes | 8 bit, lossy. |
| Camera raw (DNG, CR2/CR3, NEF, ARW, RAF, ...) | yes | no | The sensor mosaic, untouched. |

PNG and JPEG hold encoded display values, so images read from them have no
value unit unless you pass `value_unit=`. Writing them requires `uint8` (or
`uint16` PNG) data rather than guessing a scaling. Native `.gsmath` records
that hold images are zip containers: the JSON record plus each array as an
`.npy` file. So is any record holding an array over 1 MiB, such as a large
stray-light matrix or a calibration that includes one. Small records,
including spectra, stay plain JSON you can read. `gsmath.read` opens any of
these, including a saved `FactoryLineModel`.

A raw file reads as a single-channel image in counts. `image.mosaic` (a
`Mosaic`) holds the colour-filter pattern (`"RGGB"`, ...), the black level
per site and the white level. `metadata["camera"]` holds white balance, the
camera's RGB-XYZ matrix and shot settings, including `exposure_s`. Nothing
is applied implicitly. Slicing, flipping and rotating keep the pattern
aligned, and the mosaic survives per-pixel steps, so demosaicing is a step in
the calibration. To build a raw image yourself, pass the pattern:
`Image(counts, value_unit="count", mosaic="RGGB")`.

```python
import gsmath
from gsmath import Calibration, Demosaic, FlatField, Measurement, PerExposure, SubtractDark
from gsmath.units import Q_

raw = gsmath.read("shot.dng")
dark = gsmath.read("dark.dng")
camera = Calibration("camera", [
    SubtractDark(),            # or BlackLevel() when there is no dark frame
    FlatField(raw_flat),       # a raw flat field: same mosaic
    Demosaic("bilinear"),      # or "superpixel": half resolution, no interpolation
    PerExposure(),
])
exposure = Q_(raw.metadata["camera"]["exposure_s"], "s")
rgb = Measurement(raw=raw, dark=dark, exposure=exposure).process(camera)
```

Both demosaicing methods are linear in the data, so they leave a radiometric
calibration intact. Images, image measurements and calibrations holding
images save to native `.gsmath` files and load back with `gsmath.read`.

## Measurements and calibration

`Measurement` records an instrument acquisition so it can be processed again
later. It holds raw data, an optional dark measurement, exposure time, a label,
acquisition settings, instrument details, metadata, and an optional reference
to the calibration used at capture time. The raw spectrum stays in detector
units, such as counts; processing produces a new calibrated spectrum.

The measurement is the record of **what was acquired**; it does not apply
corrections when it is created. `raw` is the required data object, and `dark`
must be the same kind of object when present. `exposure` is a unit-aware time
quantity when a pipeline needs it. Keep acquisition-specific values such as
averaging, gain, instrument identity, and file provenance in `settings`,
`instrument`, and `metadata`. These dictionaries are preserved when the
measurement is saved. The optional `calibration` field is a reference to the
calibration in force at capture time, not the calibration payload itself; use
`Calibration.reference` to record its name and content fingerprint.

```python
from gsmath import Measurement, Spectrum
from gsmath.units import Q_

raw = Spectrum.from_lighttouch("raw_scan.csv").copy(value_unit="count")
measurement = Measurement(
    raw=raw,
    exposure=Q_(100, "ms"),
    label="green left",
    settings={"averages": 3, "gain": "high"},
    instrument={"model": "example-spectrometer", "serial_number": "example"},
    metadata={"source_file": "raw_scan.csv"},
)
measurement.write("runs/green-left.gsmath")
measurement = Measurement.read("runs/green-left.gsmath")
```

A `Calibration` names an ordered sequence of processing steps. Each step
describes one operation in the equation that maps a measurement's raw data to
derived data. A calibration is reusable across measurements that share the
same instrument setup and input assumptions. For spectra, common steps
include:

- `SubtractDark()` subtracts the measurement's dark spectrum.
- `StrayLight(correction)` applies a `StrayLightCorrection` on its wavelength
grid.
- `PerExposure()` divides by the recorded exposure time.
- `WavelengthAxis(solution)` assigns calibrated wavelengths to detector
samples without resampling their values.
- `Responsivity(response)` converts exposure-normalized counts to radiometric
spectral density.

The order matters. Subtract the dark before correcting stray light; apply the
stray-light matrix on the grid for which it was made; apply a wavelength
solution before using wavelength-dependent responsivity or calculating color.
Required inputs are checked when processing: a required dark or exposure that
is absent raises an error. Mark a step `optional=True` only when it is valid to
skip that operation for some measurements. Skipped steps are listed in the
processing report and make its `complete` flag false. `until="StepName"`
processes only through the named step; that result is also marked incomplete.

For example:

```python
from gsmath import (
    Calibration,
    Measurement,
    PerExposure,
    Responsivity,
    Spectrum,
    StrayLight,
    SubtractDark,
    WavelengthAxis,
)
from gsmath.units import Q_

raw = Spectrum.from_lighttouch("raw_scan.csv").copy(value_unit="count")
dark = Spectrum.from_lighttouch("dark_scan.csv").copy(value_unit="count")
response = Spectrum(
    wavelength=calibrated_wavelength_nm,
    values=radiance_per_count_per_ms,
    value_unit="uW / cm^2 / sr / nm * ms / count",
    label="Instrument responsivity",
)
calibration = Calibration("spectral radiance", [
    SubtractDark(),
    StrayLight(stray_light_correction),
    PerExposure(),
    WavelengthAxis(wavelength_solution),
    Responsivity(response),
])
measurement = Measurement(
    raw=raw, dark=dark, exposure=Q_(100, "ms"), label="green left",
    settings={"averages": 3},
    instrument={"model": "example-spectrometer", "serial_number": "example"},
    calibration=calibration.reference,
)
radiance = measurement.process(calibration)
print(radiance.color(observer="2deg").summary)
print(radiance.metadata["processing"])
```

Use `SubtractDark(optional=True)` only when scans may genuinely lack dark data;
when it is skipped, the processing report marks the result incomplete. If the
instrument has already subtracted the background, do not subtract it again.
`PerExposure` requires `Measurement.exposure`, supplied as a time quantity.
The responsivity spectrum must use the same calibrated wavelength grid as the
measurement after the wavelength step.

Save and reload the pipeline to reproduce the processing later:

```python
calibration.write("calibrations/spectral-radiance.gsmath")
loaded = Calibration.read("calibrations/spectral-radiance.gsmath")
reprocessed = measurement.process(loaded)
```

`Calibration.with_steps([...], after="PerExposure")` creates a new pipeline
with steps inserted at a named point; it does not mutate the original. Each
calibration has a fingerprint derived from its name, ordered steps, and the
data that determines each step's effect. The native calibration file stores
the steps and their data, including a stray-light matrix or wavelength
solution. Save the calibration once and measurements separately; the
measurement's calibration reference lets a run identify which saved pipeline
was used without duplicating that payload. Each processed spectrum records the
calibration name and fingerprint, applied and skipped steps, and whether the
pipeline completed.

### High dynamic range: an exposure series

An `ExposureSeries` holds one `Measurement` per exposure time, each with its
own dark, and combines them as it is processed. It works the same for a
spectrometer's spectra and a camera's images: an element of either clips the
same way.

Each measurement goes through the calibration up to `combine_after`. By
default that is its dark subtraction, `SubtractDark` or a dark model's
`SubtractDarkModel`. Each element's rate is then the exposure-weighted mean
over the exposures where it is usable: its raw value is below `saturation`
and its processed value is finite. That works out to its summed counts over
its summed exposure. A pixel that an earlier `MaskSaturated` step made NaN is
left out, not spread. The remaining steps run once, on the combined data,
expressed as counts at the longest exposure:

```python
from gsmath import ExposureSeries

series = ExposureSeries([short, medium, long], saturation=65535)
radiance = series.process(calibration)
report = radiance.metadata["processing"]["reports"]["ExposureSeries"]
report["elements_by_exposures_used"]   # {"0": 2, "1": 40, "2": ..., "3": ...}
report["saturated_in_all"]             # elements usable in no exposure
series.exposures_used()                # the map, shaped like the data

frames = ExposureSeries(raw_measurements)   # saturation="auto": 98% of the raw white level
xyz = frames.process(camera)                # combines after the dark model's step
```

An element usable in no exposure takes the shortest exposure's value, which
is NaN if it was masked. `saturation="auto"`, the default, uses the white
level a raw image records. Spectra record none, so give their `saturation`
explicitly. The report stays small whatever the data's size.
`exposures_used()` gives the full map as a Spectrum or an Image.

The exposures are combined before any step that mixes pixels, such as
`StrayLight`, so a saturated pixel's error never spreads into its neighbours.
A series saves with `series.write(path)` and loads with
`ExposureSeries.read(path)` or `gsmath.read(path)`, so it can be combined and
calibrated again later. `Calibration.process(measurement, after="Step")`
resumes a calibration after a named step, for data that has already been
through the steps before it.

## Experimental wavelength calibration

`calibrate_wavelength` detects lamp peaks, refines their centers to subpixel
positions, and uses RANSAC pattern matching to fit sample index to wavelength.
Peak heights are not used to assign line identities. The input scan is unchanged.

```python
from gsmath import Spectrum, calibrate_wavelength

scan = Spectrum.from_lighttouch("lamp.csv")
result = calibrate_wavelength(
    scan,
    degree=4,  # default: fourth-order wavelength polynomial
    tolerance_nm=1.0,
    direction="increasing",
    wavelength_bounds_nm=(250, 1250),  # optional instrument-specific bounds
    dispersion_bounds=(0.7, 1.2),     # optional absolute nm/sample bounds
)
print(result.coefficients)  # descending powers, suitable for numpy.polyval
print(result.rms_nm, result.max_error_nm, result.matched_pixel_fraction)
for match in result.matches:
    print(match.pixel, match.wavelength_nm, match.residual_nm)
```

Coordinates are zero-based indices in the supplied array, including for a
`Spectrum`; its existing wavelength axis is not used as a prior. Trimming or
resampling changes these coordinates. `result.wavelength(pixels)` evaluates the
fit. Detection and matching can also be used independently through
`detect_calibration_peaks` and `identify_calibration_lines`.

Apply the identified solution through the existing Spectrum API, or save it as
part of a measurement-processing calibration:

```python
from gsmath import Calibration, Measurement, SubtractDark

corrected = scan.calibrate_wavelength(result)

pipeline = Calibration("HgAr wavelength", [
    SubtractDark(optional=True),
    result.to_step(),  # WavelengthAxis(result)
])
pipeline.write("wavelength.gsmath")
loaded = Calibration.read("wavelength.gsmath")
corrected = Measurement(raw=scan).process(loaded)
```

This replaces the axis with wavelengths in nm without interpolating signal
values. Decreasing detector dispersion reverses both axis and values to produce
an increasing wavelength grid. Application requires the same full sample count
and zero-based pixel domain as the lamp scan; keep binning and detector layout
consistent too. Apply before trimming/resampling, after dark subtraction and
corrections tied to the old grid. Any subsequent spectral responsivity must use
the calibrated grid. Saved pipelines retain the fitted solution and matched-line
diagnostics, which also appear in the processing report. Pass the result object
to `Spectrum.calibrate_wavelength`, not its coefficients: legacy coefficient
arrays use ascending powers and operate on the existing wavelength axis.

The default `HGAR_LINES_NM` list is **provisional**, with coverage from 365 to
965.7786 nm. Its Hg and Ar air wavelengths are drawn from the NIST strong-line
tables for [mercury](https://physics.nist.gov/PhysRefData/Handbook/Tables/mercurytable2.htm)
and [argon](https://physics.nist.gov/PhysRefData/Handbook/Tables/argontable2.htm).
Pass `reference_lines=` with
your validated wavelengths in nm and a consistent air/vacuum convention. Close
doublets can be unresolved and bias fitted centers; the peak-only RANSAC stage
does not model blends. The joint profile fitter below does. Inspect the correspondences and detector coverage before using
a result. RMS measures agreement with assigned lines, not absolute accuracy or
identification confidence. Sparse or ambiguous patterns can produce incorrect
solutions; instrument bounds help, and more representative scans are needed.
The search is reproducible by default (`random_seed=1`), supports polynomial
degrees 1–4 (default 4), and raises if it cannot find the required consensus.
`ransac_degree` defaults to 2 (or the requested degree if lower); the search mixes
affine seeds with that order and refines consensus before the final fit. The
final fit always uses `degree`, default **4**, and raises if it fails the
constraints rather than silently returning a lower-order calibration. It does not
quantify competing-solution confidence. Increasing `iterations` may help a
missed solution but does not resolve physical ambiguity.

Run `python examples/wavelength_calibration.py lamp.csv` to produce a diagnostic
PNG and a CSV of matched lines under `artifacts/wlcal/`. Use `--wavelength-bounds`,
`--dispersion-bounds`, and `--direction` for instrument priors. Bounds in the
example above suit the supplied development scan and are not universal defaults.

## Joint lamp profiles and factory slit characterization

`fit_factory_lines` simultaneously fits a factory narrow/working-slit pair.
The scans must use the same lamp, probe, optical setup and detector sampling;
slit and exposure may differ. Both wavelength solutions default to fourth order.
The model integrates each line over detector pixels and convolves a
mean-centered asymmetric Gaussian with an effective rectangular slit. Width
varies along the detector. Nearby lines are added before comparison with the
observed signal, so unresolved maxima are not treated as individual line centers.
Line areas are shared across the factory pair, with separate gains and linear
backgrounds per region. This assumes relative line intensities within a region
are stable during the factory pair; inspect residuals when that assumption fails.

```python
from gsmath import (Spectrum, calibrate_wavelength, fit_factory_lines,
                    FactoryLineModel, Calibration)

narrow = Spectrum.from_lighttouch("factory_narrow.csv")
working = Spectrum.from_lighttouch("factory_working.csv")
initial = calibrate_wavelength(narrow, degree=4, ransac_degree=2)
factory, fit = fit_factory_lines(
    narrow, working, initial=initial,
    metadata={"instrument": "serial-number", "slit": "working setting"},
)
print(fit.normalized_rms, fit.warnings)
factory.write("factory_lines.gsmath")

# Later, in the field: a working-slit lamp scan is sufficient.
factory = FactoryLineModel.read("factory_lines.gsmath")
field_scan = Spectrum.from_lighttouch("field_lamp.csv")
field_fit = factory.fit(field_scan)
solution = field_fit.calibrations[0]
pipeline = Calibration("field wavelength", [solution.to_step()])
pipeline.write("field_wavelength.gsmath")
```

Field fitting starts from the stored working-slit calibration and constrains
width/asymmetry and relative line areas softly; it permits different gains and
backgrounds. `shape_prior_sigma` and `area_prior_sigma` control those constraints.
It does not require another narrow scan or assume fixed absolute intensities.
Keep instrument configuration consistent; the library can check sample count,
but cannot infer a change in probe, binning, grating, or physical slit.

For a single scan without factory characterization, use
`fit_lamp_lines(scan, initial=reviewed_solution, reference_lines=validated_lines)`.
If `initial` is omitted, RANSAC provides it. A joint fit is local and cannot
repair a grossly incorrect initial identification. `max_shift_nm=2` bounds the
total correction to the initial wavelength curve across the detector. Supply
`working_initial` for a known larger factory slit shift. `saturation=...` excludes
samples at or above the specified signal level; use unsaturated scans where possible.

Inspect `models`, `components`, `backgrounds`, `residuals`, `masks`, and `warnings`
on the returned fit. Each component array has shape (samples, reference lines),
in original signal units; components plus background sum to the modeled signal.
Unmodeled pixels are NaN. `normalized_rms` is a signal-profile residual divided
by each region's signal range, **not a wavelength uncertainty**. The calibration's
ordinary peak residuals still refer to measured maxima, which can be displaced
by blends. Near-coincident lines can have indeterminate individual amplitudes.
The current model is experimental: shape mismatch, missing reference lines,
poor wavelength coverage, and parameters at bounds require review. Optimizer
convergence alone is not acceptance, and no absolute uncertainty is estimated.

`python examples/joint_wavelength_calibration.py NARROW.csv WORKING.csv` saves
a factory model, both calibration pipelines, signal-fit arrays, a JSON report,
and a diagnostic PNG under `artifacts/wlcal/`. The example saves candidates for
review; it does not update an instrument.

## Held-out wavelength validation

`validate_factory_lines` compares peak-only and joint fourth-order calibrations
by withholding one isolated-line region at a time from **both** scans. It
removes neighboring references inside that region, masks the corresponding
pixels even in overlapping fit windows, and rebuilds the initial polynomials
using only the retained peak correspondences. Factory line shapes and area
priors are fitted afresh for every fold.

```python
from gsmath import validate_factory_lines

validation = validate_factory_lines(
    narrow, working,
    heldout_lines=[435.833, 546.074, 696.543, 763.511, 826.452, 852.144, 912.297],
)
print(validation.summary())
print(validation.summary(interpolation_only=True))
print(validation.failures)
```

Provide `narrow_initial` and `working_initial` to reuse reviewed peak identities;
otherwise RANSAC supplies them. This is **conditional validation of the fit**,
not an independent test of blind line identification. The supplied initial
coefficients are not reused as fold starting curves. Failed folds remain in the
report, including lines too close to neighbors to score as isolated features.
Predictions outside the retained anchor range are marked as extrapolation and
reported separately.

Each method is scored in its own coordinate convention: peak-only errors use
the measured parabolic maximum; joint errors use a local translation of the
line profile learned from the training lines. The held-out fit adjusts only
position, area and background, keeping the wavelength polynomial and line
shape fixed. Its `joint_profile_rms` reports whether that learned shape also
describes the test feature. These errors do not establish traceable absolute
accuracy or resolve uncertain reference identities.

The default policy is **at most 0.5 nm absolute error at every resolvable peak**
over the instrument's full wavelength range. Call `validation.assess()`;
override `max_error_nm` for a slit-specific specification. RMS is descriptive
unless an additional `max_rms_nm` limit is explicitly provided. A low RMS never
compensates for an individual peak outside tolerance.

Use `working_range_nm=(lower_nm, upper_nm)` to explicitly restrict application
requirements, for example to a visible-only range. Endpoints are inclusive;
out-of-range errors/failures do not affect that restricted assessment.
Extrapolated lines within the selected range remain subject to the tolerance.

Pass `required_lines_nm` with a reviewed, complete inventory of resolvable peaks
to establish coverage. It may be one wavelength list or a dictionary with
separate `narrow` and `working` lists, since resolution depends on slit setting.
Do not count individually inferred components of an unresolved blend as
independently resolved peaks. Unidentified features require review before
declaring the inventory complete. The provisional reference list and a small
selection of held-out lines do not by themselves establish full-range coverage.

The returned `status` is `fail` for measured violations, `incomplete` for unknown
coverage or unscored/failed required peaks, and `pass` only when the declared
inventory is covered and meets the limits. `passes_prediction_limits` describes
the available numerical checks and is not a full-range pass when coverage is
incomplete. Shape warnings and traceability still require separate review.

`python examples/validate_wavelength_calibration.py NARROW.csv WORKING.csv --holdout 404.656 546.074`
saves per-line errors, failures, aggregate/interpolation-only statistics and a
comparison plot. Use `--holdout` for your validated isolated-line list and
`--narrow-initial`/`--working-initial` to reuse saved calibration identities.
`--holdout` is required because resolvable, well-isolated lines depend on the
lamp, slit and detector. `--max-error-nm` defaults to 0.5;
`--max-rms-nm` is optional, and
`--working-range-nm LOWER UPPER` restricts the assessment. The example does not
declare its selected holdouts to be a complete inventory.

The fitter also accepts `sample_masks`, a sequence containing one boolean array
per input scan, for explicitly excluding measurements. Wavelength corrections
use a shared coefficient budget bounded by `max_shift_nm` over the full
detector; a constant offset can use the whole budget. Parameters at bounds and
an exhausted shift budget are identified in the joint-fit diagnostics.

## Regridding stray-light corrections

`StrayLightCorrection.distribution()` recovers D = inverse(C) - I. Both matrix
classes provide `regrid(new_wavelength, bounds="raise")`; correction regridding
recovers D, regrids it, and reinverts I+D. For repeated trials, recover D once.
The approximation linearly interpolates a wavelength kernel on both axes and
accounts for changed incident sample spacing. `bounds="edge"` explicitly holds
kernel edge values when target wavelengths extend outside the input support.
It is not pixel-axis relabeling or an exact reconstruction of changed LSF
normalization, peak masks, clipping, and smoothing. Validate a regridded matrix
against representative measurements before using it in a calibration.
