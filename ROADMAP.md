# Roadmap

Planned and deferred work, so it isn't forgotten. Ordered roughly by
priority within each section. Move an item into the changelog or delete it
when it lands.

## Spectra: next

- **Colour rendering of illuminants.** Implement CIE 13.3-1995 CRI as a
  separate `Spectrum.color_rendering()` analysis: general `R_a`, special
  indices R1-R14, the prescribed test-colour reflectance spectra, generated
  Planckian/daylight reference illuminants, chromatic adaptation, and
  validation against published reference results. Record the source and
  redistribution terms for every bundled reflectance table. Design the result
  so CIE 224:2017 `R_f` and its 99-sample set can be added without changing
  the CRI API; report both during the standards transition.

## Imaging: now

- **Calibrating a scientific camera as an imaging colorimeter.** Done so
  far: spectral sensitivity from a monochromator scan, predicted camera
  signals for any spectrum, colour matrix fits (least squares with white
  preservation, Δu′v′ + ΔY, ΔE₀₀) with validation, and Bayer site means.
  Acquisition is done too: dark model (temperature-dependent, for the
  uncooled camera), saturation masking, linearity and its correction, flat
  building, and an end-to-end simulated example
  (`examples/camera_colorimeter.py`). Still to do:
  - an uncertainty budget: noise, matrix fit, flat, dark and linearity
    contributions propagated to Y and u′v′;
  - per-pixel or per-region doubling temperatures for the dark rate, if
    the real sensor's hot pixels turn out to follow a different exponent
    from the rest (the model now fits one for the whole sensor);
  - deconvolving the monochromator's bandwidth from the measured
    sensitivity, which matters for bands that aren't narrow next to the
    sensitivity's features;
  - the simulated example's uniformity min/max ratio runs about 0.006 low
    at 64×96 pixels (for every seed), likely the smoothed flat's edge
    bias: confirm the cause, and whether `build_flat` should fit the edges
    rather than smooth them;
  - checking it against representative real-camera datasets.
- **Filter-wheel (monochrome, XYZ-filter) camera**: frames per filter
  combined into channels (each with its own exposure and dark), then the
  same matrix fitting.
- **NIST four-colour method** (Ohno & Hardis) for display-specific
  colorimeter correction: implement exactly from the paper. The current
  `fit_color_matrix` on a display's R, G, B and white is already exact
  for that display.

## Imaging: next

- **Named uniformity standards** as presets on the uniformity engine
  (`Image.uniformity`, which already takes any point layout and spot size):
  sample layout, spot size and formulas, implemented from the documents
  rather than from memory. Candidates: IDMS sampled uniformity
  (5/9/13/25 points), VESA FPDM, ISO 9241-305, and application specifications.
  Pass/fail limits already work (`result.check(spec)`), so a preset is a
  layout, a spot size and a specification mapping.
- **Mura**, e.g. SEMI D31 SEMU: local, perceptually weighted
  non-uniformity. This is separate from min/max uniformity.
- **Lens distortion correction**, fitted from a calibration target. Also
  needed for exact keystone correction when the lens distorts, which
  shows as a raised `Quad.edge_rms`.
- **Inverse lookup for per-pixel pixel maps**, so `interpolate` and
  undistortion can resample from non-separable maps.
- **Imaging stray light and veiling glare correction**, by point spread
  function deconvolution. It matters for contrast and for dark regions
  next to bright ones.
- **Frame stacking and noise**: averaging repeated frames, plus shot and
  read noise estimates propagated into results.
- **Multi-edge MTF charts**: find the edges of ISO 12233 or SFRplus-style
  targets automatically.
- **Linearising encoded images** (sRGB and other transfer curves) before
  MTF or colorimetry. Also the reverse: encoding float data as
  uint8/uint16 for PNG previews.

## Imaging: later

- Hyperspectral cubes: an image whose channel axis is a wavelength grid,
  with a `Spectrum` at any pixel or region.
- Image summary and comparison profiles, like `Spectrum.summary(profile=...)`,
  and collections of images.
- Region outlines drawn on image plots (sample spots, active area).
- FITS reading and writing; EXIF metadata (exposure, ISO) from JPEG and
  TIFF.
- Display detection that copes with content frames and bright
  surroundings, not only a white frame on a dark surround.
- Better demosaicing (e.g. Malvar–He–Cutler) and X-Trans sensors.
- Analyses on line cuts, such as FWHM and edge width.
- Checking the RAW reader against real camera files from several makers.

## Architecture cleanup

The major cleanup items from the imaging architecture review are complete:

Items 1–3 (one colorimetry model, step boilerplate and naming, public
helpers) are done.

Item 4 (one codec and record registry for native files, one format table)
is done.

Item 5 (the mosaic as an `Image` attribute; sensor temperature and
per-channel exposure on `Measurement`) is done.

Item 6 (`Sampled` for what all data shares; a channel-aware `Series` with
`Spectrum`, `SpectralSensitivity` and `LineCut` as kinds of it; one `copy`
rule) is done.

Item 7 (one `Result` protocol: summaries with declared units, export,
`check` against minimum/maximum limits, `compare` between results; images
summarise, compare and check like spectra) is done. The architecture review
is complete.

## Housekeeping

- **Frames up to 65 MP** are expected eventually. At that size an X, Y, Z
  frame in float64 is about 1.6 GB, so memory comes before speed: avoid
  whole-frame copies (the pattern of the full-field uniformity rewrite),
  and consider float32 for calibrated frames. For speed, an optional
  `gsmath[fast]` extra using numba for the per-pixel hot paths (colour
  maps, demosaicing, full-field metrics), with NumPy as the fallback, would
  fuse each chain of operations into one pass, roughly 5-10x on these.
  Current 12 MP timings: demosaic 1.1 s, dark + flat + demosaic + exposure
  1.9 s, 3x3 uniformity 21 ms, full-field uniformity 1.7 s.

- (The two items that were here are now architecture items 1 and 6.)
