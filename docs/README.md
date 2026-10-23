# Documentation

The documentation is split by purpose:

- [User guide](user-guide.md) explains concepts, workflows, feature behavior,
  caveats, and focused examples.
- [API reference](api.md) is a compact inventory of every name exported from
  `gsmath`, grouped by task.
- [Runnable examples](../examples/README.md) lists end-to-end scripts and their
  input requirements.
- [Reference-data notice](../NOTICE.md) records sources, transformations, and
  licensing for bundled CIE and Robertson data.

## Topics

- Fundamentals: [spectrum objects](user-guide.md#spectrum-objects),
  [units](user-guide.md#units), and
  [transformations](user-guide.md#transforming-spectra)
- Analysis and QA: [comparing spectra](user-guide.md#comparing-spectra),
  profiles, tolerances, summaries, and reports
- Data exchange: [reading and writing](user-guide.md#reading-and-writing-spectra)
- Imaging: [images](user-guide.md#images), camera colorimetry, uniformity,
  MTF, mosaics, raw files, and perspective correction
- Reproducibility: [measurements and calibration](user-guide.md#measurements-and-calibration)
  and high-dynamic-range exposure series
- Instrument work: [experimental wavelength calibration](user-guide.md#experimental-wavelength-calibration),
  joint lamp profiles, held-out validation, and stray-light correction

## Conventions

Examples import the stable public surface from `gsmath`. Array values are
ordinary NumPy arrays; units are represented by Pint. Most transformations
return a new object rather than modifying their input. Numeric wavelength
arguments are interpreted in the object's current wavelength unit unless an
example says otherwise.
