"""Build, save, reload, and apply a small spectral calibration pipeline.

Run::

    python examples/measurement_pipeline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gsmath import Calibration, Measurement, PerExposure, Responsivity, Spectrum, SubtractDark
from gsmath.units import Q_


def synthetic_run():
    """Return a known-radiance measurement and its calibration."""
    wavelength = np.arange(380., 781., 5.)
    radiance = 0.1 * np.exp(-0.5 * ((wavelength - 550.) / 35.) ** 2)
    exposure = Q_(100., "ms")
    responsivity = Spectrum(
        wavelength,
        0.001 * (1. + 0.15 * (wavelength - 550.) / 200.),
        label="Responsivity",
        value_unit="uW / cm^2 / sr / nm * ms / count",
    )
    dark = Spectrum(wavelength, np.full(len(wavelength), 10.),
                    label="Dark", value_unit="count")
    raw = Spectrum(wavelength, radiance / responsivity.values * exposure.magnitude + dark.values,
                   label="Synthetic lamp", value_unit="count")
    calibration = Calibration("Synthetic spectral radiance", [
        SubtractDark(), PerExposure(), Responsivity(responsivity),
    ])
    measurement = Measurement(
        raw=raw, dark=dark, exposure=exposure, label="Synthetic lamp",
        settings={"averages": 1}, instrument={"model": "synthetic"},
        calibration=calibration.reference,
    )
    return measurement, calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("artifacts/examples/measurement_pipeline"),
                        help="Directory for the saved measurement, calibration, and result")
    args = parser.parse_args(argv)

    measurement, calibration = synthetic_run()
    args.output.mkdir(parents=True, exist_ok=True)
    measurement_path = measurement.write(args.output / "measurement.gsmath")
    calibration_path = calibration.write(args.output / "calibration.gsmath")
    restored = Measurement.read(measurement_path).process(Calibration.read(calibration_path))
    result_path = restored.write(args.output / "radiance.gsmath")
    color = restored.color(observer="2deg")
    print(f"Processed {len(restored.wavelength)} samples to {result_path}")
    print(f"Y={color.Y:.6g}, x={color.x:.6f}, y={color.y:.6f}")


if __name__ == "__main__":
    main()
