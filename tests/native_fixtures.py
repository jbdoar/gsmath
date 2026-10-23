"""Objects for the native-format compatibility fixtures in tests/data/native.

The fixture files were written from these objects by the native writer as it
was before the codec-registry refactoring (2026-10-20);
test_native_compatibility.py checks the current code still reads them and
writes them identically. Regenerate only for a deliberate format change.

Changes so far: 2026-10-21, the raw-frame fixtures (measurement_image,
calibration_image) store the colour-filter mosaic as an image field instead
of inside metadata; legacy_mosaic_in_metadata.gsmath keeps the older layout,
which must still read. On 2026-10-23, identifying instrument and absolute-path
metadata were replaced with neutral examples in the spectral measurement and
calibration fixtures. To regenerate:

    python tests/native_fixtures.py
"""

from pathlib import Path

import numpy as np

from gsmath import (
    AdaptiveSmooth, BlackLevel, Calibration, ColorMatrix, DarkModel, Demosaic, Despike, FlatField,
    Image, Linearize, MaskSaturated, Measurement, PixelMap, Responsivity, Spectrum, StrayLight,
    SubtractDark, WavelengthAxis,
)
from gsmath.mosaic import Mosaic
from gsmath.straylight import StrayLightCorrection, StrayLightDistribution
from gsmath.units import Q_
from gsmath.wavelength import CalibrationPeak, LineMatch, WavelengthCalibration

DIRECTORY = Path(__file__).parent / "data" / "native"
WL = np.array([400, 401, 402, 403], dtype=np.int32)


def spectrum():
    return Spectrum(WL, np.array([1.25, 2.5, 0.0, -1.0]), label="lamp", value_unit="uW/cm^2/sr/nm",
                    metadata={"instrument": "example", "averages": 3, "nested": {"a": [1, 2]}})


def wavelength_solution():
    peaks = (CalibrationPeak(10.5, 100.0, 90.0, 2.0), CalibrationPeak(500.25, 80.0, 70.0, 2.5),
             CalibrationPeak(900.0, 60.0, 50.0, 3.0))
    matches = (LineMatch(0, 10.5, 404.656, 404.6, -0.056),
               LineMatch(1, 500.25, 546.074, 546.1, 0.026),
               LineMatch(2, 900.0, 763.511, 763.5, -0.011))
    return WavelengthCalibration(np.array([1e-6, 0.3, 401.0]), matches, peaks, (0.0, 1023.0),
                                 {"method": "example"})


def raw_image():
    values = (np.arange(48, dtype=np.uint16).reshape(6, 8) * 37) % 4000
    return Image(values, label="raw", value_unit="count",
                 mosaic=Mosaic("RGGB", [64, 64, 64, 64], 4095.0))


def objects():
    """{file name: object} for every native record and step-argument kind."""
    s = spectrum()
    dark = Spectrum(WL, np.full(4, 0.5), label="dark", value_unit="uW/cm^2/sr/nm")
    rgb = Image(np.linspace(0, 1, 36).reshape(3, 4, 3), label="rgb", value_unit="cd/m^2",
                channels=("X", "Y", "Z"), pixel_map=PixelMap.linear((3, 4), 0.5, unit="deg"),
                metadata={"note": "x"})
    flat = Image(np.linspace(0.8, 1.0, 48).reshape(6, 8), label="flat")
    darks = [raw_image().copy(values=np.full((6, 8), 100.0) + t) for t in (1, 2, 4)]
    dark_model = DarkModel.fit(darks, [Q_(t, "ms") for t in (10, 20, 40)])
    return {
        "spectrum.gsmath": s,
        "measurement_spectrum.gsmath": Measurement(
            raw=s, dark=dark, exposure=Q_(100, "ms"), label="run 1",
            settings={"averages": 3}, instrument={"model": "example-spectrometer"},
            metadata={"operator": "lab"}, calibration={"name": "c", "fingerprint": "abc"}),
        "calibration_spectral.gsmath": Calibration("spectral", [
            SubtractDark(optional=True),
            StrayLight(StrayLightCorrection(WL.astype(float), np.eye(4, dtype=np.float32) * 0.5)),
            WavelengthAxis(wavelength_solution()),
            Responsivity(s),
            Responsivity(Q_(3, "1/sr")),
            Responsivity(2.0),
            Responsivity(np.array([1.0, 2.0, 3.0, 4.0])),
            Despike(),
            AdaptiveSmooth(),
        ], metadata={"sources": {"cal": "Amplitude.cal"}}),
        "straylight_distribution.gsmath": StrayLightDistribution(WL.astype(float), np.eye(4) * 0.1),
        "straylight_correction.gsmath": StrayLightCorrection(WL.astype(float), np.eye(4) * 0.9),
        "image.gsmath": rgb,
        "measurement_image.gsmath": Measurement(raw=raw_image(), dark=raw_image(),
                                                exposure=Q_(20, "ms"), label="frame"),
        "calibration_image.gsmath": Calibration("camera", [
            MaskSaturated(), dark_model.step(),
            Linearize(np.array([[0.0] * 3, [1000.0] * 3]), np.array([[0.0] * 3, [1010.0] * 3])),
            FlatField(flat), BlackLevel(optional=True), Demosaic("superpixel"),
            ColorMatrix(np.eye(3) * 2, unit="cd/m^2 / count"),
        ]),
    }


def write_all(directory=DIRECTORY):
    directory.mkdir(parents=True, exist_ok=True)
    for name, obj in objects().items():
        obj.write(directory / name)


if __name__ == "__main__":
    write_all()
