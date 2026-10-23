"""Build LightTouch and native gsmath amplitude calibrations from scans.

Example::

    python -m gsmath.scripts.generate_amplitude_cal \
        artifacts/generate_amplitude_cal/cal_standard.std \
        artifacts/generate_amplitude_cal/cal_standard_raw_scan.csv \
        artifacts/generate_amplitude_cal/slmat.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gsmath import (
    Calibration,
    Measurement,
    Responsivity,
    Spectrum,
    StrayLight,
    StrayLightCorrection,
)
from gsmath.units import Q_


RADIANCE_UNIT = "uW / cm^2 / sr / nm"
COUNT_UNIT = "count"
INVALID_RESPONSE = 1.0e4


def generate_amplitude_calibration(
    standard_path,
    raw_scan_path,
    stray_light_path,
    *,
    name="LightTouch amplitude calibration",
):
    """Build an amplitude-calibration pipeline and its spectral factors.

    Factors are standard radiance divided by stray-light-corrected raw counts,
    matching LightTouch's amplitude-calibration convention. The standard is
    linearly regridded to the raw scan wavelengths. Outside its support the
    factors are zero; nonpositive corrected counts within support receive the
    LightTouch invalid-response sentinel (10000).
    """
    standard_path = Path(standard_path).resolve()
    raw_scan_path = Path(raw_scan_path).resolve()
    stray_light_path = Path(stray_light_path).resolve()
    source_names = {
        "standard_file": standard_path.name,
        "raw_scan_file": raw_scan_path.name,
        "stray_light_file": stray_light_path.name,
    }

    standard = Spectrum.from_lighttouch(standard_path)
    raw = Spectrum.from_lighttouch(raw_scan_path).copy(value_unit=COUNT_UNIT)
    if stray_light_path.suffix.lower() == ".bin":
        expected_matrix_bytes = len(raw.wavelength) ** 2 * np.dtype(np.float32).itemsize
        actual_matrix_bytes = stray_light_path.stat().st_size
        if actual_matrix_bytes != expected_matrix_bytes:
            raise ValueError(
                f"stray-light file has {actual_matrix_bytes} bytes; expected "
                f"{expected_matrix_bytes} for a {len(raw.wavelength)}-sample float32 matrix"
            )
    if not np.all(np.isfinite(raw.values)):
        raise ValueError("raw calibration scan contains non-finite values")

    correction = StrayLightCorrection.read(stray_light_path, raw.wavelength)
    integration_ms = raw.metadata.get("integration_time")
    exposure = Q_(integration_ms, "ms") if integration_ms and integration_ms > 0 else None
    measurement = Measurement(
        raw=raw,
        exposure=exposure,
        label=raw.label,
        metadata={"source_file": raw_scan_path.name},
    )

    # Use the calibration step itself so the output model exactly matches the
    # operation that a later measurement will run.
    correction_only = Calibration(f"{name} stray-light correction", [
        StrayLight(correction),
    ])
    corrected = measurement.process(correction_only)

    standard_on_scan = standard.interpolate(xnew=raw.wavelength)
    reference = np.asarray(standard_on_scan.values, dtype=float)
    in_standard_range = np.isfinite(reference)
    corrected_values = np.asarray(corrected.values, dtype=float)
    usable_counts = np.isfinite(corrected_values) & (corrected_values > 0)

    factors = np.zeros(len(raw.wavelength), dtype=float)
    invalid = in_standard_range & ~usable_counts
    factors[invalid] = INVALID_RESPONSE
    valid = in_standard_range & usable_counts
    factors[valid] = reference[valid] / corrected_values[valid]
    if not np.any(valid):
        raise ValueError("no positive corrected samples overlap the standard")

    response = Spectrum(
        wavelength=raw.wavelength,
        values=factors,
        label=f"{name} responsivity",
        wavelength_unit=raw.wavelength_unit,
        value_unit=f"{RADIANCE_UNIT} / {COUNT_UNIT}",
        metadata={
            "method": "standard radiance / stray-light-corrected raw counts",
            "invalid_response": INVALID_RESPONSE,
            "standard_support_nm": [float(standard.wavelength[0]),
                                    float(standard.wavelength[-1])],
            **source_names,
        },
    )
    calibration = Calibration(name, [
        StrayLight(correction),
        Responsivity(response),
    ], metadata={
        "purpose": "LightTouch amplitude calibration",
        **source_names,
        "factor_method": "standard radiance / stray-light-corrected raw counts",
        "standard_interpolation": "linear; zero response outside standard support",
        "invalid_response": INVALID_RESPONSE,
        "integration_time_ms": integration_ms,
    })
    return calibration, response, measurement, {
        "valid_samples": int(np.count_nonzero(valid)),
        "outside_standard": int(np.count_nonzero(~in_standard_range)),
        "nonpositive_corrected_counts": int(np.count_nonzero(invalid)),
    }


def parser():
    p = argparse.ArgumentParser(
        description=(
            "Generate native gsmath and LightTouch amplitude calibration "
            "files from a calibration standard, its raw scan, and a "
            "stray-light correction matrix."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("standard", type=Path, metavar="CAL_STANDARD.std",
                   help="LightTouch spectral-radiance standard file")
    p.add_argument("raw_scan", type=Path, metavar="RAW_SCAN.csv",
                   help="LightTouch raw scan of the standard")
    p.add_argument("stray_light", type=Path, metavar="SLMAT",
                   help="Native .gsmath or legacy float32 .bin stray-light correction matrix")
    p.add_argument("-o", "--output-dir", type=Path,
                   default=Path("artifacts/generate_amplitude_cal/output"),
                   metavar="DIR", help="Directory for generated calibration files")
    p.add_argument("--output-stem", default="Amplitude Cal",
                   metavar="NAME", help="Filename stem for both outputs")
    p.add_argument("--name", default="LightTouch amplitude calibration",
                   help="Name stored in the native calibration")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if Path(args.output_stem).name != args.output_stem:
        raise SystemExit("--output-stem must be a filename, not a path")
    calibration, response, measurement, counts = generate_amplitude_calibration(
        args.standard, args.raw_scan, args.stray_light, name=args.name,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    native_path = calibration.write(args.output_dir / f"{args.output_stem}.gsmath")
    lighttouch_path = response.write(
        args.output_dir / f"{args.output_stem}.cal",
        integration_time_ms=(measurement.exposure.to("ms").magnitude
                             if measurement.exposure is not None else 0.0),
    )
    print(f"Native calibration: {native_path}")
    print(f"LightTouch amplitude calibration: {lighttouch_path}")
    print("Samples: " + ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
