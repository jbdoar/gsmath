"""Build LightTouch and native gsmath amplitude calibrations from LightTouch files.

Kept for its original command line; ``gsmath-amplitude calibrate`` does the
same and more (native inputs, optional stray light, comparison with a previous
calibration). The maths is :func:`gsmath.amplitude.amplitude_calibration`.
The native output is normalised by integration time (``PerExposure`` +
``Responsivity`` per count/ms); the ``.cal`` keeps LightTouch's convention.

Example::

    python -m gsmath.scripts.generate_amplitude_cal \
        artifacts/generate_amplitude_cal/cal_standard.std \
        artifacts/generate_amplitude_cal/cal_standard_raw_scan.csv \
        artifacts/generate_amplitude_cal/slmat.bin
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gsmath.amplitude import LIGHTTOUCH_INVALID as INVALID_RESPONSE  # noqa: F401  (its old name)
from gsmath.amplitude import amplitude_calibration
from gsmath.scripts.amplitude import read_spectrum, read_stray_light


def generate_amplitude_calibration(standard_path, raw_scan_path, stray_light_path, *,
                                   name="LightTouch amplitude calibration"):
    """The calibration result for a LightTouch standard, raw scan and stray-light matrix."""
    raw = read_spectrum(raw_scan_path)
    return amplitude_calibration(raw, read_spectrum(standard_path),
                                 stray_light=read_stray_light(stray_light_path, raw.wavelength),
                                 name=name)


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
                   help="LightTouch spectral radiance or irradiance standard file")
    p.add_argument("raw_scan", type=Path, metavar="RAW_SCAN.csv",
                   help="LightTouch raw (background-subtracted) scan of the standard")
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
    result = generate_amplitude_calibration(args.standard, args.raw_scan, args.stray_light,
                                            name=args.name)
    native_path, lighttouch_path = result.write(args.output_dir / args.output_stem)
    print(f"Native calibration: {native_path}")
    print(f"LightTouch amplitude calibration: {lighttouch_path}")
    print(result)


if __name__ == "__main__":
    main()
