"""``gsmath-straylight``: build and evaluate a spectral stray-light correction
from monochromator line scans.

    gsmath-straylight calibrate SCANS [--output DIR]
    gsmath-straylight evaluate SCANS --matrix C.gsmath [--output DIR]

``SCANS`` is a directory of line scans, one per monochromator wavelength.
With a ``manifest.csv`` (columns ``file`` and ``set_wavelength_nm``, as the
stray-light station writes), the manifest lists them; otherwise every file
matching ``--scan-glob`` is read and its set wavelength taken from its name
(``StrayLight_550.0.csv``). A scan is a native ``.gsmath`` measurement (its
dark is subtracted) or spectrum, a LightTouch export, or a two-column CSV.

The science, and reading the scans, is in :mod:`gsmath.straylight`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gsmath.straylight import (
    ONE_COUNT, StrayLightCorrection, characterize_scan, evaluate_correction, plot_correction,
    read_line_scans, stray_light_calibration, write_rows,
)


def plot_scans(scans, corrected, directory, every):
    if not every:
        return
    directory.mkdir(parents=True, exist_ok=True)
    for scan, result in list(zip(scans, corrected))[::every]:
        plot_correction(scan.spectrum, result, directory / f"{Path(scan.label).stem}.png")


def calibrate(args) -> int:
    scans = read_line_scans(args.directory, args.scan_glob)
    print(f"{len(scans)} scans, {scans[0].set_wavelength_nm:g}-{scans[-1].set_wavelength_nm:g} nm")
    result = stray_light_calibration(
        scans, max_center_error_nm=args.max_center_error, band_factor=args.band_factor,
        suppression=args.suppression, one_count=args.one_count,
        gaussian_sigma=args.gaussian_sigma, min_accepted=args.min_accepted)
    matrix = result.write(args.output)
    (args.output / "evaluation.txt").write_text(result.evaluation.table() + "\n", encoding="utf-8")
    accepted = result.accepted
    plot_scans(accepted, result.correction.apply([s.spectrum for s in accepted]),
               args.output / "plots" / "scans", args.plot_every)
    report(result.summary, result.scans)
    print(f"Correction matrix: {matrix} (and C.bin)")
    print(f"Artifacts: {args.output.resolve()}")
    return 0


def evaluate(args) -> int:
    scans = read_line_scans(args.directory, args.scan_glob)
    correction = StrayLightCorrection.read(args.matrix, scans[0].spectrum.wavelength)
    scans = [characterize_scan(s, max_center_error_nm=args.max_center_error,
                               band_factor=args.band_factor, guard_fwhm=0.0) for s in scans]
    evaluation = evaluate_correction(correction, scans, band_factor=args.band_factor)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(evaluation.rows(), args.output / "evaluation.csv")
    (args.output / "evaluation.txt").write_text(evaluation.table() + "\n", encoding="utf-8")
    measured = [s for s in scans if s.center_nm is not None]
    plot_scans(measured, correction.apply([s.spectrum for s in measured]),
               args.output / "plots" / "scans", args.plot_every)
    print(f"{args.matrix.name} on {len(evaluation.metrics)} of {len(scans)} scans")
    report_evaluation(evaluation.summary)
    print(f"Artifacts: {args.output.resolve()}")
    return 0


def report(summary, scans):
    print(f"Accepted {summary['accepted']} of {summary['scans']} scans; "
          f"centres {summary['centers_nm'][0]:.1f}-{summary['centers_nm'][1]:.1f} nm, "
          f"detector {summary['detector_nm'][0]:.1f}-{summary['detector_nm'][1]:.1f} nm")
    for scan in scans:
        if not scan.accepted:
            print(f"  rejected {scan.label}: {scan.reason}")
    if summary["edge_extrapolation"]:
        print("  D's end columns are extended beyond the outermost scans, not measured.")
    print(f"Condition number of C: {summary['condition_number']:.4g}")
    report_evaluation(summary["evaluation"])


def report_evaluation(summary):
    if not summary["scans"]:
        print("No scan with a measured peak to evaluate.")
        return
    print(f"Out-of-band signal after correction: median {summary['median_suppression_ratio']:.3f} "
          f"of before, worst {summary['max_suppression_ratio']:.3f}")
    print(f"Negative signal: worst area {summary['max_negative_area']:.3g} nm, "
          f"minimum {summary['minimum']:.3g} of peak")
    print(f"In-band signal after / before: {summary['in_band_ratio_min']:.4f}"
          f"-{summary['in_band_ratio_max']:.4f}")


def add_input_args(p):
    p.add_argument("directory", type=Path, metavar="DIRECTORY",
                   help="Directory containing the wavelength-scan files (and optionally manifest.csv).")
    p.add_argument("--scan-glob", default="StrayLight_*.csv", metavar="PATTERN",
                   help="Glob pattern used to select scan files when there is no manifest.csv.")
    p.add_argument("--output", type=Path, default="artifacts/straylight", metavar="DIRECTORY",
                   help="Directory for generated matrices, reports, and plots.")
    p.add_argument("--plot-every", type=int, default=5, metavar="N",
                   help="Plot every Nth corrected scan; use 0 to disable scan plots.")
    p.add_argument("--band-factor", type=float, default=3.0, metavar="FACTOR",
                   help="Width of the in-band region expressed as a multiple of FWHM.")
    p.add_argument("--max-center-error", type=float, default=2.5, metavar="NM",
                   help="Maximum difference in nanometers between a scan's set wavelength and "
                        "measured peak center.")


def parser():
    p = argparse.ArgumentParser(
        prog="gsmath-straylight",
        description="Generate and evaluate spectral stray-light corrections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser(
        "calibrate", aliases=["run", "generate"], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Generate a correction matrix and evaluate it on the same scans.",
        description="Generate a stray-light distribution and correction matrix from wavelength "
                    "scans, then evaluate the correction against those scans.")
    add_input_args(c)
    c.add_argument("--suppression", type=float, default=1.0, metavar="FACTOR",
                   help="Scale factor applied to the estimated out-of-band signal.")
    c.add_argument("--one-count", type=float, default=ONE_COUNT, metavar="LEVEL",
                   help="Normalized signal level treated as one detector count; lower values are "
                        "set to zero.")
    c.add_argument("--gaussian-sigma", type=float, default=1.0, metavar="SIGMA",
                   help="Standard deviation, in matrix samples, for Gaussian smoothing; use 0 to "
                        "disable smoothing.")
    c.add_argument("--min-accepted", type=int, default=2, metavar="N",
                   help="Fewest accepted scans to build a matrix from.")
    c.set_defaults(func=calibrate)

    e = sub.add_parser(
        "evaluate", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Evaluate an existing correction matrix.",
        description="Apply an existing correction matrix to wavelength scans and write evaluation "
                    "metrics and plots.")
    add_input_args(e)
    e.add_argument("--matrix", type=Path, required=True, metavar="PATH",
                   help="Path to a native .gsmath or legacy .bin correction matrix.")
    e.set_defaults(func=evaluate)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"gsmath-straylight: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
