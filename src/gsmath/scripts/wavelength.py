"""Wavelength calibration from a lamp scan: ``gsmath-wavelength calibrate|evaluate``.

``calibrate`` identifies the lamp's emission lines, fits a pixel-to-wavelength
polynomial to their measured centres, and evaluates it. ``evaluate`` checks an
existing calibration (coefficients, a saved calibration, or the scan's own
wavelength axis) against the same lines. Both print a line-by-line table and a
verdict; ``--output`` also writes CSV, JSON, plot and native files. The exit
status is 0 for PASS and 1 for FAIL.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from gsmath import (Calibration, Spectrum, WavelengthCalibration, evaluate_wavelength,
                    fit_wavelength, lamp_lines, plot_evaluation, read)
from gsmath.wavelength_lines import METHODS


def read_scan(path):
    """A lamp scan from a native .gsmath spectrum, LightTouch export or CSV.

    A plain CSV's first column is its axis (pixel or wavelength) and its last
    column the signal; a header row is skipped.
    """
    path = Path(path)
    if path.suffix.lower() == ".gsmath":
        scan = read(path)
        if not isinstance(scan, Spectrum):
            raise ValueError(f"{path} holds a {type(scan).__name__}, not a processed spectrum")
        return scan
    try:
        return Spectrum.read(path)
    except (ValueError, KeyError):
        pass
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            try:
                rows.append([float(value) for value in row])
            except ValueError:
                if rows:
                    raise ValueError(f"{path}: non-numeric row after the data started: {row}")
    data = np.array(rows)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path}: expected at least two numeric columns")
    return Spectrum(wavelength=data[:, 0], values=data[:, -1], label=path.stem)


def read_calibration(args, n_pixels):
    """The calibration to evaluate: coefficients, a saved file, or None (the scan's axis)."""
    if args.coefficients:
        try:
            ascending = [float(value) for value in args.coefficients.split(",")]
        except ValueError:
            raise ValueError(f"--coefficients must be comma-separated numbers, not {args.coefficients!r}")
        return np.asarray(ascending[::-1], dtype=float)
    if args.calibration is None:
        return None
    path = args.calibration
    if path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(document["coefficients_descending"], dtype=float)
    calibration = read(path)
    steps = getattr(calibration, "steps", ())
    solutions = [getattr(step, "solution", None) for step in steps]
    solutions = [s for s in solutions if isinstance(s, WavelengthCalibration)]
    if not solutions:
        raise ValueError(f"{path} has no wavelength calibration step")
    return solutions[0]


def coefficients_document(evaluation, args):
    descending = [float(c) for c in evaluation.calibration.coefficients]
    return {
        "coefficients_ascending": descending[::-1],
        "coefficients_descending": descending,
        "coefficient_note": "wavelength_nm = sum(c[k] * pixel**k) for the ascending list",
        "pixel_domain": [int(v) for v in evaluation.calibration.pixel_domain],
        "degree": len(descending) - 1,
        "method": evaluation.method,
        "lines": str(args.lines),
        "scan": str(args.scan),
    }


def write_outputs(output, evaluation, scan, args, document=None):
    output.mkdir(parents=True, exist_ok=True)
    rows = evaluation.rows()
    with (output / "lines.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"command": args.command, "tolerance_nm": evaluation.tolerance_nm,
               "method": evaluation.method, "min_lines": evaluation.min_lines,
               **evaluation.summary}
    if document is not None:
        summary["calibration"] = document
        (output / "coefficients.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
        Calibration("Wavelength calibration", [evaluation.calibration.step()]).write(
            output / "calibration.gsmath")
        evaluation.calibration.apply(scan).write(output / "calibrated_scan.gsmath")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "table.txt").write_text(evaluation.table() + "\n", encoding="utf-8")
    plot_evaluation(evaluation, scan, output / "wavelength.png")


def report(evaluation, scan, args, document=None):
    print(evaluation.table())
    if document is not None:
        coefficients = "  ".join(f"{c:.10g}" for c in document["coefficients_ascending"])
        print(f"Coefficients c0..c{document['degree']} (ascending): {coefficients}")
    if args.output is not None:
        write_outputs(args.output, evaluation, scan, args, document)
        print(f"Artifacts: {args.output.resolve()}")
    return 0 if evaluation.passed else 1


def calibrate(args):
    scan = read_scan(args.scan)
    evaluation = fit_wavelength(
        scan, lamp_lines(args.lines), degree=args.degree, tolerance_nm=args.tolerance,
        method=args.method, search_nm=args.search, blend_fwhm=args.blend_fwhm,
        crowding_nm=args.crowding, iterations=args.iterations, direction=args.direction,
        wavelength_bounds_nm=args.wavelength_bounds, dispersion_bounds=args.dispersion_bounds)
    return report(evaluation, scan, args, coefficients_document(evaluation, args))


def evaluate(args):
    scan = read_scan(args.scan)
    calibration = read_calibration(args, len(scan.values))
    evaluation = evaluate_wavelength(
        scan, calibration, lamp_lines(args.lines), tolerance_nm=args.tolerance,
        method=args.method, search_nm=args.search, blend_fwhm=args.blend_fwhm,
        crowding_nm=args.crowding, min_lines=args.min_lines)
    return report(evaluation, scan, args)


def add_common_args(p):
    p.add_argument("scan", type=Path,
                   help="lamp scan: native .gsmath spectrum, LightTouch export, or CSV "
                        "(axis column first, signal column last)")
    p.add_argument("--lines", default="hgar", metavar="FILE",
                   help="reference lines: a bundled lamp name or a CSV with wavelength_nm, "
                        "species and use columns")
    p.add_argument("--tolerance", type=float, default=0.5, metavar="NM",
                   help="largest acceptable difference for each line")
    p.add_argument("--method", choices=METHODS, default="fit",
                   help="line centre: fitted line-shape mean, background-subtracted centroid, "
                        "or parabolic peak maximum")
    p.add_argument("--search", type=float, default=2.0, metavar="NM",
                   help="pair a peak with a reference line within this distance")
    p.add_argument("--blend-fwhm", type=float, default=1.5, metavar="WIDTHS",
                   help="lines closer than this many line widths are an unresolved blend")
    p.add_argument("--crowding", type=float, default=0.1, metavar="NM",
                   help="flag a line as crowded when its centroid and fitted mean differ by more")
    p.add_argument("--output", type=Path, metavar="DIR",
                   help="also write lines.csv, summary.json, table.txt and wavelength.png here")


def parser():
    p = argparse.ArgumentParser(
        prog="gsmath-wavelength",
        description="Fit or evaluate a pixel-to-wavelength calibration from a lamp scan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    calibrate_parser = sub.add_parser(
        "calibrate",
        help="Fit a new calibration and evaluate it.",
        description=(
            "Identify the lamp's lines (RANSAC), fit a polynomial to their measured "
            "centres, and compare each line with its reference. The scan's sample "
            "index is the pixel; any wavelength axis it has is ignored."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(calibrate_parser)
    calibrate_parser.add_argument("--degree", type=int, choices=(1, 2, 3, 4), default=3,
                                  help="polynomial degree")
    calibrate_parser.add_argument("--iterations", type=int, default=20000,
                                  help="RANSAC hypotheses for line identification")
    calibrate_parser.add_argument("--direction", choices=("either", "increasing", "decreasing"),
                                  default="either", help="wavelength direction along the detector")
    calibrate_parser.add_argument("--wavelength-bounds", nargs=2, type=float, metavar=("LOW", "HIGH"),
                                  help="plausible wavelength range of the detector, nm")
    calibrate_parser.add_argument("--dispersion-bounds", nargs=2, type=float, metavar=("LOW", "HIGH"),
                                  help="plausible nm per pixel")
    calibrate_parser.set_defaults(func=calibrate)

    evaluate_parser = sub.add_parser(
        "evaluate",
        help="Evaluate an existing calibration.",
        description=(
            "Compare each lamp line's measured wavelength with its reference, through "
            "the given calibration, or the scan's own wavelength axis if none is given."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(evaluate_parser)
    source = evaluate_parser.add_mutually_exclusive_group()
    source.add_argument("--coefficients", metavar="C0,C1,...",
                        help="polynomial coefficients, comma-separated and ascending: "
                             "wavelength = c0 + c1*pixel + ... (write --coefficients=... "
                             "if c0 is negative)")
    source.add_argument("--calibration", type=Path, metavar="FILE",
                        help="coefficients.json from calibrate, or a .gsmath calibration")
    evaluate_parser.add_argument("--min-lines", type=int, default=5,
                                 help="fewest measured lines for a PASS")
    evaluate_parser.set_defaults(func=evaluate)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"gsmath-wavelength: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
