"""``gsmath-straylight``: build and evaluate a spectral stray-light correction
from monochromator line scans.

    gsmath-straylight calibrate SCANS [--output DIR]
    gsmath-straylight evaluate SCANS --matrix C.gsmath [--output DIR]
    gsmath-straylight apply SPECTRUM... --matrix C.gsmath [--output DIR]

``SCANS`` is a directory of line scans, one per monochromator wavelength.
With a ``manifest.csv`` (columns ``file`` and ``set_wavelength_nm``, as the
stray-light station writes), the manifest lists them; otherwise every file
matching ``--scan-glob`` is read and its set wavelength taken from its name
(``StrayLight_550.0.csv``). A scan is a native ``.gsmath`` measurement (its
dark is subtracted) or spectrum, a LightTouch export, or a two-column CSV.

``apply`` corrects any spectra, broadband or not, read the same way, writing
``<name> corrected.csv`` (wavelength, original, corrected, difference) and a
plot for each. A spectrum on the matrix's pixels to within a hundredth of a
pixel is corrected by C directly; one on another grid inside C's support is
corrected by C regridded to it.

The science, and reading the scans, is in :mod:`gsmath.straylight`.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gsmath.straylight import (
    INTERPOLATIONS, ONE_COUNT, StrayLightCorrection, characterize_scan, evaluate_correction, plot_correction,
    read_line_scan, read_line_scans, stray_light_calibration, write_rows,
)

GRID_TOLERANCE = 0.01  # of a pixel: closer than this is the same pixel


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
        gaussian_sigma=args.gaussian_sigma, min_accepted=args.min_accepted,
        interpolation=args.interpolation)
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


def expand(paths):
    """Expand wildcards the shell left alone (Windows); keep order, drop repeats."""
    found = []
    for path in paths:
        matches = sorted(glob.glob(str(path))) if glob.has_magic(str(path)) else [str(path)]
        if not matches:
            raise ValueError(f"no file matches {path}")
        found += [Path(m) for m in matches if Path(m) not in found]
    return found


def correction_for(correction, spectrum, cache):
    """C itself when the spectrum is on its pixels, else C regridded to the
    spectrum's grid (which raises if the grid reaches outside C's support)."""
    x, w = correction.wavelength, spectrum.wavelength
    if len(w) == len(x) and np.max(np.abs(w - x)) <= GRID_TOLERANCE * np.min(np.abs(np.diff(x))):
        return correction
    key = w.tobytes()
    if key not in cache:
        cache[key] = correction.regrid(w)
    return cache[key]


def background_subtracted(spectrum):
    """A LightTouch header's Background Subtract flag, or None without one."""
    for line in spectrum.metadata.get("header", []):
        if line.endswith("[Background Subtract]"):
            return line.split()[0].lower() == "true"
    return None


def plot_applied(spectrum, corrected, path):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    top.plot(spectrum.wavelength, spectrum.values, label="Original")
    top.plot(spectrum.wavelength, corrected, label="Corrected")
    top.set_ylabel("Signal")
    top.set_title(spectrum.label)
    top.legend()
    bottom.plot(spectrum.wavelength, corrected - spectrum.values)
    bottom.axhline(0.0, color="k", lw=0.5)
    bottom.set_ylabel("Corrected - original")
    bottom.set_xlabel("Wavelength (nm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def apply(args) -> int:
    paths = expand(args.spectra)
    grid = None if args.matrix.suffix.lower() == ".gsmath" else read_line_scan(paths[0]).wavelength
    correction = StrayLightCorrection.read(args.matrix, grid)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = {}
    for path in paths:
        spectrum = read_line_scan(path)
        y = spectrum.values
        matrix = correction_for(correction, spectrum, cache)
        corrected = matrix.apply_matrix(y)
        np.savetxt(args.output / f"{path.stem} corrected.csv",
                   np.column_stack((spectrum.wavelength, y, corrected, corrected - y)),
                   delimiter=",", fmt="%.6f", comments="",
                   header="wavelength_nm,original,corrected,difference")
        if args.plot:
            plot_applied(spectrum, corrected, args.output / f"{path.stem} corrected.png")
        change = f"{100 * (corrected.sum() / y.sum() - 1):+.2f}%" if y.sum() else "n/a"
        negative = int((corrected < 0).sum())
        print(f"{path.name}: total signal {change}"
              f"{'' if matrix is correction else ' (matrix regridded)'}; "
              f"{negative} negative pixel{'' if negative == 1 else 's'}"
              f"{f', minimum {corrected.min():.4g}' if negative else ''}")
        if background_subtracted(spectrum) is False:
            print("  warning: the header says Background Subtract False; the correction expects "
                  "dark-subtracted signal and redistributes any offset as if it were light.")
    print(f"Corrected {len(paths)} spectr{'um' if len(paths) == 1 else 'a'} with {args.matrix.name}")
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
                   help="Standard deviation, in samples, for Gaussian smoothing (shifted: along "
                        "detector wavelength; linear: both matrix axes); use 0 to disable smoothing.")
    c.add_argument("--interpolation", choices=INTERPOLATIONS, default="shifted",
                   help="How D is filled in between scans: shifted (in distance from the line) or "
                        "linear (at fixed detector wavelength).")
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

    a = sub.add_parser(
        "apply", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Correct spectra with an existing correction matrix.",
        description="Apply a correction matrix to spectra and write each corrected spectrum as CSV "
                    "(wavelength, original, corrected, difference), with a plot.")
    a.add_argument("spectra", type=Path, nargs="+", metavar="SPECTRUM",
                   help="Spectrum files (LightTouch export, two-column CSV, or native .gsmath); "
                        "wildcards are expanded.")
    a.add_argument("--matrix", type=Path, required=True, metavar="PATH",
                   help="Path to a native .gsmath or legacy .bin correction matrix (a .bin is taken "
                        "to be on the first spectrum's grid).")
    a.add_argument("--output", type=Path, default="artifacts/straylight/applied", metavar="DIRECTORY",
                   help="Directory for the corrected spectra and plots.")
    a.add_argument("--no-plot", dest="plot", action="store_false", help="Skip the plots.")
    a.set_defaults(func=apply)
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
