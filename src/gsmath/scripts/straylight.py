# gsmath/scripts/straylight.py

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage

from gsmath import Spectrum
from gsmath.straylight import StrayLightDistribution, StrayLightCorrection

# load scans
# characterize peaks
# reject invalid scans
# build D
# build C
# apply C
# report/plot

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class Scan:
    path: Path
    set_wavelength: float
    spectrum: Spectrum
    center_wavelength: float | None = None
    fwhm: float | None = None
    valid: bool = False
    reason: str | None = None




# ---------------------------------------------------------------------------
# Load input
# ---------------------------------------------------------------------------

def parse_set_wavelength(path: Path) -> float:
    match = re.search(f"StrayLight_([0-9.]+)", path.stem)

    if match is None:
        raise ValueError(
            f"Cannot determine set wavelength from {path.name!r}"
        )

    return float(match.group(1))


def load_scans(
        directory: Path,
        pattern="StrayLight_*.csv",
        *,
        skip_header=None,
) -> list[Scan]:
    scans = []

    for path in sorted(directory.glob(pattern)):
        if skip_header is None:
            spectrum = Spectrum.from_lighttouch(path)
        else:
            spectrum = Spectrum.from_csv(path, skip_header=skip_header)
            
        scans.append(
            Scan(
                path=path,
                set_wavelength=parse_set_wavelength(path),
                spectrum=spectrum,
            )
        )

    return scans

# ---------------------------------------------------------------------------
# Scan characterization / rejection
# ---------------------------------------------------------------------------

def characterize_scan(
        scan: Scan,
        *,
        max_center_error: float,
        band_factor: float,
        guard_fwhm: float = 1.0,
) -> Scan:

    peaks = scan.spectrum.peaks(primary_only=True)

    if len(peaks.peak_indices) != 1:
        scan.reason = f"no primary peak"
        return scan

    properties = peaks.properties

    if (
        "left_ips" not in properties
        or "right_ips" not in properties
        or len(properties["left_ips"]) != 1
        or len(properties["right_ips"]) != 1
    ):
        scan.reason = "missing FWHM intersections"
        return scan

    center = float(peaks.center_wavelength)
    fwhm = float(peaks.fwhm)

    if not np.isfinite(center) or not np.isfinite(fwhm):
        scan.reason = "invalid peak geometry"
        return scan

    error = abs(center - scan.set_wavelength)

    if error > max_center_error:
        scan.reason = (
            f"center error {error:.3f} nm > "
            f"{max_center_error:.3f} nm"
        )
        scan.center_wavelength = center
        scan.fwhm = fwhm
        return scan

    scan.center_wavelength = center
    scan.fwhm = fwhm

    if not is_trustworthy_scan(
        scan,
        band_factor=band_factor,
        guard_fwhm=guard_fwhm,
    ):
        scan.reason = "insufficient edge margin"
        return scan
    
    scan.valid = True

    return scan


def is_trustworthy_scan(
        scan,
        *,
        band_factor: float,
        guard_fwhm: float = 1.0,
) -> bool:
    x = scan.spectrum.wavelength

    radius = (scan.fwhm / 2) * band_factor
    guard = scan.fwhm * guard_fwhm

    return (
        scan.center_wavelength - radius - guard >= x[0]
        and scan.center_wavelength + radius + guard <= x[-1]
    )


def recover_edge_scans(accepted, rejected, *, edge_fit_points=5):
    if len(accepted) < edge_fit_points:
        raise ValueError("Not enough accepted scans for edge recovery.")

    accepted = sorted(accepted, key=lambda s: s.set_wavelength)

    left = accepted[:edge_fit_points]
    right = accepted[-edge_fit_points:]

    center_left = np.polyfit(
        [s.set_wavelength for s in left],
        [s.center_wavelength for s in left],
        1,
    )

    center_right = np.polyfit(
        [s.set_wavelength for s in right],
        [s.center_wavelength for s in right],
        1,
    )

    fwhm_left = np.polyfit(
        [s.center_wavelength for s in left],
        [s.fwhm for s in left],
        1,
    )

    fwhm_right = np.polyfit(
        [s.center_wavelength for s in right],
        [s.fwhm for s in right],
        1,
    )
    
    lo = accepted[0].set_wavelength
    hi = accepted[-1].set_wavelength

    recovered = []

    for scan in rejected:
        if scan.reason != "missing FWHM intersections":
            continue

        if scan.set_wavelength < lo:
            center_fit = center_left
            fwhm_fit = fwhm_left
        elif scan.set_wavelength > hi:
            center_fit = center_right
            fwhm_fit = fwhm_right
        else:
            continue

        center = float(np.polyval(center_fit, scan.set_wavelength))
        fwhm = float(np.polyval(fwhm_fit, center))
        
        recovered.append(
            Scan(
                path=scan.path,
                set_wavelength=scan.set_wavelength,
                spectrum=scan.spectrum,
                center_wavelength=center,
                fwhm=fwhm,
                valid=True,
                reason="recovered edge scan",
            )
        )

    return recovered
    

def synthetic_distribution_column(
    template: Scan,
    target_center: float,
    detector_wavelength,
    *,
    band_factor: float,
    suppression: float,
    one_count: float,
):
    # Start from a trustworthy, fully measured LSF.
    spectrum = template.spectrum.normalize(by="peak")

    shift = target_center - template.center_wavelength
    shifted = spectrum.shift_wavelength(shift)

    # Assume local FWHM stays approximately constant initially.
    fwhm = template.fwhm
    radius = (fwhm / 2) * band_factor

    # IMPORTANT: denominator is calculated on the full shifted LSF,
    # whose wavelength axis has moved with it.
    y_full = shifted.values.copy()
    y_full[y_full < one_count] = 0

    in_band_full = (
        np.abs(shifted.wavelength - target_center) < radius
    )

    denominator = np.sum(y_full[in_band_full])

    if denominator <= 0:
        raise ValueError("Synthetic LSF has zero in-band signal.")

    # Now evaluate the shifted LSF on the actual detector grid.
    # Outside the shifted template's support we want zero, not
    # spline extrapolation.
    clipped = shifted.interpolate(
        xnew=detector_wavelength,
        spline_order=1,
        extrapolate=False,
    )

    y = np.nan_to_num(clipped.values, nan=0.0)
    y[y < one_count] = 0

    column = y / denominator * suppression

    in_band = (
        np.abs(detector_wavelength - target_center) < radius
    )
    column[in_band] = 0

    return column


def extend_distribution_with_lsf_templates(
    scans,
    centers,
    D_measured,
    detector_wavelength,
    *,
    band_factor,
    suppression,
    one_count,
    step=None,
):
    x = detector_wavelength

    if step is None:
        step = np.median(np.diff(centers))

    left_template = scans[0]
    right_template = scans[-1]

    left_centers = np.arange(
        centers[0] - step,
        x[0] - step,
        -step,
    )[::-1]

    right_centers = np.arange(
        centers[-1] + step,
        x[-1] + step,
        step,
    )

    left_columns = [
        synthetic_distribution_column(
            left_template,
            center,
            x,
            band_factor=band_factor,
            suppression=suppression,
            one_count=one_count,
        )
        for center in left_centers
    ]

    right_columns = [
        synthetic_distribution_column(
            right_template,
            center,
            x,
            band_factor=band_factor,
            suppression=suppression,
            one_count=one_count,
        )
        for center in right_centers
    ]

    parts = []

    if left_columns:
        parts.append(np.column_stack(left_columns))

    parts.append(D_measured)

    if right_columns:
        parts.append(np.column_stack(right_columns))

    D_extended = np.column_stack(parts)

    centers_extended = np.concatenate(
        (left_centers, centers, right_centers)
    )

    return centers_extended, D_extended


    
# ---------------------------------------------------------------------------
# D / C generation
# ---------------------------------------------------------------------------

def extend_distribution_columns(
    centers,
    D_measured,
    target_min,
    target_max,
    *,
    fit_points=5,
    step=None,
    mode="linear",
    floor=1e-12,
):
    centers = np.asarray(centers, dtype=float)

    if step is None:
        step = np.median(np.diff(centers))

    left_targets = np.arange(
        centers[0] - step,
        target_min - step,
        -step,
    )[::-1]

    right_targets = np.arange(
        centers[-1] + step,
        target_max + step,
        step,
    )

    left_columns = []
    right_columns = []

    for row in D_measured:
        if mode == "linear":
            y = row

        elif mode == "log":
            y = np.log10(np.maximum(row, floor))

        else:
            raise ValueError(f"Unknown extrapolation mode: {mode!r}")
        
        left_fit = np.polyfit(
            centers[:fit_points],
            y[:fit_points],
            1,
        )
        right_fit = np.polyfit(
            centers[-fit_points:],
            y[-fit_points:],
            1,
        )

        left = np.polyval(left_fit, left_targets)
        right = np.polyval(right_fit, right_targets)

        left = np.clip(left, 0, np.max(row[:fit_points]))
        right = np.clip(right, 0, np.max(row[-fit_points:]))
        
        if mode == "log":
            left = 10**left
            right = 10**right

        left_columns.append(left)
        right_columns.append(right)

    left_columns = np.asarray(left_columns)
    right_columns = np.asarray(right_columns)

    left_columns[left_columns < 0] = 0
    right_columns[right_columns < 0] = 0

    centers_ext = np.concatenate(
        (left_targets, centers, right_targets)
    )

    D_ext = np.column_stack(
        (left_columns, D_measured, right_columns)
    )

    return centers_ext, D_ext


def build_distribution(
        scans: list[Scan],
        *,
        band_factor: float = 3.0,
        suppression: float = 1.0,
        one_count: float = 1 / (2**16 - 1),
        gaussian_sigma: float = 1.0,
) -> StrayLightDistribution:

    if len(scans) < 2:
        raise ValueError("Not enough valid scans to construct D matrix.")

    scans = sorted(scans, key=lambda scan: scan.center_wavelength)

    x = scans[0].spectrum.wavelength
    N = len(x)

    D_columns = []
    centers = []

    for scan in scans:
        if not np.array_equal(scan.spectrum.wavelength, x):
            raise ValueError(
                f"{scan.path.name}: wavelength grid differs from other scans."
            )

        spectrum = scan.spectrum.normalize(by="peak")
        y = spectrum.values.copy()

        # Existing 16-bit noise-floor treatment
        y[y < one_count] = 0

        center = scan.center_wavelength
        fwhm = scan.fwhm

        radius = (fwhm / 2) * band_factor
        in_band = np.abs(x - center) < radius

        denominator = np.sum(y[in_band])

        if denominator <= 0:
            raise ValueError(
                f"{scan.path.name}: zero in-band signal."
            )

        column = y / denominator * suppression
        column[in_band] = 0

        D_columns.append(column)
        centers.append(center)

    D_measured = np.column_stack(D_columns)
    centers = np.asarray(centers)

    # Anchor D to zero at both detector limits.
    # centers = np.concatenate(([x[0]], centers, [x[-1]]))

    # D_measured = np.column_stack(
    #     (
    #         np.zeros(N),
    #         D_measured,
    #         np.zeros(N),
    #     )
    # )
    
    D_measured[D_measured < 0] = 0

    if gaussian_sigma:
        D_measured = scipy.ndimage.gaussian_filter(
            D_measured,
            sigma=gaussian_sigma,
        )

    # edge extrapolation
    # centers, D_measured = extend_distribution_columns(
    #     centers,
    #     D_measured,
    #     x[0],
    #     x[-1],
    #     fit_points=5,
    #     mode="linear",
    # )

    # distribution extension 
    # centers, D_measured = extend_distribution_with_lsf_templates(
    #     scans,
    #     centers,
    #     D_measured,
    #     x,
    #     band_factor=band_factor,
    #     suppression=suppression,
    #     one_count=one_count,
    # )

    # Interpolate sparse monochromator measurements onto the complete
    # spectroradiometer wavelength grid.
    interpolator = scipy.interpolate.RectBivariateSpline(
        x,
        centers,
        D_measured,
        kx=1,
        ky=1,
        s=0,
    )

    D = interpolator(x, x)

    return StrayLightDistribution(x, D)


# ---------------------------------------------------------------------------
# Reporting / plots
# ---------------------------------------------------------------------------

def write_scan_report(scans: list[Scan], path: Path):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "file",
                "set_wavelength_nm",
                "center_wavelength_nm",
                "fwhm_nm",
                "valid",
                "reason",
            ]
        )

        for scan in scans:
            writer.writerow(
                [
                    scan.path.name,
                    scan.set_wavelength,
                    scan.center_wavelength,
                    scan.fwhm,
                    scan.valid,
                    scan.reason or "",
                ]
            )


def evaluate_correction(
        scan: Scan,
        corrected: Spectrum,
        *,
        band_factor: float,
):
    if scan.center_wavelength is None or scan.fwhm is None:
        return None

    x = scan.spectrum.wavelength
    y0 = scan.spectrum.values.astype(float)
    y1 = corrected.values.astype(float)

    peak = np.nanmax(y0)

    if not np.isfinite(peak) or peak <= 0:
        return None

    # Use the ORIGINAL peak for both spectra.
    y0 = y0 / peak
    y1 = y1 / peak

    center = scan.center_wavelength
    radius = (scan.fwhm / 2) * band_factor

    lo = center - radius
    hi = center + radius

    left = x <= lo
    right = x >= hi
    in_band = (x > lo) & (x < hi)

    def integrate_regions(values):
        total = 0.0

        if np.count_nonzero(left) >= 2:
            total += np.trapezoid(values[left], x[left])

        if np.count_nonzero(right) >= 2:
            total += np.trapezoid(values[right], x[right])

        return float(total)

    # Positive OOB signal only. Negative correction artifacts are
    # deliberately accounted for separately below.
    oob_before = integrate_regions(
        np.clip(y0, 0, None)
    )
    oob_after = integrate_regions(
        np.clip(y1, 0, None)
    )

    suppression_ratio = (
        oob_after / oob_before
        if oob_before > 0
        else np.nan
    )

    # Negative artifacts over the entire spectrum.
    negative = np.clip(-y1, 0, None)

    negative_energy = float(
        np.trapezoid(negative, x)
    )

    minimum = float(np.nanmin(y1))

    # Useful sanity check: correction ideally shouldn't materially
    # alter the in-band energy.
    in_band_before = float(
        np.trapezoid(y0[in_band], x[in_band])
    )
    in_band_after = float(
        np.trapezoid(y1[in_band], x[in_band])
    )

    in_band_ratio = (
        in_band_after / in_band_before
        if in_band_before != 0
        else np.nan
    )

    return {
        "file": scan.path.name,
        "set_wavelength_nm": scan.set_wavelength,
        "center_wavelength_nm": center,
        "fwhm_nm": scan.fwhm,
        "oob_before": oob_before,
        "oob_after": oob_after,
        "suppression_ratio": suppression_ratio,
        "negative_energy": negative_energy,
        "minimum": minimum,
        "in_band_ratio": in_band_ratio,
    }


def write_evaluation_report(results, path: Path):
    results = [r for r in results if r is not None]

    if not results:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0]),
        )

        writer.writeheader()
        writer.writerows(results)


            
def plot_fwhm(scans: list[Scan], path: Path):
    good = [scan for scan in scans if scan.valid]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        [scan.center_wavelength for scan in good],
        [scan.fwhm for scan in good],
        ".-",
    )

    ax.set_xlabel("Center wavelength [nm]")
    ax.set_ylabel("FWHM [nm]")
    ax.set_title("Measured spectral bandpass")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correction(
    original: Spectrum,
    corrected: Spectrum,
    path: Path,
    ):
    fig, ax = plt.subplots(figsize=(9, 5))

    a = original.normalize(by="peak")
    b = corrected.normalize(by="peak")

    ax.plot(a.wavelength, a.values, label="Original")
    ax.plot(b.wavelength, b.values, label="Corrected")

    ax.set_yscale("log")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalized signal")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distribution_column(
        scan: Scan,
        D: StrayLightDistribution,
        band_factor: float,
        one_count: float,
        path: Path,
):
    x = scan.spectrum.wavelength
    y = scan.spectrum.values.copy()

    y[y < one_count] = 0
    
    # Interpolate D along the source-wavelength dimension
    # to this scan's actual measured center wavelength.
    d = np.array([
        np.interp(
            scan.center_wavelength,
            D.wavelength,
            row,
        )
        for row in D.values
    ])

    radius = (scan.fwhm / 2) * band_factor
    in_band = np.abs(scan.spectrum.wavelength - scan.center_wavelength) < radius

    denominator = np.sum(y[in_band])
    y_dnorm = y / denominator
    
    lo = scan.center_wavelength - radius
    hi = scan.center_wavelength + radius

    fig, ax = plt.subplots(figsize=(9,5 ))

    ax.semilogy(x, y_dnorm, label="Measured scan")
    ax.semilogy(x, d, label="D column")
    
    ax.axvline(lo, linestyle="--", label="In-band limits")
    ax.axvline(hi, linestyle="--")

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalized signal")
    ax.set_title(
        f"D column at {scan.center_wavelength:.2f} nm"
    )
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

def prepare_output(output: Path):
    matrices_dir = output / "matrices"
    plots_dir = output / "plots"

    matrices_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return output, matrices_dir, plots_dir


def generate(args) -> StrayLightCorrection:
    root = args.directory.resolve()

    output, matrices_dir, plots_dir = prepare_output(args.output.resolve())

    print("Loading scans...")
    scans = load_scans(
        root,
        pattern=args.scan_glob,
        skip_header=args.skip_header
    )

    if not scans:
        raise FileNotFoundError(f"No scans matching {args.scan_glob!r} in {root}")

    print("Characterizing peaks...")
    for scan in scans:
        characterize_scan(
            scan,
            max_center_error=args.max_center_error,
            band_factor=args.band_factor,
            guard_fwhm=1.0,
        )

    accepted = [scan for scan in scans if scan.valid]
    rejected = [scan for scan in scans if not scan.valid]

    # recovered = recover_edge_scans(
    #     accepted=accepted,
    #     rejected=rejected,
    # )

    # accepted.extend(recovered)
    accepted.sort(key=lambda scan: scan.center_wavelength)

    print(f"Loaded:   {len(scans)}")
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")

    write_scan_report(scans, output / "scan_evaluation.csv")

    plot_fwhm(accepted, plots_dir / "fwhm.png")

    print("Building stray-light distribution...")
    D = build_distribution(
        accepted,
        band_factor=args.band_factor,
        suppression=args.suppression,
        one_count=args.one_count,
        gaussian_sigma=args.gaussian_sigma,
    )

    # column_plots_dir = plots_dir / "distribution_columns"
    # column_plots_dir.mkdir(parents=True, exist_ok=True)

    # for scan in accepted:
    #     plot_distribution_column(
    #         scan,
    #         D,
    #         args.band_factor,
    #         args.one_count,
    #         column_plots_dir / f"{scan.path.stem}.png",
    #     )

    print("Computing correction matrix...")
    C = D.correction()

    print(f"C condition number: {np.linalg.cond(C.values):.6g}")

    D.write(matrices_dir / "D.gsmath")
    C.write(matrices_dir / "C.gsmath")
    D.write(matrices_dir / "D.bin")  # Legacy LightTouch-compatible matrix
    C.write(matrices_dir / "C.bin")

    D.plot(plots_dir / "distribution.png")
    C.plot(plots_dir / "correction.png")

    return C

    
def evaluate(args, correction=None):
    root = args.directory.resolve()

    scans = load_scans(
        root,
        pattern=args.scan_glob,
        skip_header=args.skip_header,
    )

    if not scans:
        raise FileNotFoundError(f"No scans matching {args.scan_glob!r} in {root}")

    if correction is None:
        wavelength = scans[0].spectrum.wavelength

        correction = StrayLightCorrection.read(args.matrix, wavelength)

    spectra = [scan.spectrum for scan in scans]

    print(f"Applying correction to {len(spectra)} spectra...")
    corrected = correction.apply(spectra)

    print("Characterizing evaluation scans...")
    for scan in scans:
        characterize_scan(
            scan,
            max_center_error=args.max_center_error,
            band_factor=args.band_factor,
            guard_fwhm=0.0,
        )

    results = [
        evaluate_correction(
            scan,
            corrected_spectrum,
            band_factor=args.band_factor,
        )
        for scan, corrected_spectrum in zip(scans, corrected)
    ]

    evaluation_dir = args.output.resolve() / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    write_evaluation_report(
        results,
        evaluation_dir / "metrics.csv",
    )

    if args.plot_every:
        for i in range(0, len(scans), args.plot_every):
            plot_correction(
                spectra[i],
                corrected[i],
                evaluation_dir / f"{scans[i].path.stem}.png"
            )
    
    # TODO:
    # Spectrum.write() / SpectrumCollection.write()
    # Evaluation plots/reporting will live here.

    return corrected

    
def run(args):
    C = generate(args)
    return evaluate(args, correction=C)

    
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_input_args(p):
    p.add_argument(
        "directory",
        type=Path,
        metavar="DIRECTORY",
        help="Directory containing the wavelength-scan CSV files.",
    )

    p.add_argument(
        "--scan-glob",
        default="StrayLight_*.csv",
        metavar="PATTERN",
        help="Glob pattern used to select scan files in DIRECTORY.",
    )

    p.add_argument(
        "--skip-header",
        type=int,
        default=None,
        metavar="ROWS",
        help=(
            "Number of leading CSV rows to skip. When omitted, read scans "
            "as LightTouch files."
        ),
    )


def add_output_args(p):
    p.add_argument(
        "--output",
        type=Path,
        default="artifacts/straylight",  # str: same help text on every OS; type converts it
        metavar="DIRECTORY",
        help="Directory for generated matrices, reports, and plots.",
    )

    p.add_argument(
        "--plot-every",
        type=int,
        default=5,
        metavar="N",
        help="Plot every Nth corrected scan; use 0 to disable scan plots.",
    )


def add_generation_args(p):
    p.add_argument(
        "--suppression",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Scale factor applied to the estimated out-of-band signal.",
    )

    p.add_argument(
        "--one-count",
        type=float,
        default=1 / (2**16 - 1),
        metavar="LEVEL",
        help=(
            "Normalized signal level treated as one detector count; lower "
            "values are set to zero."
        ),
    )

    p.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.0,
        metavar="SIGMA",
        help=(
            "Standard deviation, in matrix samples, for Gaussian smoothing; "
            "use 0 to disable smoothing."
        ),
    )


def add_analysis_args(p):
    p.add_argument(
        "--band-factor",
        type=float,
        default=3.0,
        metavar="FACTOR",
        help="Width of the in-band region expressed as a multiple of FWHM.",
    )

    p.add_argument(
        "--max-center-error",
        type=float,
        default=2.5,
        metavar="NM",
        help=(
            "Maximum difference in nanometers between a scan's set "
            "wavelength and measured peak center."
        ),
    )
    

def parser():
    p = argparse.ArgumentParser(
        description="Generate and evaluate spectral stray-light corrections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = p.add_subparsers(
        dest="command",
        required=True,
    )

    # generate + evaluate
    run_parser = sub.add_parser(
        "run",
        help="Generate and evaluate a correction matrix.",
        description=(
            "Generate a stray-light correction matrix from wavelength scans, "
            "then evaluate it against those scans."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_input_args(run_parser)
    add_output_args(run_parser)
    add_analysis_args(run_parser)
    add_generation_args(run_parser)
    run_parser.set_defaults(func=run)

    # generation only
    generate_parser = sub.add_parser(
        "generate",
        help="Generate a correction matrix.",
        description=(
            "Generate a stray-light distribution and correction matrix from "
            "wavelength scans."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_input_args(generate_parser)
    add_output_args(generate_parser)
    add_analysis_args(generate_parser)
    add_generation_args(generate_parser)
    generate_parser.set_defaults(func=generate)

    # evaluation only
    evaluate_parser = sub.add_parser(
        "evaluate",
        help="Evaluate an existing correction matrix.",
        description=(
            "Apply an existing correction matrix to wavelength scans and "
            "write evaluation metrics and plots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_input_args(evaluate_parser)
    add_output_args(evaluate_parser)
    add_analysis_args(evaluate_parser)
    evaluate_parser.add_argument(
        "--matrix",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to a native .gsmath or legacy .bin correction matrix.",
    )

    evaluate_parser.set_defaults(func=evaluate)

    return p
    

def main(argv=None):
    if argv is None:
        import sys
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    commands = {"run", "generate", "evaluate"}

    if argv and argv[0] not in commands:
        argv.insert(0, "run")

    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
