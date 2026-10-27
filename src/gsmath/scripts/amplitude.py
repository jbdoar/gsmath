"""Amplitude calibration from files: ``gsmath-amplitude calibrate|evaluate|show|compare``.

``calibrate`` derives a responsivity from a dark-subtracted scan of a standard
source and writes it as a native ``.gsmath`` (normalised by integration time)
and a LightTouch ``.cal``. ``evaluate`` applies a calibration to a scan of a
source with certified values and compares (il)luminance, x, y and CCT.
``show`` describes a calibration file; ``compare`` compares two. Files may be
LightTouch exports (``.std``, ``.csv``, ``.cal``) or native ``.gsmath``. The
exit status is 0 for PASS (or done), 1 for FAIL, 2 for an error.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from gsmath import Measurement, Spectrum, StrayLightCorrection, read
from gsmath.amplitude import (amplitude_calibration, compare_responsivity, evaluate_amplitude, on_wavelengths, unit_text,
                              lighttouch_unit, read_amplitude_calibration, responsivity_of)
from gsmath import units


def read_spectrum(path):
    """A spectrum from a native ``.gsmath`` file or a LightTouch export, with its unit."""
    path = Path(path)
    if path.suffix.lower() == ".gsmath":
        spectrum = read(path)
        if not isinstance(spectrum, Spectrum):
            raise ValueError(f"{path} holds a {type(spectrum).__name__}, not a spectrum")
        return spectrum
    spectrum = Spectrum.from_lighttouch(path)
    unit = lighttouch_unit(spectrum)
    return spectrum if unit is None else spectrum.copy(value_unit=unit)


def read_stray_light(path, wavelength):
    return None if path is None else StrayLightCorrection.read(Path(path), np.asarray(wavelength))


def apply_calibration(scan, calibration, *, stray_light=None, integration_time_ms=None):
    """A dark-subtracted counts spectrum through a native amplitude calibration."""
    integration = integration_time_ms or scan.metadata.get("integration_time_ms",
                                                           scan.metadata.get("integration_time"))
    if not integration:
        raise ValueError("the scan's integration time is unknown")
    calibration = on_wavelengths(calibration, scan.wavelength)
    if stray_light is not None:
        from gsmath import StrayLight
        if any(isinstance(step, StrayLight) for step in calibration.steps):
            raise ValueError("the calibration already contains a stray-light matrix")
        calibration = calibration.with_steps([StrayLight(stray_light)], name=calibration.name)
    measurement = Measurement(raw=scan.copy(value_unit="count"),
                              exposure=units.Q_(float(integration), "ms"))
    return measurement.process(calibration)


def plot_calibration(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wavelength = np.asarray(result.responsivity.wavelength)
    fig, (top, middle, bottom) = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    top.plot(wavelength, result.counts_per_ms.values, color="black", lw=0.8)
    top.set(ylabel="Counts / ms", title="Scan of the standard (dark-subtracted"
            + (", stray-light corrected)" if result.stray_light is not None else ")"))
    middle.plot(result.standard.wavelength, result.standard.values, color="tab:orange", lw=0.8)
    middle.set(ylabel=units.format_unit(result.standard.value_unit, pretty=True), title="Standard")
    responsivity = np.where(result.valid, np.asarray(result.responsivity.values), np.nan)
    bottom.semilogy(wavelength, responsivity, color="tab:blue", lw=0.8)
    low = np.where(result.low_signal, responsivity, np.nan)
    bottom.semilogy(wavelength, low, color="tab:red", lw=1.2, label="low signal")
    if result.low_signal.any():
        bottom.legend(fontsize=8)
    bottom.set(xlabel="Wavelength (nm)", title="Responsivity",
               ylabel=units.format_unit(result.responsivity.value_unit, pretty=True))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibrate(args):
    scan = read_spectrum(args.scan)
    standard = read_spectrum(args.standard)
    result = amplitude_calibration(scan, standard,
                                   stray_light=read_stray_light(args.slc, scan.wavelength),
                                   integration_time_ms=args.integration_time, name=args.name)
    print(result)
    previous = None
    if args.previous is not None:
        previous = compare_responsivity(result.calibration, read_amplitude_calibration(args.previous),
                                        band_nm=args.band, tolerance=args.tolerance)
        print(comparison_text(previous, "new / previous"))
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        native, lighttouch = result.write(args.output / args.stem)
        plot_calibration(result, args.output / f"{args.stem}.png")
        summary = {**result.summary, "scan": str(args.scan), "standard": str(args.standard),
                   "stray_light": None if args.slc is None else str(args.slc)}
        if previous is not None:
            summary["previous"] = {k: v for k, v in previous.items() if k not in ("wavelength", "ratio")}
        (args.output / f"{args.stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {native} and {lighttouch}")
    return 0 if previous is None or previous["passed"] else 1


def evaluate(args):
    scan = read_spectrum(args.scan)
    calibration = read_amplitude_calibration(args.calibration)
    measured = apply_calibration(scan, calibration,
                                 stray_light=read_stray_light(args.slc, scan.wavelength),
                                 integration_time_ms=args.integration_time)
    evaluation = evaluate_amplitude(measured, read_spectrum(args.certified),
                                    Y_tolerance=args.y_tolerance / 100, xy_tolerance=args.xy_tolerance,
                                    cct_tolerance=args.cct_tolerance, band_nm=args.band)
    print(evaluation.table())
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        measured.write(args.output / "measured.gsmath")
        (args.output / "evaluation.json").write_text(json.dumps(evaluation.summary, indent=2),
                                                    encoding="utf-8")
        (args.output / "evaluation.txt").write_text(evaluation.table() + "\n", encoding="utf-8")
    return 0 if evaluation.passed else 1


def show(args):
    calibration = read_amplitude_calibration(args.calibration)
    responsivity = responsivity_of(calibration)
    values = np.asarray(responsivity.values, dtype=float)
    wavelength = np.asarray(responsivity.wavelength, dtype=float)
    valid = np.isfinite(values) & (values > 0) & (values < 1e30)
    steps = ", ".join(type(step).__name__ for step in calibration.steps)
    print(f"{args.calibration}: {calibration.name}")
    print(f"  steps: {steps}")
    print(f"  unit: {unit_text(responsivity.value_unit)}")
    print(f"  {len(values)} samples, {wavelength[0]:.1f}-{wavelength[-1]:.1f} nm; "
          f"{valid.sum()} with a positive responsivity"
          + (f" ({wavelength[valid][0]:.1f}-{wavelength[valid][-1]:.1f} nm)" if valid.any() else ""))
    time = calibration.metadata.get("calibration_integration_time_ms")
    if time:
        print(f"  calibration integration time: {time:g} ms")
    return 0


def comparison_text(result, name):
    return (f"Responsivity {name}, {result['band_nm'][0]:g}-{result['band_nm'][1]:g} nm: "
            f"median {result['median']:.4f}, 5-95% {result['p5']:.4f}-{result['p95']:.4f}, "
            f"largest {result['worst']:.4f}: "
            f"{'PASS' if result['passed'] else 'FAIL'} (+/-{100 * result['tolerance']:g} %)")


def compare(args):
    result = compare_responsivity(read_amplitude_calibration(args.calibration),
                                  read_amplitude_calibration(args.reference),
                                  band_nm=args.band, tolerance=args.tolerance)
    print(comparison_text(result, f"{args.calibration.name} / {args.reference.name}"))
    return 0 if result["passed"] else 1


def add_band(p, tolerance=None):
    p.add_argument("--band", type=float, nargs=2, default=(380.0, 780.0), metavar=("MIN", "MAX"),
                   help="wavelength band for responsivity or spectral ratios, nm")
    if tolerance is not None:
        p.add_argument("--tolerance", type=float, default=tolerance, metavar="FRACTION",
                       help="largest acceptable responsivity ratio deviation from 1 in the band")


def parser():
    p = argparse.ArgumentParser(
        prog="gsmath-amplitude", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build, check and compare amplitude (responsivity) calibrations from files.")
    p.set_defaults(func=None, help_parser=p)
    sub = p.add_subparsers(title="commands", metavar="COMMAND")

    c = sub.add_parser("calibrate", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                       help="Derive a calibration from a scan of a standard.",
                       description="Divide the standard (interpolated to the scan's wavelengths) by "
                                   "the dark-subtracted scan per ms; write .gsmath and .cal.")
    c.add_argument("scan", type=Path, help="dark-subtracted scan of the standard (LightTouch or .gsmath)")
    c.add_argument("--standard", type=Path, required=True,
                   help="the standard's certified values (.std, LightTouch scan, or .gsmath); "
                        "its unit sets the calibration's")
    c.add_argument("--slc", type=Path, help="stray-light matrix (.bin or .gsmath)")
    c.add_argument("--integration-time", type=float, metavar="MS",
                   help="the scan's integration time, if its file doesn't record it")
    c.add_argument("--previous", type=Path, metavar="CAL", help="compare with this calibration")
    add_band(c, tolerance=0.02)
    c.add_argument("--output", type=Path, metavar="DIR", help="write the calibration files here")
    c.add_argument("--stem", default="amplitude", help="file name for the outputs")
    c.add_argument("--name", default="amplitude calibration", help="name stored in the calibration")
    c.set_defaults(func=calibrate, help_parser=c)

    e = sub.add_parser("evaluate", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                       help="Check a calibration against a source with certified values.",
                       description="Apply a calibration to a dark-subtracted scan and compare "
                                   "(il)luminance, x, y and CCT with the certified spectrum.")
    e.add_argument("scan", type=Path, help="dark-subtracted scan of the source")
    e.add_argument("--calibration", type=Path, required=True, help="the calibration (.cal or .gsmath)")
    e.add_argument("--certified", type=Path, required=True, help="the source's certified spectrum")
    e.add_argument("--slc", type=Path, help="stray-light matrix, if the calibration has none")
    e.add_argument("--integration-time", type=float, metavar="MS",
                   help="the scan's integration time, if its file doesn't record it")
    e.add_argument("--y-tolerance", type=float, default=2.0, metavar="PERCENT",
                   help="(il)luminance tolerance")
    e.add_argument("--xy-tolerance", type=float, default=0.002, help="CIE x and y tolerance")
    e.add_argument("--cct-tolerance", type=float, default=30.0, metavar="K", help="CCT tolerance")
    add_band(e)
    e.add_argument("--output", type=Path, metavar="DIR", help="write the results here")
    e.set_defaults(func=evaluate, help_parser=e)

    s = sub.add_parser("show", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                       help="Describe a calibration file.", description="Describe a calibration file.")
    s.add_argument("calibration", type=Path)
    s.set_defaults(func=show, help_parser=s)

    m = sub.add_parser("compare", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                       help="Compare two calibration files.",
                       description="Ratio of two calibrations' responsivities within a band.")
    m.add_argument("calibration", type=Path)
    m.add_argument("reference", type=Path)
    add_band(m, tolerance=0.02)
    m.set_defaults(func=compare, help_parser=m)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.func is None:
        args.help_parser.print_help()
        return 2
    try:
        return args.func(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"gsmath-amplitude: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
