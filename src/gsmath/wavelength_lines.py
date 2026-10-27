"""Wavelength calibration from emission lines: measure centres, fit, evaluate.

A lamp's reference lines are a labelled :class:`~gsmath.Collection` of
:class:`EmissionLine`. Detected peaks are paired with them through a
calibration (RANSAC's when fitting from scratch), each paired line's centre is
measured in detector pixels, and a polynomial is fitted to the centres.
Evaluation maps each centre to nm through the calibration under test and
compares it with its reference line using :func:`~gsmath.compare`, with a
per-line tolerance (default 0.5 nm).

Line centre methods:

- ``'fit'`` (default): the mean of an asymmetric line shape fitted to the
  samples around the peak, with a linear background;
- ``'centroid'``: the signal-weighted mean pixel above a linear background;
- ``'peak'``: the parabolic maximum from peak detection.

An asymmetric line's maximum sits several tenths of a nm from its mean, so the
method matters at a 0.5 nm tolerance. Lines with another listed line within
``blend_fwhm`` widths are unresolved blends: they are reported, not fitted or
judged.

Every measured line also gets both a fitted mean and a centroid, whatever the
method. On an isolated line the two agree within a few hundredths of a nm; a
neighbour's tail inside the window pulls the centroid away. Lines where they
differ by more than ``crowding_nm`` are flagged as crowded: still fitted and
judged, but less trustworthy.
"""

from dataclasses import dataclass, replace
import csv
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import least_squares

from .analysis.base import Result
from .collection import Collection
from .comparison import compare
from .linefit import LineSpreadFunction
from .profile import Field, Profile
from .wavelength import (CalibrationPeak, LineMatch, WavelengthCalibration,
                         detect_calibration_peaks, identify_calibration_lines,
                         _assign, _positive, _vector)

METHODS = ("fit", "centroid", "peak")

LAMPS_DIR = Path(__file__).parent / "reference" / "data" / "lamps"


@dataclass(frozen=True)
class EmissionLine:
    """A reference emission line, labelled by species and wavelength.

    ``use=False`` keeps the line out of the fit and the evaluation; it is
    listed so that a neighbour within reach is recognised as a blend.
    """
    wavelength_nm: float
    species: str = ""
    use: bool = True

    @property
    def label(self):
        return f"{self.species} {float(self.wavelength_nm)!r}".strip()


WAVELENGTH_PROFILE = Profile("wavelength", {
    "wavelength_nm": Field(getter=lambda line: line.wavelength_nm, unit="nm"),
})
"""Compares a measured line's wavelength with its reference's."""


def lamp_lines(source="hgar"):
    """Read reference lines: a bundled lamp name (``'hgar'``) or a CSV file.

    The CSV has a ``wavelength_nm`` column and optional ``species`` and ``use``
    (yes/no) columns; lines starting with ``#`` are comments.
    """
    path = Path(source)
    if not path.suffix and (LAMPS_DIR / f"{source}.csv").exists():
        path = LAMPS_DIR / f"{source}.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(line for line in stream if not line.lstrip().startswith("#"))
        if "wavelength_nm" not in (rows.fieldnames or ()):
            raise ValueError(f"{path}: expected a wavelength_nm column")
        lines = []
        for row in rows:
            use = (row.get("use") or "yes").strip().lower()
            if use not in ("yes", "no", "true", "false", "1", "0"):
                raise ValueError(f"{path}: use must be yes or no, not {row['use']!r}")
            lines.append(EmissionLine(float(row["wavelength_nm"]), (row.get("species") or "").strip(),
                                      use in ("yes", "true", "1")))
    lines.sort(key=lambda line: line.wavelength_nm)
    if len({line.wavelength_nm for line in lines}) != len(lines):
        raise ValueError(f"{path}: reference wavelengths must be unique")
    return Collection(lines, label=path.stem, metadata={"source": str(path)})


@dataclass(frozen=True)
class MeasuredLine:
    """One reference line's outcome: its measured centre, or why it has none.

    ``status`` is ``'measured'``, ``'blend'``, ``'not found'`` or ``'failed'``.
    ``pixel`` is the measured centre (zero-based sample index);
    ``wavelength_nm`` is that centre through the calibration being evaluated,
    and ``holdout_nm`` its error when the line is left out of the fit.
    ``cross_check_nm`` is the centroid minus the fitted mean, and ``crowded``
    flags a difference beyond the evaluation's limit.
    """
    line: EmissionLine
    status: str
    reason: str = ""
    peak_index: int | None = None
    peak_pixel: float | None = None
    fwhm_px: float | None = None
    window: tuple[int, int] | None = None
    pixel: float | None = None
    pixel_uncertainty: float | None = None
    wavelength_nm: float | None = None
    holdout_nm: float | None = None
    cross_check_nm: float | None = None
    crowded: bool = False

    @property
    def label(self):
        return self.line.label

    @property
    def reference_nm(self):
        return self.line.wavelength_nm

    @property
    def error_nm(self):
        return None if self.wavelength_nm is None else self.wavelength_nm - self.reference_nm


@dataclass(frozen=True, repr=False)
class WavelengthEvaluation(Result):
    """Measured line wavelengths compared with their references.

    ``comparison`` holds one :class:`~gsmath.Comparison` per measured line.
    The evaluation passes when every measured line is within the tolerance
    and at least ``min_lines`` were measured. Blends and lines not found are
    reported in ``lines`` but not judged.
    """
    calibration: WavelengthCalibration | None
    lines: tuple[MeasuredLine, ...]
    comparison: Collection
    tolerance_nm: float
    method: str
    min_lines: int
    initial: WavelengthCalibration | None = None

    units = {"max_abs_error_nm": "nm", "rms_error_nm": "nm", "max_abs_holdout_nm": "nm",
             "span_nm": "nm"}

    @property
    def measured(self):
        return tuple(line for line in self.lines if line.status == "measured")

    @property
    def passed(self):
        return len(self.measured) >= self.min_lines and all(c.passed for c in self.comparison)

    @property
    def summary(self):
        errors = np.array([line.error_nm for line in self.measured])
        holdouts = np.array([line.holdout_nm for line in self.measured if line.holdout_nm is not None])
        references = [line.reference_nm for line in self.measured]
        return {
            "passed": self.passed,
            "lines_measured": len(errors),
            "lines_failed": sum(not c.passed for c in self.comparison),
            "lines_crowded": sum(line.crowded for line in self.measured),
            "max_abs_error_nm": float(np.max(abs(errors))) if len(errors) else None,
            "rms_error_nm": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            "max_abs_holdout_nm": float(np.max(abs(holdouts))) if len(holdouts) else None,
            "span_nm": [min(references), max(references)] if references else None,
        }

    def rows(self):
        """One record per reference line, for tables and CSV files."""
        result = []
        judged = {c.label: c.passed for c in self.comparison}
        for line in self.lines:
            result.append({
                "line": line.label, "reference_nm": line.reference_nm,
                "measured_nm": line.wavelength_nm, "difference_nm": line.error_nm,
                "holdout_nm": line.holdout_nm, "pixel": line.pixel,
                "pixel_uncertainty": line.pixel_uncertainty, "fwhm_px": line.fwhm_px,
                "cross_check_nm": line.cross_check_nm, "crowded": line.crowded,
                "status": line.status,
                "result": {True: "PASS", False: "FAIL"}.get(judged.get(line.label), line.status),
                "reason": line.reason,
            })
        return result

    def table(self):
        """The line-by-line result and the verdict as plain text."""
        holdouts = any(line.holdout_nm is not None for line in self.lines)

        def number(value, digits=3):
            return "" if value is None else f"{value:.{digits}f}"

        header = ["Line", "Reference nm", "Measured nm", "Difference nm"]
        header += ["Holdout nm"] if holdouts else []
        header += ["Result"]
        body = []
        for row in self.rows():
            cells = [row["line"], number(row["reference_nm"], 4), number(row["measured_nm"]),
                     number(row["difference_nm"])]
            cells += [number(row["holdout_nm"])] if holdouts else []
            if row["status"] != "measured":
                note = f'{row["status"]}: {row["reason"]}'
            elif row["crowded"]:
                note = f'{row["result"]}  crowded: centroid differs by {row["cross_check_nm"]:+.2f} nm'
            else:
                note = row["result"]
            body.append(cells + [note])
        widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header) - 1)]

        def format_row(cells):
            text = [cells[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(cells[1:-1], widths[1:])]
            return "  ".join(text + [cells[-1]])

        summary = self.summary
        verdict = "PASS" if self.passed else "FAIL"
        lines = [format_row(header), format_row(["-" * w for w in widths] + ["------"])]
        lines += [format_row(cells) for cells in body]
        lines.append("")
        if summary["lines_measured"]:
            lines.append(
                f"{verdict}: {summary['lines_measured']} lines measured "
                f"({summary['span_nm'][0]:.1f}-{summary['span_nm'][1]:.1f} nm), "
                f"{summary['lines_failed']} outside +/-{self.tolerance_nm:g} nm; "
                f"largest difference {summary['max_abs_error_nm']:.3f} nm, "
                f"RMS {summary['rms_error_nm']:.3f} nm.")
            if summary["lines_crowded"]:
                lines.append(f"{summary['lines_crowded']} crowded lines: a neighbour may bias their centres.")
        else:
            lines.append(f"{verdict}: no lines measured.")
        if summary["lines_measured"] < self.min_lines:
            lines.append(f"At least {self.min_lines} measured lines are required.")
        return "\n".join(lines)

    def __str__(self):
        return self.table()


def line_center(y, window, *, method="fit", peak_pixel=None, fwhm_px=None, shape=None):
    """Measure one line's centre in pixels within ``window`` (inclusive bounds).

    Returns ``(center, uncertainty)``; the uncertainty is the fit's standard
    error (``None`` for other methods). For ``'fit'``, ``shape`` optionally
    fixes the line's ``(sigma_px, asymmetry)``. Raises ValueError when the
    samples cannot give a centre.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "peak":
        if peak_pixel is None:
            raise ValueError("the peak method needs the detected peak pixel")
        return float(peak_pixel), None
    y = np.asarray(y, dtype=float)
    x = np.arange(window[0], window[1] + 1)
    if len(x) < 7 or x[0] < 0 or x[-1] >= len(y) or not np.all(np.isfinite(y[x])):
        raise ValueError("fewer than 7 valid samples around the line")
    if method == "centroid":
        return _centroid(x, y[x]), None
    fit = _fit_shape(x, y[x], len(y), x[np.argmax(y[x])] if peak_pixel is None else peak_pixel,
                     2.0 if fwhm_px is None else fwhm_px, shape)
    return fit["center"], fit["uncertainty"]


def _centroid(x, values):
    """Signal-weighted mean pixel above a line through the window's end pairs."""
    ends = np.r_[0, 1, -2, -1]
    background = np.polyval(np.polyfit(x[ends], values[ends], 1), x)
    signal = values - background
    area = signal.sum()
    if not area > 0:
        raise ValueError("no signal above the background")
    return float(np.dot(x, signal) / area)


def _fit_shape(x, values, n_pixels, peak_pixel, fwhm_px, shape=None):
    """Fit a pixel-integrated split Gaussian on a linear background.

    Free parameters are the mean, area and background, plus the width and
    asymmetry unless ``shape`` fixes them as ``(sigma_px, asymmetry)``.
    """
    offset, scale = values.min(), np.ptp(values)
    if not scale > 0:
        raise ValueError("no signal in the window")
    signal = (values - offset) / scale
    z = (x - x.mean()) / max(1.0, np.ptp(x) / 2)

    def unpack(p):
        if shape is None:
            return p
        return p[0], np.log(shape[0]), shape[1], *p[1:]

    def model(p):
        mean, log_sigma, asymmetry, log_area, b0, b1 = unpack(p)
        lsf = LineSpreadFunction((np.exp(log_sigma),) * 3, asymmetry, 0.0)
        return np.exp(log_area) * lsf.profiles(x, [mean], n_pixels)[:, 0] + b0 + b1 * z

    sigma = max(0.5, fwhm_px / 2.355)
    lower = [x[0], np.log(0.25), -0.9, np.log(1e-3), -1.0, -1.0]
    upper = [x[-1], np.log(len(x)), 0.9, np.log(10 * len(x)), 1.0, 1.0]
    start = [np.clip(peak_pixel, x[0] + 0.1, x[-1] - 0.1), np.log(sigma), 0.0,
             np.log(max(signal.sum(), 1e-2)), 0.0, 0.0]
    if shape is not None:
        lower, upper, start = ([v[0], *v[3:]] for v in (lower, upper, start))
    fit = least_squares(lambda p: model(p) - signal, start, bounds=(lower, upper),
                        x_scale="jac", max_nfev=500)
    if not fit.success or fit.active_mask[0] or (shape is None and fit.active_mask[1]):
        raise ValueError("line shape fit did not converge inside the window")
    dof = max(1, len(x) - len(start))
    try:
        covariance = np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun @ fit.fun) / dof
        uncertainty = float(np.sqrt(covariance[0, 0]))
    except np.linalg.LinAlgError:
        uncertainty = None
    mean, log_sigma, asymmetry = unpack(fit.x)[:3]
    return {"center": float(mean), "uncertainty": uncertainty,
            "sigma_px": float(np.exp(log_sigma)), "asymmetry": float(asymmetry)}


def _smooth(pixels, values, n_pixels):
    """A quadratic (fewer lines: lower order) in pixel, refitted without 3-MAD outliers."""
    pixels, values = np.asarray(pixels), np.asarray(values)
    degree = min(2, len(pixels) - 3) if len(pixels) > 3 else 0
    keep = np.ones(len(pixels), dtype=bool)
    for _ in range(2):
        poly = Polynomial.fit(pixels[keep], values[keep], degree, domain=(0, n_pixels - 1))
        residual = values - poly(pixels)
        spread = 1.4826 * np.median(abs(residual[keep] - np.median(residual[keep])))
        keep = abs(residual) <= max(3 * spread, 1e-12)
        if keep.sum() <= degree + 1:
            break
    return poly


def _pixel_axis(calibration, spectrum, n_pixels):
    """Wavelength (nm) at every pixel for a calibration, coefficients or the scan's own axis."""
    pixels = np.arange(n_pixels)
    if calibration is None:
        axis = np.asarray(getattr(spectrum, "wavelength", ()), dtype=float)
        if axis.shape != (n_pixels,):
            raise ValueError("the spectrum has no wavelength axis to evaluate; supply a calibration")
    elif isinstance(calibration, WavelengthCalibration):
        axis = calibration.wavelength(pixels)
    else:
        axis = np.polyval(_vector(calibration, "coefficients", 2), pixels)
    delta = np.diff(axis)
    if not np.all(np.isfinite(axis)) or not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError("the wavelength axis must be finite and strictly monotonic")
    return axis


def measure_lines(spectrum, calibration=None, lines="hgar", *, method="fit", search_nm=2.0,
                  blend_fwhm=1.5, crowding_nm=0.1, peaks=None, detection=None):
    """Pair detected peaks with reference lines and measure their centres.

    ``calibration`` (a :class:`WavelengthCalibration`, descending polynomial
    coefficients, or ``None`` for the spectrum's own wavelength axis) only
    predicts where each line is: a peak pairs with a line within
    ``search_nm``. Each line's window spans about two widths either side of
    its peak, stopping halfway to a neighbouring peak. A line whose centroid
    and fitted mean differ by more than ``crowding_nm`` is flagged crowded.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    _positive(search_nm, "search_nm")
    _positive(blend_fwhm, "blend_fwhm")
    _positive(crowding_nm, "crowding_nm")
    y = _vector(getattr(spectrum, "values", spectrum), "intensity", 12)
    if not isinstance(lines, Collection):
        lines = lamp_lines(lines)
    axis = _pixel_axis(calibration, spectrum, len(y))
    dispersion = np.abs(np.gradient(axis))
    peaks = detect_calibration_peaks(y, **(detection or {})) if peaks is None else tuple(peaks)
    pixels = np.array([peak.pixel for peak in peaks])
    catalog = np.array([line.wavelength_nm for line in lines])
    used = [line for line in lines if line.use]
    found = {}
    if len(peaks) and used:
        predicted = np.interp(pixels, np.arange(len(y)), axis)
        rows, cols = _assign(predicted, np.array([line.wavelength_nm for line in used]), search_nm)
        found = {int(col): int(row) for row, col in zip(rows, cols)}
    result = []
    for j, line in enumerate(used):
        if j not in found:
            result.append(MeasuredLine(line, "not found", f"no peak within {search_nm:g} nm"))
            continue
        index = found[j]
        peak = peaks[index]
        fwhm_nm = peak.width * float(np.interp(peak.pixel, np.arange(len(y)), dispersion))
        others = catalog[catalog != line.wavelength_nm]
        nearest = others[np.argmin(abs(others - line.wavelength_nm))] if len(others) else None
        known = dict(line=line, peak_index=index, peak_pixel=peak.pixel, fwhm_px=peak.width)
        if nearest is not None and abs(nearest - line.wavelength_nm) < blend_fwhm * fwhm_nm:
            neighbour = next(other for other in lines if other.wavelength_nm == nearest)
            result.append(MeasuredLine(status="blend", reason=(
                f"unresolved from {neighbour.label} ({abs(nearest - line.wavelength_nm):.2f} nm "
                f"apart, FWHM {fwhm_nm:.2f} nm)"), **known))
            continue
        reach = max(4.0, 2.0 * peak.width)
        left, right = peak.pixel - reach, peak.pixel + reach
        neighbours = pixels[pixels != peak.pixel]
        if np.any(neighbours < peak.pixel):
            left = max(left, (peak.pixel + neighbours[neighbours < peak.pixel].max()) / 2)
        if np.any(neighbours > peak.pixel):
            right = min(right, (peak.pixel + neighbours[neighbours > peak.pixel].min()) / 2)
        window = (max(0, int(np.ceil(left))), min(len(y) - 1, int(np.floor(right))))
        result.append(MeasuredLine(status="measured", window=window, **known))
    result = _measure_centers(y, result, method, dispersion, crowding_nm)
    # A blend's single peak pairs with one of its lines; the others are part
    # of the same blend, not missing.
    blends = {line.line.wavelength_nm: line for line in result if line.status == "blend"}
    for i, line in enumerate(result):
        if line.status != "not found":
            continue
        partner = next((blend for w, blend in blends.items()
                        if abs(w - line.reference_nm) < blend_fwhm * blend.fwhm_px
                        * float(np.interp(blend.peak_pixel, np.arange(len(y)), dispersion))), None)
        if partner is not None:
            result[i] = replace(line, status="blend", reason=f"unresolved from {partner.label}")
    return tuple(result)


def _measure_centers(y, lines, method, dispersion, crowding_nm):
    """Measure the centre of every line marked measured, in its window.

    The line shape is fitted freely first; its width and asymmetry are then
    smoothed across the detector, and each mean is refitted with that shape
    fixed. The instrument's line shape varies slowly with pixel, and a free
    asymmetry trades off against the centre. The centroid in the same window
    cross-checks the mean, whichever method gives the reported centre.
    """
    lines = list(lines)
    pending = [i for i, line in enumerate(lines) if line.status == "measured"]
    free = {}
    for i in pending:
        line = lines[i]
        x = np.arange(line.window[0], line.window[1] + 1)
        if len(x) >= 7:
            try:
                free[i] = _fit_shape(x, y[x], len(y), line.peak_pixel, line.fwhm_px)
            except ValueError:
                pass
    shapes = {}
    if len(free) >= 3:
        pixels = [free[i]["center"] for i in free]
        sigma = _smooth(pixels, [np.log(free[i]["sigma_px"]) for i in free], len(y))
        asymmetry = _smooth(pixels, [free[i]["asymmetry"] for i in free], len(y))
        shapes = {i: (float(np.exp(sigma(lines[i].peak_pixel))),
                      float(np.clip(asymmetry(lines[i].peak_pixel), -0.9, 0.9))) for i in pending}
    for i in pending:
        line = lines[i]
        centers, errors = {}, {}
        for estimator in ("fit", "centroid"):
            try:
                centers[estimator] = line_center(y, line.window, method=estimator,
                                                 peak_pixel=line.peak_pixel, fwhm_px=line.fwhm_px,
                                                 shape=shapes.get(i))
            except ValueError as exc:
                errors[estimator] = str(exc)
        if method == "peak":
            centers["peak"] = (float(line.peak_pixel), None)
        if method not in centers:
            lines[i] = replace(line, status="failed", reason=errors[method])
            continue
        cross_check = None
        if "fit" in centers and "centroid" in centers:
            scale = float(np.interp(centers["fit"][0], np.arange(len(y)), dispersion))
            cross_check = float((centers["centroid"][0] - centers["fit"][0]) * scale)
        center, uncertainty = centers[method]
        lines[i] = replace(line, pixel=center, pixel_uncertainty=uncertainty,
                           cross_check_nm=cross_check,
                           crowded=cross_check is None or abs(cross_check) > crowding_nm)
    return lines


def _fit_polynomial(lines, degree, n_pixels):
    """Descending coefficients of a polynomial that is monotonic over the detector.

    Unweighted: the centres' statistical uncertainties are smaller than the
    polynomial's misfit, so weighting by them lets a few lines pin the curve.
    """
    poly = Polynomial.fit([line.pixel for line in lines], [line.reference_nm for line in lines],
                          degree, domain=(0, n_pixels - 1)).convert()
    coefficients = np.pad(poly.coef, (0, degree + 1 - len(poly.coef)))[::-1]
    delta = np.diff(np.polyval(coefficients, np.arange(n_pixels)))
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError(f"the degree {degree} fit is not monotonic over the detector")
    return coefficients


def _calibration(coefficients, measured, peaks, n_pixels):
    matches = tuple(LineMatch(line.peak_index, line.pixel, line.reference_nm,
                              float(np.polyval(coefficients, line.pixel)),
                              float(np.polyval(coefficients, line.pixel) - line.reference_nm))
                    for line in measured)
    coefficients = np.asarray(coefficients, dtype=float).copy()
    coefficients.setflags(write=False)
    return WavelengthCalibration(coefficients, matches, tuple(peaks), (0, n_pixels - 1))


def _evaluation(lines, axis_or_calibration, *, tolerance_nm, method, min_lines, n_pixels,
                initial=None):
    """Map measured centres to nm and compare each with its reference line."""
    if isinstance(axis_or_calibration, WavelengthCalibration):
        to_nm = axis_or_calibration.wavelength
    else:
        to_nm = lambda pixel: np.interp(pixel, np.arange(n_pixels), axis_or_calibration)
    lines = tuple(replace(line, wavelength_nm=float(to_nm(line.pixel)))
                  if line.status == "measured" else line for line in lines)
    measured = Collection([line for line in lines if line.status == "measured"], label="measured")
    reference = Collection([line.line for line in measured], label="reference")
    profile = WAVELENGTH_PROFILE.with_limits({"wavelength_nm": {"tolerance": tolerance_nm}})
    comparison = compare(measured, reference, profile=profile)
    calibration = axis_or_calibration if isinstance(axis_or_calibration, WavelengthCalibration) else None
    return WavelengthEvaluation(calibration, lines, comparison, float(tolerance_nm), method,
                                int(min_lines), initial)


def evaluate_wavelength(spectrum, calibration=None, lines="hgar", *, tolerance_nm=0.5,
                        method="fit", search_nm=2.0, blend_fwhm=1.5, crowding_nm=0.1,
                        min_lines=5, detection=None):
    """Evaluate an existing wavelength calibration against a lamp scan.

    ``calibration`` is a :class:`WavelengthCalibration`, descending polynomial
    coefficients, or ``None`` to evaluate the spectrum's own wavelength axis.
    Every measured line must be within ``tolerance_nm`` of its reference, and
    at least ``min_lines`` must be measured, for the evaluation to pass.
    """
    _positive(tolerance_nm, "tolerance_nm")
    y = _vector(getattr(spectrum, "values", spectrum), "intensity", 12)
    measured = measure_lines(spectrum, calibration, lines, method=method, search_nm=search_nm,
                             blend_fwhm=blend_fwhm, crowding_nm=crowding_nm, detection=detection)
    if isinstance(calibration, WavelengthCalibration):
        target = calibration
    else:
        target = _pixel_axis(calibration, spectrum, len(y))
    return _evaluation(measured, target, tolerance_nm=tolerance_nm, method=method,
                       min_lines=min_lines, n_pixels=len(y))


def fit_wavelength(spectrum, lines="hgar", *, degree=3, initial=None, tolerance_nm=0.5,
                   method="fit", search_nm=2.0, blend_fwhm=1.5, crowding_nm=0.1, min_lines=None,
                   detection=None, **identify):
    """Fit a pixel-to-wavelength polynomial to measured line centres, and evaluate it.

    Lines are identified by RANSAC (:func:`identify_calibration_lines`, given
    ``identify`` options) unless ``initial`` supplies a calibration, such as
    one already stored in the instrument. The spectrum's sample index is the
    pixel; any wavelength axis it has is ignored. After the first fit, lines
    are paired again through it, so lines the first pass missed are added.
    ``holdout_nm`` is each line's error when the fit omits it. ``min_lines``
    defaults to ``degree + 3``.
    """
    _positive(tolerance_nm, "tolerance_nm")
    if degree not in (1, 2, 3, 4):
        raise ValueError("degree must be 1, 2, 3, or 4")
    y = _vector(getattr(spectrum, "values", spectrum), "intensity", 12)
    n = len(y)
    if not isinstance(lines, Collection):
        lines = lamp_lines(lines)
    peaks = detect_calibration_peaks(y, **(detection or {}))
    if initial is None:
        identify.setdefault("pixel_domain", (0, n - 1))
        initial = identify_calibration_lines(
            peaks, [line.wavelength_nm for line in lines if line.use], degree=degree, **identify)
    guide, previous = initial, None
    for _ in range(3):
        result = measure_lines(y, guide, lines, method=method, search_nm=search_nm,
                               blend_fwhm=blend_fwhm, crowding_nm=crowding_nm, peaks=peaks)
        measured = [line for line in result if line.status == "measured"]
        if len(measured) < degree + 2:
            raise ValueError(f"only {len(measured)} lines measured; a degree {degree} fit "
                             f"needs at least {degree + 2}")
        coefficients = _fit_polynomial(measured, degree, n)
        guide = _calibration(coefficients, measured, peaks, n)
        labels = {line.label for line in measured}
        if labels == previous:
            break
        previous = labels
    holdouts = {}
    if len(measured) > degree + 2:
        for i, line in enumerate(measured):
            rest = measured[:i] + measured[i + 1:]
            try:
                c = _fit_polynomial(rest, degree, n)
            except ValueError:
                continue
            holdouts[line.label] = float(np.polyval(c, line.pixel) - line.reference_nm)
    result = tuple(replace(line, holdout_nm=holdouts.get(line.label)) for line in result)
    return _evaluation(result, guide, tolerance_nm=tolerance_nm, method=method,
                       min_lines=degree + 3 if min_lines is None else min_lines, n_pixels=n,
                       initial=initial)


def plot_evaluation(evaluation, spectrum, path=None):
    """Plot the scan with its reference lines, and each line's difference.

    The top panel shows the signal against wavelength, with measured lines
    marked green (pass) or red (fail) and unjudged lines grey. The bottom
    panel shows each measured line's difference with the tolerance band, and
    its holdout error as an open marker.
    """
    import matplotlib.pyplot as plt

    y = np.asarray(getattr(spectrum, "values", spectrum), dtype=float)
    calibration = evaluation.calibration
    axis = (calibration.wavelength(np.arange(len(y))) if calibration is not None
            else np.asarray(spectrum.wavelength, dtype=float))
    judged = {c.label: c.passed for c in evaluation.comparison}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]}, constrained_layout=True)
    top.plot(axis, y, color="black", lw=0.8)
    for line in evaluation.lines:
        color = {True: "tab:green", False: "tab:red"}.get(judged.get(line.label), "0.6")
        top.axvline(line.reference_nm, color=color, lw=0.8, alpha=0.7)
        top.text(line.reference_nm, 0.98, f"{line.reference_nm:g}", rotation=90, va="top",
                 ha="right", fontsize=7, color=color, transform=top.get_xaxis_transform())
    top.set(ylabel="Signal", title=f"{'PASS' if evaluation.passed else 'FAIL'}: "
            f"{len(evaluation.measured)} lines measured, ±{evaluation.tolerance_nm:g} nm tolerance")
    tolerance = evaluation.tolerance_nm
    bottom.axhspan(-tolerance, tolerance, color="tab:green", alpha=0.1, lw=0)
    bottom.axhline(0, color="0.5", lw=0.8)
    measured = evaluation.measured
    for crowded, marker, name in ((False, "o", "difference"), (True, "^", "difference, crowded")):
        group = [line for line in measured if line.crowded == crowded]
        if group:
            bottom.scatter([line.reference_nm for line in group], [line.error_nm for line in group],
                           c=["tab:green" if judged[line.label] else "tab:red" for line in group],
                           marker=marker, zorder=3, label=name)
    held = [line for line in measured if line.holdout_nm is not None]
    if held:
        bottom.scatter([line.reference_nm for line in held], [line.holdout_nm for line in held],
                       facecolors="none", edgecolors="tab:blue", zorder=3, label="left out of fit")
    bottom.legend(loc="upper right", fontsize=8)
    limit = max([1.5 * tolerance] + [abs(line.error_nm) * 1.1 for line in measured]
                + [abs(line.holdout_nm) * 1.1 for line in held])
    bottom.set(xlabel="Wavelength (nm)", ylabel="Measured − reference (nm)", ylim=(-limit, limit))
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig
