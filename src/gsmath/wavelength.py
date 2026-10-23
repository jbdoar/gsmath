"""Experimental wavelength calibration by RANSAC peak-pattern matching.

Positions are detector sample indices, never the spectrum's existing wavelength
axis. Reference wavelengths and residuals are in nm. Peak amplitudes select
candidates but do not determine line identities.
"""

from dataclasses import dataclass
import numbers

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


# Provisional selection from the NIST Hg and Ar strong-line tables, in air:
# https://physics.nist.gov/PhysRefData/Handbook/Tables/mercurytable2.htm
# https://physics.nist.gov/PhysRefData/Handbook/Tables/argontable2.htm
# It is not a resolution-independent atlas: close doublets may be unresolved,
# so applications should supply a reviewed, resolution-appropriate list.
HGAR_LINES_NM = (
    365.015, 404.656, 435.833, 546.074, 576.960, 579.066,
    696.543, 706.722, 714.704, 727.294, 738.398, 750.387, 751.465,
    763.511, 772.376, 772.421, 794.818, 800.616, 801.479, 810.369,
    811.531, 826.452, 840.821, 842.465, 852.144, 866.794, 912.297,
    965.7786,
)


@dataclass(frozen=True)
class CalibrationPeak:
    """Candidate with a parabolic subpixel center and width in samples."""

    pixel: float
    height: float
    prominence: float
    width: float


@dataclass(frozen=True)
class LineMatch:
    """A unique peak/reference correspondence; residual is predicted - reference."""

    peak_index: int
    pixel: float
    wavelength_nm: float
    predicted_nm: float
    residual_nm: float


@dataclass(frozen=True)
class WavelengthCalibration:
    """Reviewed candidate solution, with descending-order polynomial coefficients.

    A small residual is not proof of correct identification. Inspect matches,
    their pixel coverage, and unresolved blends before using a calibration.
    ``pixel_domain`` is the interval checked for monotonicity, not a guarantee
    of accuracy outside the matched lines.
    """

    coefficients: np.ndarray
    matches: tuple[LineMatch, ...]
    peaks: tuple[CalibrationPeak, ...]
    pixel_domain: tuple[float, float]
    fit_diagnostics: dict | None = None

    def wavelength(self, pixel):
        """Evaluate pixel -> nm (including extrapolation if requested)."""
        return np.polyval(self.coefficients, pixel)

    def apply(self, spectrum):
        """Return a spectrum on the fitted nm axis, without interpolation.

        Requires the same full, zero-based sample layout as the lamp scan.
        A decreasing detector axis reverses both wavelengths and values so
        downstream spectrum operations receive increasing wavelengths. Equal
        length cannot detect a different binning or detector configuration;
        callers must keep the acquisition layout consistent.
        """
        values = np.asarray(spectrum.values)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("wavelength calibration requires a 1-D spectrum")
        if tuple(self.pixel_domain) != (0, values.size - 1):
            raise ValueError("spectrum must use the same full pixel domain as the lamp scan")
        wavelength = self.wavelength(np.arange(values.size))
        if not np.all(np.isfinite(wavelength)):
            raise ValueError("calibrated wavelengths must be finite")
        delta = np.diff(wavelength)
        if np.all(delta < 0):
            wavelength, values = wavelength[::-1], values[::-1]
        elif not np.all(delta > 0):
            raise ValueError("calibrated wavelengths must be strictly monotonic")
        metadata = dict(spectrum.metadata)
        metadata["wavelength_calibration"] = self.report
        return spectrum.copy(wavelength=wavelength.copy(), values=values.copy(),
                             wavelength_unit="nm", metadata=metadata,
                             label=f"{spectrum.label} [wavelength calibrated]")

    @property
    def report(self):
        """JSON-compatible fit diagnostics and provenance for processed data."""
        from dataclasses import asdict
        report = {"coefficients": self.coefficients.tolist(),
                "coefficient_order": "descending", "wavelength_unit": "nm",
                "pixel_domain": list(self.pixel_domain),
                "matches": [asdict(m) for m in self.matches],
                "rms_nm": self.rms_nm, "max_error_nm": self.max_error_nm,
                "matched_pixel_fraction": self.matched_pixel_fraction}
        if self.fit_diagnostics is not None:
            report["joint_fit"] = self.fit_diagnostics
        return report

    def step(self, *, optional=False):
        """Use this fitted solution in a reusable measurement Calibration."""
        from .calibration import WavelengthAxis
        return WavelengthAxis(self, optional=optional)

    to_step = step  # its original name

    @property
    def rms_nm(self):
        return float(np.sqrt(np.mean([m.residual_nm**2 for m in self.matches])))

    @property
    def max_error_nm(self):
        return max(abs(m.residual_nm) for m in self.matches)

    @property
    def unmatched_peak_indices(self):
        used = {m.peak_index for m in self.matches}
        return tuple(i for i in range(len(self.peaks)) if i not in used)

    @property
    def matched_pixel_fraction(self):
        pixels = [m.pixel for m in self.matches]
        return float(np.ptp(pixels) / np.ptp(self.pixel_domain))


def _vector(value, name, minimum=1):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size < minimum or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 1-D array with at least {minimum} entries")
    return result


def _positive(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def detect_calibration_peaks(intensity, *, prominence_sigma=5.0,
                             prominence=None, min_distance_px=3, min_width_px=1.0):
    """Detect emission peaks using first-difference MAD noise and local refinement.

    ``prominence`` overrides the noise threshold in intensity units. Width and
    distance are in samples. A three-point parabola refines isolated maxima;
    flat tops retain their midpoint. NaNs/infinities are rejected so missing
    samples cannot silently shift pixel coordinates.
    """
    y = _vector(intensity, "intensity", 3)
    _positive(prominence_sigma, "prominence_sigma")
    _positive(min_distance_px, "min_distance_px")
    _positive(min_width_px, "min_width_px")
    if min_distance_px < 1:
        raise ValueError("min_distance_px must be at least 1")
    if prominence is None:
        dy = np.diff(y)
        sigma = 1.4826 * np.median(np.abs(dy - np.median(dy))) / np.sqrt(2)
        prominence = max(prominence_sigma * sigma, np.finfo(float).eps * max(1, np.max(np.abs(y))))
    _positive(prominence, "prominence")
    indices, props = find_peaks(y, prominence=prominence, distance=min_distance_px,
                               width=min_width_px, plateau_size=(None, None))
    peaks = []
    for j, i in enumerate(indices):
        pixel = (props["left_edges"][j] + props["right_edges"][j]) / 2
        if props["plateau_sizes"][j] == 1:
            curvature = y[i - 1] - 2 * y[i] + y[i + 1]
            if curvature < 0:
                pixel = i + np.clip(0.5 * (y[i - 1] - y[i + 1]) / curvature, -0.5, 0.5)
        peaks.append(CalibrationPeak(float(pixel), float(y[i]),
                                     float(props["prominences"][j]), float(props["widths"][j])))
    return tuple(peaks)


def _assign(predicted, reference, tolerance):
    errors = np.abs(predicted[:, None] - reference)
    # Dummy columns let every peak remain unmatched. A missed match costs more
    # than ALL valid residuals together: maximize count first, minimize error next.
    penalty = len(predicted) + 1.0
    cost = np.where(errors <= tolerance, errors / tolerance, 2 * penalty)
    cost = np.concatenate((cost, np.full((len(predicted), len(predicted)), penalty)), axis=1)
    rows, cols = linear_sum_assignment(cost)
    valid = cols < len(reference)
    return rows[valid], cols[valid]


def identify_calibration_lines(peaks, reference_lines=HGAR_LINES_NM, *, degree=4,
                               ransac_degree=None,
                               tolerance_nm=1.0, iterations=20_000, random_seed=1,
                               min_matches=None, pixel_domain=None,
                               wavelength_bounds_nm=None, dispersion_bounds=None,
                               direction="either"):
    """Find a polynomial by randomly sampling ordered peak/line correspondences.

    Defaults to fourth order; supports degrees 1..4. Every hypothesis is scored by unique match count,
    then RMS residual, and promising hypotheses are refitted to their inliers.
    At least ``degree + 2`` matches are required (default: max(5, degree + 2)).
    Failure raises RuntimeError; a finite random search can miss a valid solution.
    ``ransac_degree`` defaults to min(2, degree). Identification uses that
    lower order, then the final inlier fit uses exactly ``degree``; failure
    of the final fit raises rather than silently returning a lower order.

    Optional wavelength bounds constrain the full pixel domain; dispersion
    bounds constrain absolute nm/sample throughout that domain. ``direction``
    is 'increasing', 'decreasing', or 'either'. No existing wavelength axis is
    used implicitly. Use explicit bounds to reject physically implausible fits.
    """
    if not isinstance(degree, numbers.Integral) or isinstance(degree, bool) or degree not in (1, 2, 3, 4):
        raise ValueError("degree must be 1, 2, 3, or 4")
    seed_degree = min(2, degree) if ransac_degree is None else ransac_degree
    if (not isinstance(seed_degree, numbers.Integral) or isinstance(seed_degree, bool)
            or not 1 <= seed_degree <= degree):
        raise ValueError("ransac_degree must be an integer between 1 and degree")
    if not isinstance(iterations, numbers.Integral) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    _positive(tolerance_nm, "tolerance_nm")
    if direction not in ("increasing", "decreasing", "either"):
        raise ValueError("direction must be increasing, decreasing, or either")
    required = max(5, degree + 2) if min_matches is None else min_matches
    if not isinstance(required, numbers.Integral) or isinstance(required, bool) or required < degree + 2:
        raise ValueError("min_matches must be an integer at least degree + 2")
    peaks = tuple(peaks)
    pixels = _vector([p.pixel for p in peaks], "peak pixels", required)
    if np.unique(pixels).size != pixels.size:
        raise ValueError("peak pixels must be unique")
    reference = np.sort(_vector(reference_lines, "reference_lines", required))
    if np.any(np.diff(reference) <= 0):
        raise ValueError("reference_lines must be unique")

    def bounds(value, name):
        pair = _vector(value, name, 2)
        if pair.size != 2 or pair[0] >= pair[1]:
            raise ValueError(f"{name} must contain two increasing bounds")
        return pair

    domain = bounds(pixel_domain if pixel_domain is not None else [pixels.min(), pixels.max()], "pixel_domain")
    if pixels.min() < domain[0] or pixels.max() > domain[1]:
        raise ValueError("pixel_domain must contain all peak pixels")
    wave_bounds = None if wavelength_bounds_nm is None else bounds(wavelength_bounds_nm, "wavelength_bounds_nm")
    disp_bounds = None if dispersion_bounds is None else bounds(dispersion_bounds, "dispersion_bounds")
    if disp_bounds is not None and disp_bounds[0] <= 0:
        raise ValueError("dispersion_bounds must be positive")
    # Work on a scaled coordinate for stable polynomial fitting and root tests.
    center, scale = float(np.mean(domain)), float(np.ptp(domain) / 2)
    x = (pixels - center) / scale

    def admissible(model):
        if not np.all(np.isfinite(model.coef)):
            return False
        derivative = model.deriv()
        roots = derivative.deriv().roots()
        extrema = [r.real for r in roots if abs(r.imag) < 1e-10 and -1 < r.real < 1]
        slopes = derivative(np.array([-1, 1, *extrema])) / scale
        increasing, decreasing = np.all(slopes > 0), np.all(slopes < 0)
        if not ((increasing and direction != "decreasing") or (decreasing and direction != "increasing")):
            return False
        endpoints = model(np.array([-1, 1]))
        if wave_bounds is not None and (endpoints.min() < wave_bounds[0] or endpoints.max() > wave_bounds[1]):
            return False
        return disp_bounds is None or (np.min(abs(slopes)) >= disp_bounds[0] and np.max(abs(slopes)) <= disp_bounds[1])

    def evaluate(model):
        rows, cols = _assign(model(x), reference, tolerance_nm)
        rms = float(np.sqrt(np.mean((model(x[rows]) - reference[cols])**2))) if len(rows) else np.inf
        return (len(rows), -rms), rows, cols

    rng = np.random.default_rng(random_seed)
    best = None
    best_score = (required - 1, -np.inf)
    order = np.argsort(pixels)
    orientations = (1, -1) if direction == "either" else ((1,) if direction == "increasing" else (-1,))
    for trial in range(iterations):
        # Mix affine seeds into higher-order searches: a correct pair is much
        # easier to sample than a correct triple/quintuple. Consensus is still
        # refined at ransac_degree before the requested final polynomial fit.
        hypothesis_degree = 1 if seed_degree > 1 and trial % 2 == 0 else seed_degree
        chosen = order[np.sort(rng.choice(len(x), hypothesis_degree + 1, replace=False))]
        lines = np.sort(rng.choice(reference, hypothesis_degree + 1, replace=False))
        for orientation in orientations:
            model = Polynomial.fit(x[chosen], lines[::orientation], hypothesis_degree).convert()
            if not admissible(model):
                continue
            score, rows, cols = evaluate(model)
            if score <= best_score:
                continue
            # Keep only improvements; refinement cannot replace a valid fit
            # with a nonmonotonic fit or one with fewer consistent matches.
            for _ in range(10):
                refined = Polynomial.fit(x[rows], reference[cols], seed_degree).convert()
                if not admissible(refined):
                    break
                new_score, new_rows, new_cols = evaluate(refined)
                if new_score <= score:
                    break
                model, score, rows, cols = refined, new_score, new_rows, new_cols
            best, best_score = model, score
    if best is None:
        raise RuntimeError("No calibration reached min_matches within the supplied constraints; inspect peaks, references, or increase iterations")
    if degree != seed_degree:
        for _ in range(10):
            _, rows, cols = evaluate(best)
            refined = Polynomial.fit(x[rows], reference[cols], degree).convert()
            score, new_rows, new_cols = evaluate(refined)
            if not admissible(refined) or score[0] < required:
                raise RuntimeError("Final polynomial failed the constraints; inspect identifications or instrument bounds")
            best = refined
            if np.array_equal(rows, new_rows) and np.array_equal(cols, new_cols):
                break
    _, rows, cols = evaluate(best)
    matches = tuple(LineMatch(int(i), float(pixels[i]), float(reference[j]),
                              float(best(x[i])), float(best(x[i]) - reference[j]))
                    for i, j in zip(rows, cols))
    coefficients = best(Polynomial([-center / scale, 1 / scale])).coef[::-1].copy()
    coefficients.setflags(write=False)
    return WavelengthCalibration(coefficients, matches, peaks, tuple(map(float, domain)))


def calibrate_wavelength(spectrum, reference_lines=HGAR_LINES_NM, *,
                         prominence_sigma=5.0, prominence=None,
                         min_distance_px=3, min_width_px=1.0, **fit_options):
    """Detect and identify lines from a Spectrum or 1-D intensity array.

    Uses zero-based sample indices and checks the entire sampled detector
    domain. Returns a candidate calibration without modifying input data.
    Trimmed/rebinned scans therefore use their own sample coordinates.
    """
    y = _vector(getattr(spectrum, "values", spectrum), "intensity", 3)
    peaks = detect_calibration_peaks(y, prominence_sigma=prominence_sigma,
                                    prominence=prominence, min_distance_px=min_distance_px,
                                    min_width_px=min_width_px)
    fit_options.setdefault("pixel_domain", (0, len(y) - 1))
    return identify_calibration_lines(peaks, reference_lines, **fit_options)
