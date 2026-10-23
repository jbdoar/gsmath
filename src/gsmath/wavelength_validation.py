"""Conditional held-out-line validation of factory wavelength calibration.

Line identities are treated as reviewed inputs. This checks prediction after
withholding spectral regions, not blind identification or traceable accuracy.
Peak-only errors use the peak-maximum coordinate convention; joint errors use
a local translation of the line shape learned WITHOUT the held-out region.
Only test-line position, area and background are fitted during scoring; the
training polynomial and line shape stay fixed.
"""
from dataclasses import asdict, dataclass

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import least_squares

from .linefit import _intensities, _monotonic, fit_factory_lines
from .wavelength import HGAR_LINES_NM, WavelengthCalibration, calibrate_wavelength, _vector, _positive


@dataclass(frozen=True)
class HeldOutLine:
    wavelength_nm: float
    scan: str
    peak_error_nm: float
    joint_error_nm: float
    peak_pixel: float
    fitted_center_pixel: float
    joint_profile_rms: float
    extrapolated: bool
    training_matches: int
    training_signal_rms: float
    training_warnings: tuple[str, ...]


@dataclass(frozen=True)
class WavelengthValidation:
    """Held-out prediction errors with failures retained in the report."""
    records: tuple[HeldOutLine, ...]
    failures: tuple[dict, ...]
    requested_lines: tuple[float, ...]
    degree: int
    holdout_radius_px: float

    def summary(self, *, interpolation_only=False):
        summary = {}
        for scan in ('narrow', 'working'):
            rows = [r for r in self.records if r.scan == scan and
                    (not interpolation_only or not r.extrapolated)]
            summary[scan] = {}
            for method, name in [('peak', 'peak_error_nm'), ('joint', 'joint_error_nm')]:
                errors = np.array([getattr(r, name) for r in rows])
                summary[scan][method] = {
                    'count': len(errors),
                    'bias_nm': float(errors.mean()) if len(errors) else None,
                    'rms_nm': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                    'max_abs_nm': float(np.max(abs(errors))) if len(errors) else None,
                }
        return summary

    def assess(self, *, max_error_nm=0.5, max_rms_nm=None, working_range_nm=None,
               required_lines_nm=None, interpolation_only=False):
        """Assess every resolvable peak against a default 0.5 nm absolute limit.

        The default scope is the full instrument range; ``working_range_nm``
        explicitly restricts it (inclusive endpoints). RMS is descriptive unless
        an additional ``max_rms_nm`` limit is supplied. Slit-specific tolerances
        can override ``max_error_nm``.

        ``required_lines_nm`` declares a reviewed, complete inventory of
        resolvable peaks in the selected scope: either one wavelength sequence
        or a mapping with 'narrow'/'working' sequences for different resolution.
        Without that inventory coverage is unknown, so status cannot be 'pass'.
        Status is 'fail' for a measured violation, otherwise 'incomplete' for
        unknown coverage or unscored/failed required lines, otherwise 'pass'.
        ``passes_prediction_limits`` only describes the tested/required lines;
        it must not be interpreted as full-range acceptance when incomplete.
        Shape warnings and traceability still require separate review.
        """
        _positive(max_error_nm, 'max_error_nm')
        if max_rms_nm is not None:
            _positive(max_rms_nm, 'max_rms_nm')
        bounds = None
        if working_range_nm is not None:
            bounds = _vector(working_range_nm, 'working_range_nm', 2)
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError('working_range_nm must contain two increasing bounds')

        def in_range(line):
            return bounds is None or bounds[0] <= line <= bounds[1]

        def contains(lines, line):
            return bool(np.any(np.isclose(lines, line, rtol=0, atol=1e-8)))

        if isinstance(required_lines_nm, dict) and set(required_lines_nm) - {'narrow', 'working'}:
            raise ValueError('required_lines_nm mapping keys must be narrow or working')
        result = {}
        for scan in ('narrow', 'working'):
            inventory = required_lines_nm.get(scan) if isinstance(required_lines_nm, dict) else required_lines_nm
            if inventory is not None:
                inventory = _vector(inventory, 'required_lines_nm')
                if len(np.unique(inventory)) != len(inventory):
                    raise ValueError('required_lines_nm must be unique')
            expected = [float(w) for w in (self.requested_lines if inventory is None else inventory) if in_range(w)]
            rows = [r for r in self.records if r.scan == scan and in_range(r.wavelength_nm)
                    and contains(expected, r.wavelength_nm) and (not interpolation_only or not r.extrapolated)]
            scored = [r.wavelength_nm for r in rows]
            missing = [w for w in expected if not contains(scored, w)]
            failed = [f for f in self.failures if contains(expected, f['wavelength_nm'])
                      and f.get('scan', scan) == scan]
            result[scan] = {}
            for method, name in [('peak', 'peak_error_nm'), ('joint', 'joint_error_nm')]:
                errors = np.array([getattr(r, name) for r in rows])
                finite = bool(np.all(np.isfinite(errors)))
                rms = float(np.sqrt(np.mean(errors**2))) if len(errors) and finite else None
                violations = [{'wavelength_nm': r.wavelength_nm, 'error_nm': float(e)}
                              for r, e in zip(rows, errors) if np.isfinite(e) and abs(e) > max_error_nm]
                exceeded = bool(violations or (max_rms_nm is not None and rms is not None and rms > max_rms_nm))
                tested_pass = bool(len(rows) and finite and not exceeded and not missing and not failed)
                complete = bool(inventory is not None and len(expected) and not missing and not failed and finite)
                result[scan][method] = {
                    'count': len(rows), 'bias_nm': float(errors.mean()) if len(errors) and finite else None,
                    'rms_nm': rms, 'max_abs_nm': float(np.max(abs(errors))) if len(errors) and finite else None,
                    'max_error_nm': float(max_error_nm), 'max_rms_nm': max_rms_nm,
                    'scope': 'full_instrument' if bounds is None else 'working_range',
                    'working_range_nm': None if bounds is None else bounds.tolist(),
                    'required_lines_nm': expected, 'unscored_lines_nm': missing,
                    'failed_lines': failed, 'violations': violations,
                    'coverage_complete': complete,
                    'passes_prediction_limits': tested_pass,
                    'status': 'fail' if exceeded else ('pass' if tested_pass and complete else 'incomplete'),
                }
        return result

    def to_dict(self):
        return {'degree': self.degree, 'requested_lines_nm': list(self.requested_lines),
                'holdout_radius_px': self.holdout_radius_px,
                'summary': self.summary(), 'interpolation_summary': self.summary(interpolation_only=True),
                'records': [asdict(r) for r in self.records], 'failures': list(self.failures),
                'interpretation': 'Conditional on reviewed identities. Peak and joint methods use maximum '
                    'and mean-centered-profile coordinates respectively. Not a traceable accuracy certification.'}


def _training_calibration(initial, omitted_lines, mask, degree):
    matches = tuple(m for m in initial.matches
                    if not np.any(np.isclose(m.wavelength_nm, omitted_lines, rtol=0, atol=1e-8))
                    and mask[int(round(m.pixel))])
    if len(matches) < degree+2:
        raise ValueError('Too few independent training peak matches after withholding the region')
    x = np.array([m.pixel for m in matches])
    y = np.array([m.wavelength_nm for m in matches])
    domain = initial.pixel_domain
    scaled = Polynomial.fit(x, y, degree, domain=domain)
    # The original coefficients/diagnostics never initialize the fold fit.
    if not _monotonic(Polynomial(scaled.coef)):
        raise ValueError('Training-only polynomial is not monotonic over the detector')
    coefficients = np.pad(scaled.convert().coef, (0, degree+1-len(scaled.coef)))[::-1].copy()
    from dataclasses import replace
    matches = tuple(replace(m, predicted_nm=float(np.polyval(coefficients, m.pixel)),
                            residual_nm=float(np.polyval(coefficients, m.pixel)-m.wavelength_nm)) for m in matches)
    # Retain only training peaks; update diagnostic indices to this new list.
    peaks = tuple(initial.peaks[m.peak_index] for m in matches)
    matches = tuple(replace(m, peak_index=i) for i, m in enumerate(matches))
    return WavelengthCalibration(coefficients, matches, peaks, domain)


def _score_profile(y, peak, shape, radius):
    pixels = np.arange(max(0, int(np.ceil(peak-radius))), min(len(y), int(np.floor(peak+radius))+1))
    if len(pixels) < 7:
        raise ValueError('Held-out feature is too close to a neighbor or detector edge')
    values = y[pixels]
    scale = float(np.ptp(values))
    if scale <= 0:
        raise ValueError('No signal contrast in held-out region')
    signal = (values-values.min())/scale
    x = (pixels-pixels.mean())/max(1., np.ptp(pixels)/2)

    def residual(p):
        profile = shape.profiles(pixels, [p[0]], len(y))[:, 0]
        return np.exp(p[1])*profile+p[2]+p[3]*x-signal

    fit = least_squares(residual, [peak, np.log(max(1, signal.sum())), 0, 0],
        bounds=([peak-radius/2, -8, -.3, -.3], [peak+radius/2, 10, .3, .3]),
        max_nfev=200, loss='linear')
    if not fit.success or np.any(fit.active_mask):
        raise RuntimeError('Held-out position/background fit failed or reached a bound')
    return float(fit.x[0]), float(np.sqrt(np.mean(fit.fun**2)))


def validate_factory_lines(narrow, working, heldout_lines, *, narrow_initial=None,
                           working_initial=None, reference_lines=HGAR_LINES_NM,
                           degree=4, holdout_radius_px=20., evaluation_radius_px=12.,
                           progress=None, **fit_options):
    """Leave one isolated-line region out of both factory scans per fold.

    Removes neighboring reference lines inside the masked region as well.
    Rebuilds fourth-order starting curves from retained reviewed peak matches;
    factory line-shape/area priors are never reused across folds. Sample masks
    exclude held-out pixels even if adjacent training windows overlap them.

    ``narrow_initial``/``working_initial`` supply reviewed peak correspondences;
    if omitted they are identified with RANSAC. This is conditional validation
    of the fit, since the identities may have been established with all lines.
    ``progress`` optionally receives a message after each completed fold.
    Failures are recorded, not silently dropped. Extrapolation beyond retained
    training anchors is flagged separately for each scan.
    """
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree not in (1, 2, 3, 4):
        raise ValueError('degree must be 1, 2, 3, or 4')
    _positive(holdout_radius_px, 'holdout_radius_px')
    _positive(evaluation_radius_px, 'evaluation_radius_px')
    if holdout_radius_px < evaluation_radius_px:
        raise ValueError('holdout_radius_px must cover the evaluation region')
    if 'sample_masks' in fit_options:
        raise ValueError('Validation manages sample_masks itself')
    signals = [_intensities(narrow), _intensities(working)]
    if len(signals[0]) != len(signals[1]):
        raise ValueError('Factory scans must have the same detector sampling')
    requested = _vector(heldout_lines, 'heldout_lines')
    refs = np.sort(_vector(reference_lines, 'reference_lines', degree+2))
    if len(np.unique(requested)) != len(requested) or len(np.unique(refs)) != len(refs):
        raise ValueError('Reference and held-out lines must be unique')
    if any(not np.any(np.isclose(refs, line, rtol=0, atol=1e-8)) for line in requested):
        raise ValueError('Every held-out line must occur in reference_lines')
    initials = [narrow_initial, working_initial]
    for s in range(2):
        if initials[s] is None:
            initials[s] = calibrate_wavelength(signals[s], refs, degree=degree)
        if initials[s].pixel_domain != (0, len(signals[s])-1):
            raise ValueError('Initial calibration must match the full detector domain')
    records, failures = [], []
    grid = np.arange(len(signals[0]))
    for line in requested:
        try:
            test_matches = []
            for initial in initials:
                found = [m for m in initial.matches if np.isclose(m.wavelength_nm, line, rtol=0, atol=1e-8)]
                if len(found) != 1:
                    raise ValueError('Held-out line does not have a unique reviewed match in both scans')
                test_matches.append(found[0])
            masks = [abs(grid-m.pixel) > holdout_radius_px for m in test_matches]
            omitted = {float(line)}
            for initial, mask in zip(initials, masks):
                omitted.update(m.wavelength_nm for m in initial.matches if not mask[int(round(m.pixel))])
                wave = initial.wavelength(grid)
                positions = (np.interp(refs, wave, grid) if wave[0] < wave[-1]
                             else np.interp(refs, wave[::-1], grid[::-1]))
                omitted.update(float(refs[j]) for j in range(len(refs))
                               if not mask[int(round(positions[j]))])
            train_refs = refs[~np.any(np.isclose(refs[:, None], list(omitted), rtol=0, atol=1e-8), axis=1)]
            training = [_training_calibration(i, list(omitted), mask, degree) for i, mask in zip(initials, masks)]
            _, fitted = fit_factory_lines(*signals, initial=training[0], working_initial=training[1],
                reference_lines=train_refs, degree=degree, sample_masks=masks, **fit_options)
            # Assert the scored data really were absent from optimization.
            if any(np.any(fit_mask & ~mask) for fit_mask, mask in zip(fitted.masks, masks)):
                raise RuntimeError('Held-out pixels leaked into the training fit')
            fold = []
            for s, name in enumerate(('narrow', 'working')):
                peak = test_matches[s].pixel
                neighbors = [abs(p.pixel-peak) for p in initials[s].peaks if abs(p.pixel-peak) > 1e-6]
                radius = min(evaluation_radius_px, min(neighbors)/2-.5) if neighbors else evaluation_radius_px
                center, rms = _score_profile(signals[s], peak, fitted.line_shapes[s], radius)
                wavelengths = [m.wavelength_nm for m in training[s].matches]
                fold.append(HeldOutLine(float(line), name,
                    float(training[s].wavelength(peak)-line),
                    float(fitted.calibrations[s].wavelength(center)-line), float(peak), center, rms,
                    bool(line < min(wavelengths) or line > max(wavelengths)), len(wavelengths),
                    fitted.normalized_rms[s], fitted.warnings))
            records.extend(fold)
            if progress is not None:
                progress(f'{line:.3f} nm: peak/joint working errors {fold[1].peak_error_nm:+.4f}/{fold[1].joint_error_nm:+.4f} nm')
        except (ValueError, RuntimeError) as exc:
            failures.append({'wavelength_nm': float(line), 'reason': str(exc)})
            if progress is not None:
                progress(f'{line:.3f} nm: FAILED: {exc}')
    return WavelengthValidation(tuple(records), tuple(failures), tuple(map(float, requested)), degree, holdout_radius_px)
