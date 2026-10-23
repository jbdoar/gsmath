"""Experimental joint lamp-profile fitting and reusable factory characterization.

The factory fit shares relative line areas between narrow and working-slit
scans, with independent gains/backgrounds per region, wavelength polynomials,
and line shapes. Field fitting uses the stored working-slit solution and shape
as priors; a fresh narrow scan is not needed. Fits are candidates for review,
not certifications of absolute accuracy.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial, Legendre
from scipy.optimize import least_squares
from scipy.special import ndtr
from scipy.sparse import lil_matrix

from .wavelength import (HGAR_LINES_NM, WavelengthCalibration, LineMatch,
                         calibrate_wavelength, detect_calibration_peaks,
                         _assign, _vector, _positive)


@dataclass(frozen=True)
class LineSpreadFunction:
    """Mean-centered split Gaussian convolved with a rectangular slit.

    Sigma is log-interpolated between detector start, midpoint, and end, in
    pixels. ``asymmetry`` sets sigma_right/left = exp(+/-asymmetry)*sigma.
    The distribution's mean (not its maximum) defines the line coordinate.
    ``slit_width_px`` is a fitted effective width, not the mechanical slit size.
    """
    sigma_px: tuple[float, float, float] = (1., 1., 1.)
    asymmetry: float = 0.
    slit_width_px: float = 0.5

    def __post_init__(self):
        sigma = _vector(self.sigma_px, 'sigma_px', 3)
        if len(sigma) != 3 or np.any(sigma <= 0):
            raise ValueError('sigma_px must contain three positive widths')
        object.__setattr__(self, 'sigma_px', tuple(map(float, sigma)))
        if not np.isfinite(self.asymmetry) or abs(self.asymmetry) > 1:
            raise ValueError('asymmetry must be finite and between -1 and 1')
        if not np.isfinite(self.slit_width_px) or self.slit_width_px < 0:
            raise ValueError('slit_width_px must be finite and nonnegative')

    def profiles(self, pixels, centers, n_pixels):
        """Return pixel-integrated unit-area profiles, one column per line."""
        centers = np.asarray(centers)
        sigma = np.exp(np.interp(centers, [0, (n_pixels-1)/2, n_pixels-1],
                                np.log(self.sigma_px)))
        left, right = sigma*np.exp(-self.asymmetry), sigma*np.exp(self.asymmetry)
        mode = centers - np.sqrt(2/np.pi)*(right-left)

        def cdf(z):
            return np.where(z < 0, 2*left/(left+right)*ndtr(z/left),
                            (left + 2*right*(ndtr(z/right)-0.5))/(left+right))

        # Gauss-Legendre quadrature over the slit; CDF differences integrate
        # over each detector pixel, including undersampled narrow profiles.
        nodes, weights = _SLIT_QUADRATURE
        delta = np.asarray(pixels)[:, None] - mode
        result = np.zeros_like(delta)
        for node, weight in zip(nodes, weights):
            z = delta - node*self.slit_width_px/2
            result += weight/2*(cdf(z+0.5)-cdf(z-0.5))
        return result


_SLIT_QUADRATURE = np.polynomial.legendre.leggauss(12)


@dataclass(frozen=True)
class JointLineFit:
    """Joint profile fit with explicit masks and measured-signal residuals.

    ``models`` and ``residuals`` are in each scan's original signal units;
    unmodeled pixels are NaN. ``normalized_rms`` uses local signal ranges,
    not noise standard deviations. Peak residuals in ``calibrations`` are
    still measured peak maxima versus reference lines, not deblended accuracy.
    """
    calibrations: tuple[WavelengthCalibration, ...]
    line_shapes: tuple[LineSpreadFunction, ...]
    reference_lines: np.ndarray
    relative_areas: np.ndarray
    models: tuple[np.ndarray, ...]
    residuals: tuple[np.ndarray, ...]
    masks: tuple[np.ndarray, ...]
    components: tuple[np.ndarray, ...]
    backgrounds: tuple[np.ndarray, ...]
    normalized_rms: tuple[float, ...]
    warnings: tuple[str, ...]
    nfev: int


@dataclass(frozen=True)
class FactoryLineModel:
    """Per-instrument factory pair, reusable for working-slit field fitting.

    Metadata should identify instrument, slit, binning, probe and lamp. Length
    is checked automatically; matching physical configuration is the caller's
    responsibility. Relative areas are soft field priors, never fixed ratios.
    """
    narrow_calibration: WavelengthCalibration
    working_calibration: WavelengthCalibration
    narrow_shape: LineSpreadFunction
    working_shape: LineSpreadFunction
    reference_lines: np.ndarray
    relative_areas: np.ndarray
    metadata: dict

    def fit(self, spectrum, **options):
        """Fit a field working-slit scan without acquiring a narrow-slit scan."""
        return fit_lamp_lines(spectrum, factory=self, **options)

    def write(self, path):
        """Save the characterization, provenance and diagnostics with a checksum."""
        from dataclasses import asdict
        from .io.native import _value_payload
        data = {"narrow_calibration": _value_payload(self.narrow_calibration),
                "working_calibration": _value_payload(self.working_calibration),
                "narrow_shape": asdict(self.narrow_shape),
                "working_shape": asdict(self.working_shape),
                "reference_lines": self.reference_lines.tolist(),
                "relative_areas": self.relative_areas.tolist(), "metadata": self.metadata}
        checksum = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
        payload = {"format": "gsmath.factory_lines", "version": 1,
                   "checksum": checksum, "model": data}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        return path

    @classmethod
    def read(cls, path):
        from .io.native import _value_from_payload
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('format') != 'gsmath.factory_lines' or payload.get('version') != 1:
            raise ValueError('Unsupported factory line model format/version')
        data = payload['model']
        checksum = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if checksum != payload['checksum']:
            raise ValueError('Factory line model checksum mismatch')
        return cls(_value_from_payload(data['narrow_calibration']),
                   _value_from_payload(data['working_calibration']),
                   LineSpreadFunction(**data['narrow_shape']), LineSpreadFunction(**data['working_shape']),
                   np.array(data['reference_lines']), np.array(data['relative_areas']), data['metadata'])


def _intensities(scan):
    values = _vector(getattr(scan, 'values', scan), 'intensity', 10)
    if np.ptp(values) == 0:
        raise ValueError('intensity must contain spectral features, not a constant signal')
    return values


def _monotonic(poly):
    derivative = poly.deriv()
    roots = derivative.deriv().roots()
    points = [-1., 1., *[r.real for r in roots if abs(r.imag) < 1e-9 and -1 < r.real < 1]]
    slopes = derivative(points)
    return np.all(slopes > 0) or np.all(slopes < 0)


def fit_lamp_lines(spectrum, *, initial=None, factory=None,
                   reference_lines=None, degree=4, **options):
    """Jointly fit a wavelength polynomial, line shape, blends and backgrounds.

    Provide a reviewed initial calibration, or a factory model. Without either,
    lower-order RANSAC supplies the initial solution. Existing spectrum labels
    are never used implicitly. ``degree`` defaults to four in all paths.

    See :func:`fit_factory_lines` for factory-pair fitting. Additional options:
    ``window_px=12``, ``max_shift_nm=2``, ``max_nfev=500``,
    ``shape_prior_sigma=0.2`` and ``area_prior_sigma=1.5`` (field priors in
    log-width/log-area units), ``saturation=None`` (reject samples >= threshold).
    """
    y = _intensities(spectrum)
    if factory is not None:
        if initial is not None or reference_lines is not None:
            raise ValueError('factory supplies initial calibration and reference lines')
        initial = factory.working_calibration
        reference_lines = factory.reference_lines
        shape = factory.working_shape
    else:
        reference_lines = HGAR_LINES_NM if reference_lines is None else reference_lines
        initial = calibrate_wavelength(y, reference_lines, degree=degree) if initial is None else initial
        shape = LineSpreadFunction((1.5, 1.5, 1.5), 0, 1)
    return _fit([y], [initial], [shape], reference_lines, degree=degree,
                factory=factory, **options)


def fit_factory_lines(narrow, working, *, initial=None, working_initial=None,
                      reference_lines=HGAR_LINES_NM, degree=4, metadata=None, **options):
    """Fit a matched factory narrow/working pair with shared relative line areas.

    Returns ``(FactoryLineModel, JointLineFit)``. Per-region throughput and
    backgrounds are independent. Both wavelength polynomials and line shapes
    are fitted simultaneously. The pair must have the same detector sampling,
    lamp and optical path; slit and exposure may differ. A supplied reviewed
    ``initial`` avoids repeating identification. ``working_initial`` permits a
    larger known slit offset; otherwise the narrow solution seeds both scans.
    """
    a, b = _intensities(narrow), _intensities(working)
    if len(a) != len(b):
        raise ValueError('Factory scans must have the same detector sampling')
    initial = calibrate_wavelength(a, reference_lines, degree=degree) if initial is None else initial
    result = _fit([a, b], [initial, initial if working_initial is None else working_initial],
                  [LineSpreadFunction(), LineSpreadFunction((1.2, 1.2, 1.2), 0, 6)],
                  reference_lines, degree=degree, **options)
    model = FactoryLineModel(*result.calibrations, *result.line_shapes,
                             result.reference_lines.copy(), result.relative_areas.copy(),
                             dict(metadata or {}))
    return model, result


def _fit(signals, initials, shapes, reference_lines, *, degree=4, window_px=12,
         max_shift_nm=2., max_nfev=500, factory=None, shape_prior_sigma=0.2,
         area_prior_sigma=1.5, saturation=None, sample_masks=None):
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree not in (1, 2, 3, 4):
        raise ValueError('degree must be 1, 2, 3, or 4')
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise ValueError('max_nfev must be a positive integer')
    for name, value in [('window_px', window_px), ('max_shift_nm', max_shift_nm),
                        ('shape_prior_sigma', shape_prior_sigma), ('area_prior_sigma', area_prior_sigma)]:
        _positive(value, name)
    if saturation is not None:
        _positive(saturation, 'saturation')
    refs = np.sort(_vector(reference_lines, 'reference_lines', degree+2))
    if np.any(np.diff(refs) <= 0):
        raise ValueError('reference_lines must be unique')
    n, ns = len(signals[0]), len(signals)
    if sample_masks is None:
        sample_masks = [np.ones(n, dtype=bool) for _ in signals]
    if len(sample_masks) != ns:
        raise ValueError('sample_masks must contain one boolean mask per scan')
    sample_masks = [np.asarray(mask) for mask in sample_masks]
    if any(mask.dtype != np.dtype(bool) or mask.shape != (n,) for mask in sample_masks):
        raise ValueError('sample_masks must be boolean arrays matching the scans')
    pixel_map = Polynomial([(n-1)/2, (n-1)/2])
    bases = []
    for initial in initials:
        if initial.pixel_domain != (0, n-1):
            raise ValueError('Initial calibration must have the same full pixel domain')
        base = Polynomial(initial.coefficients[::-1])(pixel_map)
        if len(initial.coefficients) > degree+1:
            raise ValueError('Initial polynomial degree exceeds requested final degree')
        if not _monotonic(base):
            raise ValueError('Initial calibration must be monotonic')
        bases.append(base)
    low = max(min(p(-1), p(1)) for p in bases)
    high = min(max(p(-1), p(1)) for p in bases)
    refs = refs[(refs > low) & (refs < high)]
    if len(refs) < degree+2:
        raise ValueError('Too few reference lines within the detector range')
    zgrid = np.linspace(-1, 1, n*4)

    def inverse(poly, lines):
        wave = poly(zgrid)
        if wave[0] < wave[-1]:
            z = np.interp(lines, wave, zgrid)
        else:
            z = np.interp(lines, wave[::-1], zgrid[::-1])
        for _ in range(3):
            z -= (poly(z)-lines)/poly.deriv()(z)
        return pixel_map(z)

    centers = [inverse(p, refs) for p in bases]
    # Merge overlapping windows. Each independent region gets a local linear
    # background and gain; individual line areas are shared across factory scans.
    order = np.argsort(centers[0])
    groups = [[int(order[0])]]
    for j in order[1:]:
        # Account for both scans and rounding window edges to integer pixels.
        # Separate region backgrounds must never describe the same pixel.
        if any(abs(c[j]-c[groups[-1][-1]]) < 2*window_px+2 for c in centers):
            groups[-1].append(int(j))
        else:
            groups.append([int(j)])
    if np.ptp(centers[0]) < (n-1)*0.3:
        raise ValueError('Reference lines cover less than 30% of the detector')
    records = []
    for s, y in enumerate(signals):
        for g, indices in enumerate(groups):
            c = centers[s][indices]
            pixels = np.arange(max(0, int(np.floor(c.min()-window_px))),
                               min(n, int(np.ceil(c.max()+window_px))+1))
            pixels = pixels[sample_masks[s][pixels]]
            if saturation is not None:
                pixels = pixels[y[pixels] < saturation]
            if len(pixels) < 5:
                raise ValueError('Too few unsaturated samples in a fitting region')
            local = y[pixels]
            baseline = float(np.percentile(local, 10))
            scale = max(float(np.ptp(local)), 1.)
            records.append((s, g, np.array(indices), pixels, baseline, scale,
                            (local-baseline)/scale))
    params, lower, upper = [], [], []

    def add(values, lo, hi):
        start = len(params)
        values = np.atleast_1d(values).astype(float)
        params.extend(values)
        lower.extend(np.broadcast_to(lo, values.shape))
        upper.extend(np.broadcast_to(hi, values.shape))
        return np.arange(start, len(params))

    wave_idx, shape_idx = [], []
    for shape in shapes:
        wave_idx.append(add(np.zeros(degree+1), -1., 1.))
        shape_idx.append(add([*np.log(shape.sigma_px), shape.asymmetry, shape.slit_width_px],
                             [np.log(.3)]*3+[-.7, .05], [np.log(10)]*3+[.7, 15.]))
    area0 = np.full(len(refs), 3.)
    for s, g, indices, pixels, baseline, scale, data in records:
        if s == 0:
            for j in indices:
                area0[j] = max(.03, np.interp(centers[0][j], pixels, data)*3)
    if factory is not None:
        area0 = np.interp(refs, factory.reference_lines, factory.relative_areas)
    area_idx = add(np.log(area0), -9, 7)
    nuisance = []
    for s, g, indices, pixels, baseline, scale, data in records:
        bg = add([0., 0.], [-.3, -.3], [.3, .3])
        gain = add([0.], -4, 4) if s > 0 else np.array([], dtype=int)
        nuisance.append((bg, gain))
    params = np.array(params)

    def unpack(p, s):
        # A shared L1 budget guarantees |correction| <= max_shift_nm on
        # [-1, 1], since every Legendre basis is bounded there. Unlike giving
        # each coefficient 1/(degree+1) of the budget, this permits a constant
        # shift to use the whole allowance (essential for peak -> mean centers).
        correction = p[wave_idx[s]]
        correction = correction * max_shift_nm / max(1., np.sum(abs(correction)))
        poly = bases[s]+Legendre(correction).convert(kind=Polynomial)
        q = p[shape_idx[s]]
        shape = LineSpreadFunction(tuple(np.exp(q[:3])), float(q[3]), float(q[4]))
        return poly, shape

    def predictions(p):
        unpacked = [unpack(p, s) for s in range(ns)]
        positions = [inverse(poly, refs) for poly, shape in unpacked]
        values = []
        for record, (bg, gain) in zip(records, nuisance):
            s, g, indices, pixels, baseline, scale, data = record
            profile = unpacked[s][1].profiles(pixels, positions[s][indices], n)
            amplitude = np.exp(p[area_idx[indices]])
            multiplier = float(np.exp(p[gain[0]])) if len(gain) else 1.
            local_x = (pixels-pixels.mean())/max(1., np.ptp(pixels)/2)
            values.append(multiplier*(profile@amplitude)+p[bg[0]]+p[bg[1]]*local_x)
        return values

    def residual(p):
        chunks = [model-record[-1] for model, record in zip(predictions(p), records)]
        if factory is not None:
            q0 = params[shape_idx[0]]
            chunks.append((p[shape_idx[0]]-q0)/np.array([shape_prior_sigma]*4+[1.]))
            for indices in groups:
                delta = p[area_idx[indices]]-np.log(area0[indices])
                chunks.append((delta-delta.mean())/area_prior_sigma)
        return np.concatenate(chunks)

    # Every region depends on its scan's polynomial/shape, its line areas and
    # its own nuisance terms. Sparse numerical derivatives avoid a dense solve.
    total = sum(len(r[-1]) for r in records)
    prior_count = 5+len(refs) if factory is not None else 0
    sparsity = lil_matrix((total+prior_count, len(params)), dtype=int)
    offset = 0
    for record, (bg, gain) in zip(records, nuisance):
        s, g, indices, pixels, baseline, scale, data = record
        columns = np.r_[wave_idx[s], shape_idx[s], area_idx[indices], bg, gain]
        sparsity[offset:offset+len(data), columns] = 1
        offset += len(data)
    if factory is not None:
        sparsity[offset:offset+5, shape_idx[0]] = 1
        offset += 5
        for indices in groups:
            sparsity[offset:offset+len(indices), area_idx[indices]] = 1
            offset += len(indices)
    fitted = least_squares(residual, params, bounds=(lower, upper), jac_sparsity=sparsity.tocsr(),
                           max_nfev=max_nfev, loss='soft_l1', f_scale=.05,
                           ftol=1e-6, xtol=1e-7, gtol=1e-6)
    if not fitted.success:
        raise RuntimeError(f'Joint line fit did not converge: {fitted.message}')
    models = [np.full(n, np.nan) for _ in signals]
    components = [np.zeros((n, len(refs))) for _ in signals]
    masks = [np.zeros(n, dtype=bool) for _ in signals]
    normalized = [[] for _ in signals]
    for prediction, record, (bg, gain) in zip(predictions(fitted.x), records, nuisance):
        s, g, indices, pixels, baseline, scale, data = record
        models[s][pixels] = prediction*scale+baseline
        masks[s][pixels] = True
        normalized[s].extend(prediction-data)
        poly, shape = unpack(fitted.x, s)
        profile = shape.profiles(pixels, inverse(poly, refs[indices]), n)
        multiplier = float(np.exp(fitted.x[gain[0]])) if len(gain) else 1.
        components[s][np.ix_(pixels, indices)] = (profile *
            np.exp(fitted.x[area_idx[indices]]) * multiplier * scale)
    for component, mask in zip(components, masks):
        component[~mask] = np.nan
    backgrounds = tuple(model-component.sum(axis=1) for model, component in zip(models, components))
    rms = tuple(float(np.sqrt(np.mean(np.square(v)))) for v in normalized)
    notes = []
    if not np.all(np.isfinite(fitted.x)) or not all(np.isfinite(v) for v in rms):
        raise RuntimeError('Joint line fit produced nonfinite parameters or residuals')
    if np.any(fitted.active_mask):
        notes.append('One or more fit parameters reached a bound; inspect model mismatch or initial calibration.')
    shift_budget_active = [bool(np.sum(abs(fitted.x[index])) >= 1-1e-5) for index in wave_idx]
    if any(shift_budget_active):
        notes.append('A wavelength correction reached its shared shift budget; inspect the initial calibration.')
    bound_parameters = []
    for s in range(ns):
        for indices, names in [(wave_idx[s], [f'wavelength_correction_{i}' for i in range(degree+1)]),
                               (shape_idx[s], ['sigma_start', 'sigma_mid', 'sigma_end', 'asymmetry', 'slit_width'])]:
            for index, name in zip(indices, names):
                if fitted.active_mask[index]:
                    bound_parameters.append(f'scan_{s}:{name}')
    for j, index in enumerate(area_idx):
        if fitted.active_mask[index]:
            bound_parameters.append(f'line_area:{refs[j]:.6f}')
    for record, (bg, gain) in zip(records, nuisance):
        for index, name in zip(np.r_[bg, gain], ['background', 'background_slope', 'gain']):
            if fitted.active_mask[index]:
                bound_parameters.append(f'scan_{record[0]}:region_{record[1]}:{name}')
    if any(v > .05 for v in rms):
        notes.append('Profile residual RMS exceeds 5% of a local signal range; inspect line shapes and missing lines.')
    if np.ptp(centers[0]) < .7*(n-1):
        notes.append('Reference coverage is below 70%; endpoint extrapolation is weakly constrained.')
    calibrations, fitted_shapes = [], []
    for s, initial in enumerate(initials):
        poly, shape = unpack(fitted.x, s)
        if not _monotonic(poly):
            raise RuntimeError('Joint fit produced a nonmonotonic wavelength polynomial')
        coefficients = poly(Polynomial([-1, 2/(n-1)])).coef[::-1]
        # Truncate only negligible higher terms inherited from a higher-order
        # initial solution; otherwise a requested lower final order is invalid.
        if len(coefficients) > degree+1:
            raise ValueError('Initial polynomial degree exceeds requested final degree')
        coefficients = np.pad(coefficients, (degree+1-len(coefficients), 0))
        # Diagnostics refer to this measured scan, not the factory/narrow scan
        # used to initialize it. Never manufacture zero-residual line centers.
        peaks = detect_calibration_peaks(signals[s])
        peaks = tuple(p for p in peaks if masks[s][int(round(p.pixel))])
        pixels = np.array([p.pixel for p in peaks])
        predicted = np.polyval(coefficients, pixels)
        rows, cols = _assign(predicted, refs, 2.)
        if len(rows) < degree+2:
            raise RuntimeError('Too few observed peak matches support the joint solution')
        matches = tuple(LineMatch(int(i), float(pixels[i]), float(refs[j]),
                                  float(predicted[i]), float(predicted[i]-refs[j]))
                        for i, j in zip(rows, cols))
        diagnostics = {'method': 'pixel_integrated_joint_profiles', 'degree': degree,
                       'normalized_signal_rms': rms[s], 'fitted_reference_lines_nm': refs.tolist(),
                       'peak_match_tolerance_nm': 2.,
                       'warnings': notes, 'nfev': fitted.nfev,
                       'parameters_at_bounds': bound_parameters,
                       'shift_budget_active': shift_budget_active[s],
                       'peak_residual_note': 'Residuals describe observed maxima, including blends; not deblended accuracy.',
                       'line_shape': {'sigma_px': list(shape.sigma_px), 'asymmetry': shape.asymmetry,
                                      'slit_width_px': shape.slit_width_px}}
        coefficients.setflags(write=False)
        calibrations.append(WavelengthCalibration(coefficients, matches, peaks,
                                                  initial.pixel_domain, diagnostics))
        fitted_shapes.append(shape)
    return JointLineFit(tuple(calibrations), tuple(fitted_shapes), refs,
                        np.exp(fitted.x[area_idx]), tuple(models),
                        tuple(y-model for y, model in zip(signals, models)), tuple(masks),
                        tuple(components), backgrounds, rms, tuple(notes), fitted.nfev)
