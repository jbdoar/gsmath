"""Spectral stray light: characterising it from monochromator scans, and
correcting it.

A spectroradiometer's response to a narrow line is mostly in band, around the
line, with a small out-of-band part spread across the detector: stray light.
Scanning a monochromator across the detector's range records that response,
its line spread function (LSF), at each set wavelength. From those scans:

- :func:`characterize_scan` finds each scan's peak centre and width, and
  rejects scans whose peak is missing, off its set wavelength, or too near the
  detector's ends for its band to be measured;
- :func:`build_distribution` turns the accepted scans into the distribution
  matrix D: column j is the out-of-band signal, relative to the in-band sum,
  for a line at the j-th detector wavelength (interpolated between scans, by
  default in distance from the line);
- the correction is ``C = inverse(I + D)``: :meth:`StrayLightDistribution.correction`;
- :func:`evaluate_correction` applies C to scans and reports how much of the
  out-of-band signal it removes, and what it costs (negative artifacts,
  change in band).

:func:`stray_light_calibration` does all of it in one call.
"""

import csv
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline

from .analysis.base import Result
from .spectrum import Spectrum

ONE_COUNT = 1 / (2**16 - 1)
"""One count of a 16-bit detector, relative to full scale: the default noise
floor below which a peak-normalised scan is treated as zero."""


def _grid(wavelength):
    x = np.asarray(wavelength, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength must be finite, strictly increasing, with at least two samples")
    return x


def _matrix(values, wavelength):
    a = np.asarray(values, dtype=float)
    if a.shape != (len(wavelength), len(wavelength)) or not np.all(np.isfinite(a)):
        raise ValueError("matrix must be finite and square with one row/column per wavelength")
    return a


def _write_matrix(matrix, path, *, kind):
    path = Path(path)
    wavelength = _grid(matrix.wavelength)
    values = _matrix(matrix.values, wavelength)
    if path.suffix.lower() == ".gsmath":
        from .io.native import write_straylight_matrix
        return write_straylight_matrix(matrix, path, kind=kind)
    if path.suffix.lower() != ".bin":
        raise ValueError("stray-light matrix output must end in .gsmath or .bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    values.astype(np.float32).tofile(path)
    return path


def _read_matrix(cls, path, wavelength, *, kind):
    path = Path(path)
    if path.suffix.lower() == ".gsmath":
        from .io.native import read_straylight_matrix
        saved_wavelength, values = read_straylight_matrix(path, kind=kind)
        if wavelength is not None and not np.array_equal(
            _grid(wavelength), saved_wavelength
        ):
            raise ValueError("wavelength grid differs from the native stray-light matrix")
        return cls(saved_wavelength, values)
    if path.suffix.lower() != ".bin":
        raise ValueError("stray-light matrix input must end in .gsmath or .bin")
    if wavelength is None:
        raise ValueError("legacy .bin stray-light matrices require a wavelength grid")
    wavelength = _grid(wavelength)
    values = np.fromfile(path, dtype=np.float32)
    if values.size != wavelength.size ** 2:
        raise ValueError("legacy .bin matrix size does not match the wavelength grid")
    return cls(wavelength, values.reshape(wavelength.size, wavelength.size))


@dataclass
class StrayLightDistribution:
    """Represent the response matrix mapping true to measured spectra."""
    wavelength: np.ndarray
    values: np.ndarray

    def correction(self) -> "StrayLightCorrection":
        """Invert the distribution matrix to obtain a correction matrix."""
        A = np.eye(len(self.wavelength)) + self.values
        C = np.linalg.inv(A)
        return StrayLightCorrection(self.wavelength.copy(), C)

    def regrid(self, wavelength, *, bounds="raise") -> "StrayLightDistribution":
        """Approximate rebuilding D after resampling the original lamp scans.

        Divide columns by their old local sample spacing, linearly interpolate
        both wavelength axes, then multiply by the new column spacing. This
        models the builder's in-band SUM normalization, assuming spacing is
        locally constant across each narrow lamp line. Local spacings are
        midpoint-cell widths, with full nearest spacings at the endpoints.

        This is not pixel relabeling or an exact change of operator basis.
        It cannot reconstruct changed peak masks, noise clipping, smoothing,
        or in-band sums. Validate against scan regeneration before adopting.
        Signed entries and the interpolated diagonal are retained.

        bounds='raise' rejects grids outside the original support; 'edge'
        explicitly holds the normalized kernel at the boundary. Always derive
        each trial from the original D, not from a previously regridded trial.
        """
        old, new = _grid(self.wavelength), _grid(wavelength)
        values = _matrix(self.values, old)
        if bounds not in {"raise", "edge"}:
            raise ValueError("bounds must be 'raise' or 'edge'")
        if bounds == "raise" and (new[0] < old[0] or new[-1] > old[-1]):
            raise ValueError("new wavelength grid extends outside the original support")
        if np.array_equal(old, new):
            return StrayLightDistribution(new.copy(), values.copy())
        kernel = values / np.gradient(old)[None, :]
        interp = RectBivariateSpline(old, old, kernel, kx=1, ky=1, s=0)
        at = np.clip(new, old[0], old[-1])
        result = interp(at, at) * np.gradient(new)[None, :]
        return StrayLightDistribution(new.copy(), result)

    def write(self, path: Path):
        """Write native ``.gsmath`` or legacy float32 ``.bin`` data."""
        return _write_matrix(self, path, kind="distribution")

    @classmethod
    def read(cls, path: Path, wavelength=None):
        """Read a native matrix, or a legacy binary with its supplied grid."""
        return _read_matrix(cls, path, wavelength, kind="distribution")

    def plot(self, path: Path):
        """Render the distribution matrix as an image."""
        fig, ax = plt.subplots(figsize=(10,8))

        image = ax.imshow(
            self.values,
            norm="log",
            origin="lower",
            aspect="auto",
        )

        fig.colorbar(image, ax=ax, label="Distribution")
        ax.set_title("Stray light distribution matrix")
        ax.set_xlabel("Input wavelength index")
        ax.set_ylabel("Detector wavelength index")

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


@dataclass
class StrayLightCorrection:
    """Represent an inverse response matrix used to correct measurements."""
    wavelength: np.ndarray
    values: np.ndarray

    def distribution(self) -> StrayLightDistribution:
        """Recover D = inverse(C) - I, retaining signed entries."""
        x = _grid(self.wavelength)
        values = _matrix(self.values, x)
        identity = np.eye(len(x))
        return StrayLightDistribution(x.copy(), np.linalg.solve(values, identity) - identity)

    def regrid(self, wavelength, *, bounds="raise") -> "StrayLightCorrection":
        """Regrid recovered D and rebuild C; see Distribution.regrid limits.

        For repeated trials recover distribution() once and regrid that D.
        """
        old, new = _grid(self.wavelength), _grid(wavelength)
        values = _matrix(self.values, old)
        if bounds not in {"raise", "edge"}:
            raise ValueError("bounds must be 'raise' or 'edge'")
        if bounds == "raise" and (new[0] < old[0] or new[-1] > old[-1]):
            raise ValueError("new wavelength grid extends outside the original support")
        if np.array_equal(old, new):
            return StrayLightCorrection(new.copy(), values.copy())
        return self.distribution().regrid(new, bounds=bounds).correction()

    def apply_matrix(self, Y: np.ndarray) -> np.ndarray:
        """Left-multiply spectral columns by the correction matrix."""
        return self.values @ Y

    def apply(self, spectra: list[Spectrum]) -> list[Spectrum]:
        """Correct compatible spectra together while preserving metadata."""
        if not spectra:
            return []

        for spectrum in spectra:
            if not np.array_equal(spectrum.wavelength, self.wavelength):
                raise ValueError(
                    f"Spectrum {spectrum.label!r} wavelength grid "
                    "does not match correction matrix."
                )

        Y = np.column_stack([s.values for s in spectra])
        corrected = self.apply_matrix(Y)

        return [
            spectrum.copy(
                values=corrected[:, i],
                label=f"{spectrum.label} [corrected]",
            )
            for i, spectrum in enumerate(spectra)
        ]

    def write(self, path: Path):
        """Write native ``.gsmath`` or legacy float32 ``.bin`` data."""
        return _write_matrix(self, path, kind="correction")

    @classmethod
    def read(cls, path: Path, wavelength=None):
        """Read a native matrix, or a legacy binary with its supplied grid."""
        return _read_matrix(cls, path, wavelength, kind="correction")

    def plot(self, path: Path):
        """Render the correction matrix as an image."""
        fig, ax = plt.subplots(figsize=(10, 8))

        image = ax.imshow(
            self.values - np.eye(self.values.shape[0]),
            origin="lower",
            aspect="auto",
        )

        fig.colorbar(image, ax=ax, label="Correction")
        ax.set_title("Stray light correction matrix")
        ax.set_xlabel("Input wavelength index")
        ax.set_ylabel("Output wavelength index")

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Monochromator scans
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineSpreadScan:
    """One monochromator scan: the detector's dark-subtracted response to a
    narrow line at ``set_wavelength_nm``.

    ``status`` is "unchecked" until :func:`characterize_scan` sets it to
    "accepted" or "rejected" (with the ``reason``). ``center_nm`` and
    ``fwhm_nm`` are the measured peak's, when one was found.
    """
    spectrum: Spectrum
    set_wavelength_nm: float
    label: str = ""
    center_nm: float | None = None
    fwhm_nm: float | None = None
    status: str = "unchecked"
    reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def in_band(scan: LineSpreadScan, wavelength, band_factor: float):
    """Samples within ``band_factor`` half-widths of the scan's centre: its line."""
    return np.abs(np.asarray(wavelength) - scan.center_nm) < scan.fwhm_nm / 2 * band_factor


def characterize_scan(scan: LineSpreadScan, *, max_center_error_nm: float = 2.5,
                      band_factor: float = 3.0, guard_fwhm: float = 1.0) -> LineSpreadScan:
    """``scan`` with its peak centre and width, accepted or rejected.

    A scan is rejected when it has no single primary peak, the peak's
    half-maximum crossings aren't both found, the centre is more than
    ``max_center_error_nm`` from the set wavelength, or the in-band region
    plus ``guard_fwhm`` widths runs off either end of the detector.
    """
    peaks = scan.spectrum.peaks(primary_only=True)
    if len(peaks.peak_indices) != 1:
        return replace(scan, status="rejected", reason="no primary peak")
    properties = peaks.properties
    if any(len(properties.get(key, ())) != 1 for key in ("left_ips", "right_ips")):
        return replace(scan, status="rejected", reason="missing FWHM intersections")
    center, fwhm = float(peaks.center_wavelength), float(peaks.fwhm)
    if not np.isfinite(center) or not np.isfinite(fwhm):
        return replace(scan, status="rejected", reason="invalid peak geometry")
    scan = replace(scan, center_nm=center, fwhm_nm=fwhm)
    error = abs(center - scan.set_wavelength_nm)
    if error > max_center_error_nm:
        return replace(scan, status="rejected",
                       reason=f"center error {error:.3f} nm > {max_center_error_nm:.3f} nm")
    x = scan.spectrum.wavelength
    reach = fwhm / 2 * band_factor + fwhm * guard_fwhm
    if center - reach < x[0] or center + reach > x[-1]:
        return replace(scan, status="rejected", reason="insufficient edge margin")
    return replace(scan, status="accepted", reason=None)


# ---------------------------------------------------------------------------
# Reading scans
# ---------------------------------------------------------------------------

def read_line_scan(path) -> Spectrum:
    """A line scan in dark-subtracted counts: a native measurement (dark
    subtracted here) or spectrum, a LightTouch export, or a two-column CSV
    (wavelength, signal; a header row is skipped)."""
    from .calibration import Calibration, SubtractDark
    from .io.formats import read_any
    from .measurement import Measurement

    path = Path(path)
    if path.suffix.lower() == ".gsmath":
        data = read_any(path)
        if isinstance(data, Measurement):
            if data.dark is None:
                raise ValueError(f"{path.name}: the measurement has no dark")
            return data.process(Calibration("dark", [SubtractDark()]))
        if not isinstance(data, Spectrum):
            raise ValueError(f"{path.name} holds a {type(data).__name__}, not a scan")
        return data
    try:
        return Spectrum.from_lighttouch(path)
    except (ValueError, KeyError, IndexError):
        pass
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            try:
                rows.append([float(value) for value in row])
            except ValueError:
                if rows:
                    raise ValueError(f"{path.name}: non-numeric row after the data started: {row}")
    data = np.array(rows)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path.name}: expected two numeric columns")
    return Spectrum(wavelength=data[:, 0], values=data[:, -1], label=path.stem,
                    wavelength_unit="nm", value_unit="count")


def set_wavelength_from_name(path: Path) -> float:
    match = re.search(r"StrayLight_([0-9.]+?)(?:\.csv|\.gsmath)?$", path.name)
    if match is None:
        raise ValueError(f"cannot tell the set wavelength from {path.name!r}; use a manifest.csv")
    return float(match.group(1))


def read_line_scans(directory, pattern="StrayLight_*.csv") -> list[LineSpreadScan]:
    """The line scans in ``directory``, by its ``manifest.csv`` or by ``pattern``."""
    directory = Path(directory)
    manifest = directory / "manifest.csv"
    if manifest.is_file():
        with manifest.open(encoding="utf-8", newline="") as stream:
            entries = [(directory / row["file"], float(row["set_wavelength_nm"]))
                       for row in csv.DictReader(stream)]
    else:
        entries = [(path, set_wavelength_from_name(path)) for path in sorted(directory.glob(pattern))]
    if not entries:
        raise FileNotFoundError(f"no scans in {directory} (no manifest.csv, nothing matches {pattern!r})")
    scans = [LineSpreadScan(read_line_scan(path), wavelength, label=path.name) for path, wavelength in entries]
    return sorted(scans, key=lambda scan: scan.set_wavelength_nm)


# ---------------------------------------------------------------------------
# Distribution matrix
# ---------------------------------------------------------------------------

INTERPOLATIONS = ("shifted", "linear")


def build_distribution(scans, *, band_factor: float = 3.0, suppression: float = 1.0,
                       one_count: float = ONE_COUNT, gaussian_sigma: float = 1.0,
                       interpolation: str = "shifted") -> StrayLightDistribution:
    """The distribution matrix D from accepted scans (others are ignored).

    Each scan is normalised to its peak; values below ``one_count`` are set
    to zero (the detector's noise floor). Its column of D is that, divided by
    its in-band sum and scaled by ``suppression``, with the in-band samples
    zeroed. The columns are then filled in for every detector wavelength
    between the scans' centres:

    - ``"shifted"`` (default): each column is interpolated in distance from
      the line. Both neighbouring scans' columns are shifted so their lines
      sit at the source wavelength, then blended linearly, so the near-band
      shoulders move with the line. Beyond the outermost scans their columns
      are shifted. Smoothing (``gaussian_sigma`` detector samples, 0: none)
      is along detector wavelength only. On a dense sweep this lets the scans
      be many line widths apart without loss; it would smear a stray feature
      that is fixed in detector wavelength rather than moving with the line.
    - ``"linear"``: each detector sample is interpolated linearly between the
      scans' columns, at fixed detector wavelength; between two scans that
      blends two lines' shoulders at two places, so it needs scans close
      together. The columns are smoothed with a Gaussian of
      ``gaussian_sigma`` matrix samples on both axes first, and beyond the
      outermost scans the end columns are extended unchanged.
    """
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, not {interpolation!r}")
    scans = sorted((s for s in scans if s.accepted), key=lambda s: s.center_nm)
    if len(scans) < 2:
        raise ValueError("at least two accepted scans are needed to build D")
    x = scans[0].spectrum.wavelength
    columns, centers = [], []
    for scan in scans:
        name = scan.label or f"{scan.set_wavelength_nm:g} nm"
        if not np.array_equal(scan.spectrum.wavelength, x):
            raise ValueError(f"{name}: wavelength grid differs from the other scans")
        y = scan.spectrum.normalize(by="peak").values.copy()
        y[y < one_count] = 0
        band = in_band(scan, x, band_factor)
        denominator = np.sum(y[band])
        if denominator <= 0:
            raise ValueError(f"{name}: no in-band signal")
        column = y / denominator * suppression
        column[band] = 0
        columns.append(column)
        centers.append(scan.center_nm)
    centers = np.asarray(centers)
    if np.any(np.diff(centers) <= 0):
        raise ValueError("accepted scan centres must be distinct")
    measured = np.clip(np.column_stack(columns), 0, None)
    if interpolation == "shifted":
        return StrayLightDistribution(np.asarray(x, dtype=float).copy(),
                                      _shifted(measured, centers, x, gaussian_sigma))
    if gaussian_sigma:
        measured = scipy.ndimage.gaussian_filter(measured, sigma=gaussian_sigma)
    interpolator = RectBivariateSpline(x, centers, measured, kx=1, ky=1, s=0)
    return StrayLightDistribution(np.asarray(x, dtype=float).copy(), interpolator(x, x))


def _shifted(measured, centers, x, gaussian_sigma):
    """D's columns for every detector wavelength, from the scans' columns
    ``measured`` (one per centre), interpolated in distance from the line."""
    if gaussian_sigma:
        measured = scipy.ndimage.gaussian_filter1d(measured, sigma=gaussian_sigma, axis=0)
    x = np.asarray(x, dtype=float)
    right = np.clip(np.searchsorted(centers, x), 1, len(centers) - 1)
    weight = np.clip((x - centers[right - 1]) / (centers[right] - centers[right - 1]), 0, 1)
    D = np.empty((len(x), len(x)))
    for i, source in enumerate(x):
        j = right[i]
        below = np.interp(x - (source - centers[j - 1]), x, measured[:, j - 1], left=0, right=0)
        above = np.interp(x - (source - centers[j]), x, measured[:, j], left=0, right=0)
        D[:, i] = (1 - weight[i]) * below + weight[i] * above
    return D


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _scan_metrics(scan, corrected, band_factor):
    x = scan.spectrum.wavelength
    y0 = np.asarray(scan.spectrum.values, dtype=float)
    peak = np.nanmax(y0)
    if not np.isfinite(peak) or peak <= 0:
        return None
    y0 = y0 / peak  # both relative to the original peak
    y1 = np.asarray(corrected.values, dtype=float) / peak
    band = in_band(scan, x, band_factor)
    radius = scan.fwhm_nm / 2 * band_factor
    left, right = x <= scan.center_nm - radius, x >= scan.center_nm + radius

    def out_of_band(values):
        total = 0.0
        for side in (left, right):
            if np.count_nonzero(side) >= 2:
                total += np.trapezoid(values[side], x[side])
        return float(total)

    before = out_of_band(np.clip(y0, 0, None))
    after = out_of_band(np.clip(y1, 0, None))
    in_before = float(np.trapezoid(y0[band], x[band]))
    in_after = float(np.trapezoid(y1[band], x[band]))
    return {
        "scan": scan.label, "set_wavelength_nm": scan.set_wavelength_nm,
        "center_nm": scan.center_nm, "fwhm_nm": scan.fwhm_nm,
        "out_of_band_before": before, "out_of_band_after": after,
        "suppression_ratio": after / before if before > 0 else float("nan"),
        "negative_area": float(np.trapezoid(np.clip(-y1, 0, None), x)),
        "minimum": float(np.nanmin(y1)),
        "in_band_ratio": in_after / in_before if in_before else float("nan"),
    }


@dataclass(frozen=True, repr=False)
class StrayLightEvaluation(Result):
    """A correction applied to line scans: what it removed and what it cost.

    Per scan (:meth:`rows`), relative to the scan's original peak:

    - ``out_of_band_before`` / ``_after``: the positive signal outside the
      in-band region, integrated over wavelength (nm);
    - ``suppression_ratio``: after / before; smaller is better;
    - ``negative_area`` and ``minimum``: negative values the correction
      introduced (ideally none);
    - ``in_band_ratio``: the in-band integral after / before (ideally 1).

    No verdict: limits for these depend on the instrument and are not yet set.
    """
    metrics: tuple
    band_factor: float

    units = {"out_of_band_before": "nm", "out_of_band_after": "nm", "negative_area": "nm"}

    def rows(self):
        """One record per scan, for tables and CSV files."""
        return [dict(m) for m in self.metrics]

    @property
    def summary(self):
        def stat(name, fn):
            values = np.array([m[name] for m in self.metrics], dtype=float)
            values = values[np.isfinite(values)]
            return float(fn(values)) if len(values) else None
        return {
            "scans": len(self.metrics),
            "median_suppression_ratio": stat("suppression_ratio", np.median),
            "max_suppression_ratio": stat("suppression_ratio", np.max),
            "max_negative_area": stat("negative_area", np.max),
            "minimum": stat("minimum", np.min),
            "in_band_ratio_min": stat("in_band_ratio", np.min),
            "in_band_ratio_max": stat("in_band_ratio", np.max),
        }

    def table(self):
        """One line per scan, as plain text."""
        header = ["Set nm", "Centre nm", "FWHM nm", "Out of band before", "after", "Ratio",
                  "Negative", "In band"]
        body = [[f"{m['set_wavelength_nm']:g}", f"{m['center_nm']:.2f}", f"{m['fwhm_nm']:.2f}",
                 f"{m['out_of_band_before']:.3g}", f"{m['out_of_band_after']:.3g}",
                 f"{m['suppression_ratio']:.3f}", f"{m['negative_area']:.2g}",
                 f"{m['in_band_ratio']:.4f}"] for m in self.metrics]
        widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]
        lines = ["  ".join(c.rjust(w) for c, w in zip(row, widths)) for row in [header] + body]
        lines.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join(lines)


def evaluate_correction(correction: StrayLightCorrection, scans, *, band_factor: float = 3.0
                        ) -> StrayLightEvaluation:
    """Apply ``correction`` to the scans that have a measured peak, and report
    what it did (see :class:`StrayLightEvaluation`). Run the scans through
    :func:`characterize_scan` first."""
    scans = [s for s in scans if s.center_nm is not None and s.fwhm_nm is not None]
    corrected = correction.apply([s.spectrum for s in scans])
    metrics = (_scan_metrics(s, c, band_factor) for s, c in zip(scans, corrected))
    return StrayLightEvaluation(tuple(m for m in metrics if m is not None), band_factor)


# ---------------------------------------------------------------------------
# The whole calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class StrayLightCalibrationResult(Result):
    """Scans in, matrices out: every scan's verdict, D, C, and C evaluated on
    the accepted scans.

    ``edge_extrapolation`` is True when the accepted scans' centres don't
    reach both ends of the detector, so the end columns of D are extended
    rather than measured.
    """
    scans: tuple
    distribution: StrayLightDistribution
    correction: StrayLightCorrection
    evaluation: StrayLightEvaluation
    settings: dict

    units = {"condition_number": None, "centers_nm": "nm", "detector_nm": "nm"}

    @property
    def accepted(self):
        return tuple(s for s in self.scans if s.accepted)

    @property
    def condition_number(self):
        return float(np.linalg.cond(self.correction.values))

    @property
    def edge_extrapolation(self):
        x = self.correction.wavelength
        centers = [s.center_nm for s in self.accepted]
        return bool(min(centers) > x[0] or max(centers) < x[-1])

    @property
    def summary(self):
        centers = [s.center_nm for s in self.accepted]
        x = self.correction.wavelength
        return {
            "scans": len(self.scans), "accepted": len(centers),
            "rejected": len(self.scans) - len(centers),
            "condition_number": self.condition_number,
            "centers_nm": [min(centers), max(centers)],
            "detector_nm": [float(x[0]), float(x[-1])],
            "edge_extrapolation": self.edge_extrapolation,
            "evaluation": self.evaluation.summary,
        }

    def scan_rows(self):
        """One record per scan, accepted or not, for tables and CSV files."""
        return [{"scan": s.label, "set_wavelength_nm": s.set_wavelength_nm,
                 "center_nm": s.center_nm, "fwhm_nm": s.fwhm_nm, "status": s.status,
                 "reason": s.reason or ""} for s in self.scans]

    def write(self, directory):
        """Write the result to ``directory``: ``matrices/`` (native
        ``D.gsmath``, ``C.gsmath`` and legacy float32 ``D.bin``, ``C.bin``),
        ``plots/`` (D, C, bandwidth), ``scans.csv``, ``evaluation.csv`` and
        ``summary.json``. Returns the path of ``C.gsmath``."""
        directory = Path(directory)
        matrices, plots = directory / "matrices", directory / "plots"
        matrices.mkdir(parents=True, exist_ok=True)
        plots.mkdir(parents=True, exist_ok=True)
        self.distribution.write(matrices / "D.gsmath")
        self.correction.write(matrices / "C.gsmath")
        self.distribution.write(matrices / "D.bin")
        self.correction.write(matrices / "C.bin")
        self.distribution.plot(plots / "distribution.png")
        self.correction.plot(plots / "correction.png")
        plot_fwhm(self.accepted, plots / "fwhm.png")
        write_rows(self.scan_rows(), directory / "scans.csv")
        write_rows(self.evaluation.rows(), directory / "evaluation.csv")
        summary = {**self.summary, "settings": self.settings,
                   "matrix": str(matrices / "C.gsmath")}
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                encoding="utf-8")
        return matrices / "C.gsmath"


def stray_light_calibration(scans, *, max_center_error_nm: float = 2.5, band_factor: float = 3.0,
                            suppression: float = 1.0, one_count: float = ONE_COUNT,
                            gaussian_sigma: float = 1.0, guard_fwhm: float = 1.0,
                            min_accepted: int = 2, interpolation: str = "shifted"
                            ) -> StrayLightCalibrationResult:
    """Characterise the scans, build D and C from the accepted ones, and
    evaluate C on them. Raises ValueError with fewer than ``min_accepted``
    accepted scans."""
    scans = tuple(characterize_scan(s, max_center_error_nm=max_center_error_nm,
                                    band_factor=band_factor, guard_fwhm=guard_fwhm)
                  for s in scans)
    accepted = [s for s in scans if s.accepted]
    needed = max(2, min_accepted)
    if len(accepted) < needed:
        raise ValueError(f"{len(accepted)} of {len(scans)} scans accepted; at least {needed} are needed")
    distribution = build_distribution(accepted, band_factor=band_factor, suppression=suppression,
                                      one_count=one_count, gaussian_sigma=gaussian_sigma,
                                      interpolation=interpolation)
    correction = distribution.correction()
    settings = {"max_center_error_nm": max_center_error_nm, "band_factor": band_factor,
                "suppression": suppression, "one_count": one_count,
                "gaussian_sigma": gaussian_sigma, "guard_fwhm": guard_fwhm,
                "interpolation": interpolation}
    evaluation = evaluate_correction(correction, accepted, band_factor=band_factor)
    return StrayLightCalibrationResult(scans, distribution, correction, evaluation, settings)


# ---------------------------------------------------------------------------
# Reports and plots
# ---------------------------------------------------------------------------

def write_rows(rows, path):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot_fwhm(scans, path):
    """The measured bandwidth (FWHM) against centre wavelength."""
    scans = [s for s in scans if s.accepted]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([s.center_nm for s in scans], [s.fwhm_nm for s in scans], ".-")
    ax.set_xlabel("Centre wavelength (nm)")
    ax.set_ylabel("FWHM (nm)")
    ax.set_title("Measured spectral bandwidth")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correction(original: Spectrum, corrected: Spectrum, path):
    """A scan before and after correction, peak-normalised, on a log scale."""
    fig, ax = plt.subplots(figsize=(9, 5))
    a, b = original.normalize(by="peak"), corrected.normalize(by="peak")
    ax.plot(a.wavelength, a.values, label="Original")
    ax.plot(b.wavelength, b.values, label="Corrected")
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised signal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
