"""Matrix representations for spectral stray-light response and correction."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .spectrum import Spectrum


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
