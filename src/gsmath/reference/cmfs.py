"""Bundled CIE color-matching function tables."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._paths import CMFS_DIR


@dataclass(frozen=True)
class ColorMatchingFunction:
    """Store wavelength-aligned x-bar, y-bar, and z-bar observer functions."""
    name: str
    wavelength: np.ndarray
    xbar: np.ndarray
    ybar: np.ndarray
    zbar: np.ndarray

    @classmethod
    def from_csv(
            cls,
            filename: str | Path,
            name: str,
    ) -> "ColorMatchingFunction":
        """Load a named observer table from a packaged CSV file."""

        data = np.genfromtxt(filename, delimiter=",", names=True)

        return cls(
            name=name,
            wavelength=data["wavelength"],
            xbar=data["xbar"],
            ybar=data["ybar"],
            zbar=data["zbar"],
        )

    @property
    def values(self):
        """Return matching functions as wavelength-by-three columns."""
        return np.column_stack((self.xbar, self.ybar, self.zbar))

    
CMFS: dict[str, ColorMatchingFunction] = {
    "2deg": ColorMatchingFunction.from_csv(
        CMFS_DIR / "cie1931_2deg_xyzbar.csv",
        name="2deg",
        ),
    "10deg": ColorMatchingFunction.from_csv(
        CMFS_DIR / "cie1964_10deg_xyzbar.csv",
        name="10deg",
        ),
}

__all__ = [
    "ColorMatchingFunction",
    "CMFS",
]
