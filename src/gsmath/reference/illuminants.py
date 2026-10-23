"""Bundled standard illuminant spectral distributions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._paths import ILLUMINANTS_DIR


@dataclass(frozen=True)
class Illuminant:
    """Store a named illuminant on its reference wavelength grid."""
    name: str
    wavelength: np.ndarray
    values: np.ndarray
    
    @classmethod
    def from_csv(
            cls,
            filename: str | Path,
            name: str,
    ) -> "Illuminant":
        """Load a named illuminant from a packaged CSV file."""

        data = np.genfromtxt(filename, delimiter=",", names=True)

        return cls(
            name=name,
            wavelength=data["wavelength"],
            values=data["values"],
        )


ILLUMINANTS: dict[str, Illuminant] = {
    "A": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_A.csv", name="A"),
    "B": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_B.csv", name="B"),
    "C": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_C.csv", name="C"),
    "D50": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_D50.csv", name="D50"),
    "D55": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_D55.csv", name="D55"),
    "D65": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_D65.csv", name="D65"),
    "D75": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_D75.csv", name="D75"),
    "E": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_E.csv", name="E"),
    "F1": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F01.csv", name="F1"),
    "F2": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F02.csv", name="F2"),
    "F3": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F03.csv", name="F3"),
    "F4": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F04.csv", name="F4"),
    "F5": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F05.csv", name="F5"),
    "F6": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F06.csv", name="F6"),
    "F7": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F07.csv", name="F7"),
    "F8": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F08.csv", name="F8"),
    "F9": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F09.csv", name="F9"),
    "F10": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F10.csv", name="F10"),
    "F11": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F11.csv", name="F11"),
    "F12": Illuminant.from_csv(ILLUMINANTS_DIR / "illuminant_F12.csv", name="F12"),
}
    


__all__ = [
    "Illuminant",
    "ILLUMINANTS",
]
