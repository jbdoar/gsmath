"""Robertson isotemperature-line data for Planckian calculations."""

from dataclasses import dataclass

import numpy as np

from ._paths import ROBERTSON_TABLE


@dataclass(frozen=True)
class RobertsonTable:
    """Store reciprocal temperatures, UCS coordinates, and line slopes."""
    mired: np.ndarray
    temperature: np.ndarray
    u: np.ndarray
    v: np.ndarray
    slope: np.ndarray
    du: np.ndarray
    dv: np.ndarray


def _load_robertson_table() -> RobertsonTable:
    """Load the bundled Robertson table once at module import."""
    data = np.genfromtxt(
        ROBERTSON_TABLE,
        delimiter=",",
        comments="#",
        skip_header=1,
    )

    mired = data[:, 0]
    slope = data[:, 3]

    length = np.hypot(1.0, slope)

    temperature = np.empty_like(mired, dtype=float)
    mask = mired != 0
    temperature[mask] = 1e6 / mired[mask]
    temperature[~mask] = np.inf
    
    return RobertsonTable(
        mired=mired,
        temperature=temperature,
        u=data[:, 1],
        v=data[:, 2],
        slope=slope,
        du=1.0 / length,
        dv=slope / length,
    )


ROBERTSON = _load_robertson_table()
