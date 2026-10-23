"""Robertson-table estimation of correlated color temperature and Duv."""

from dataclasses import dataclass

import numpy as np

from ..reference import ROBERTSON


@dataclass(frozen=True)
class ColorTemperatureResult:
    """Hold correlated color temperature and Planckian-locus distance."""
    cct: float
    duv: float


def cct_robertson(u: float, v: float) -> ColorTemperatureResult:
    """Interpolate CCT and Duv from CIE 1960 coordinates using Robertson lines
    (one point; see :func:`cct_robertson_array`)."""
    cct, duv = cct_robertson_array(u, v)
    return ColorTemperatureResult(cct=float(cct), duv=float(duv))


def cct_robertson_array(u, v):
    """:func:`cct_robertson` for arrays of CIE 1960 ``u``, ``v`` (an image's
    pixels): ``(cct, duv)`` arrays of their shape, NaN where u or v is."""
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    shape = u.shape
    u, v = u.ravel(), v.ravel()
    finite = np.isfinite(u) & np.isfinite(v)
    table = ROBERTSON
    length = np.hypot(1.0, table.slope)
    du = 1.0 / length
    dv = table.slope / length
    last = len(table.mired) - 1

    # Signed distance from each isotemperature line i = 1..last; the
    # bracketing line is the first with distance <= 0 (else the last).
    lines = np.arange(1, last + 1)
    distance = (-(u[:, None] - table.u[lines]) * dv[lines]
                + (v[:, None] - table.v[lines]) * du[lines])
    crossed = (distance <= 0) | (lines == last)
    first = np.argmax(crossed, axis=1)
    i = lines[first]
    rows = np.arange(u.size)
    here = np.maximum(-distance[rows, first], 0.0)
    previous = np.where(first > 0, distance[rows, np.maximum(first - 1, 0)], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(i == 1, 0.0, here / (previous + here))

    def blend(values):
        return values[i - 1] * fraction + values[i] * (1.0 - fraction)

    mired = blend(table.mired)
    direction_u, direction_v = blend(du), blend(dv)
    direction_length = np.hypot(direction_u, direction_v)
    direction_u, direction_v = direction_u / direction_length, direction_v / direction_length
    duv = -((u - blend(table.u)) * direction_u + (v - blend(table.v)) * direction_v)
    with np.errstate(divide="ignore"):
        cct = np.where(mired == 0, np.inf, 1e6 / np.where(mired == 0, 1.0, mired))
    cct, duv = np.where(finite, cct, np.nan), np.where(finite, duv, np.nan)
    return cct.reshape(shape), duv.reshape(shape)
