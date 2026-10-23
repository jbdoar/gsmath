"""Writer for Gamma Scientific LightTouch amplitude-calibration files."""

from pathlib import Path

import numpy as np


def to_lighttouch(
    spectrum,
    path,
    *,
    integration_time_ms=0.0,
    name="LightTouch Amplitude Cal (1500 nm,0 ms,0 ms)",
):
    """Write a wavelength/value spectrum as a LightTouch amplitude ``.cal``.

    LightTouch amplitude calibration files store calibration factors as the
    two spectral columns. Units are identified by the file header rather than
    by per-column unit metadata in this legacy format.
    """
    path = Path(path)
    wavelength = np.asarray(spectrum.wavelength, dtype=float)
    values = np.asarray(spectrum.values, dtype=float)
    if (wavelength.ndim != 1 or values.shape != wavelength.shape or
            len(wavelength) < 2 or not np.all(np.isfinite(wavelength)) or
            not np.all(np.diff(wavelength) > 0)):
        raise ValueError("LightTouch output requires matching, finite, increasing spectral arrays")
    if not np.isfinite(integration_time_ms) or integration_time_ms < 0:
        raise ValueError("integration_time_ms must be finite and nonnegative")
    if not np.all(np.isfinite(values)):
        raise ValueError("LightTouch amplitude calibration values must be finite")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("2.0 [Gamma Scientific Spectral File]\n")
        stream.write(f"{name}\n")
        stream.write("Radiance (µW/cm²/sr)  \n")
        # LightTouch's reader locates data after 4 + this count + 3 lines.
        stream.write("1\n")
        stream.write(f"{integration_time_ms:.8g} [Integration Time ms]\n")
        stream.write(f"{wavelength[0]:.8f}\n")
        stream.write(f"{len(wavelength)}\n")
        stream.write(f"{wavelength[-1]:.8f}\n")
        for wl, value in zip(wavelength, values):
            stream.write(f"{wl:.6f}, {value:.8e}\n")
    return path
