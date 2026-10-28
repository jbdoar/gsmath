"""Shared interpretation of Gamma Scientific LightTouch quantity headers."""

LIGHTTOUCH_QUANTITIES = {
    "radiance": ("Radiance (µW/cm²/sr)", "uW / cm^2 / sr / nm"),
    "irradiance": ("Irradiance (µW/cm²)", "uW / cm^2 / nm"),
    "counts": ("Counts", "count"),
}


def lighttouch_unit_from_header(header):
    """Return the spectral unit named by a LightTouch header, or ``None``."""
    words = header[2].split() if len(header) > 2 else []
    entry = LIGHTTOUCH_QUANTITIES.get(words[0].lower()) if words else None
    return None if entry is None else entry[1]


def lighttouch_unit(spectrum):
    """The spectral quantity unit named by a LightTouch spectrum's header.

    ``Radiance (µW/cm²/sr)`` is µW/cm²/sr/nm, ``Irradiance (µW/cm²)`` is
    µW/cm²/nm, and ``Counts`` is count. For an amplitude ``.cal`` this is the
    calibrated output quantity; the file's factors carry that unit per count.
    """
    return lighttouch_unit_from_header(spectrum.metadata.get("header") or [])
