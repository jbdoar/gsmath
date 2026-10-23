from gsmath import Spectrum, units


def test_spectrum_summary_can_include_per_quantity_units():
    spectrum = Spectrum(
        [400, 500, 600],
        [1.0, 2.0, 1.0],
        value_unit="uW / nm",
    )

    summary = spectrum.summary(include_units=True)

    assert summary["power"]["integral"] == {
        "value": 300.0,
        "unit": "µW",
    }
    assert summary["peaks"]["fwhm"]["unit"] == "nm"
    assert summary["statistics"]["mean"]["unit"] == "µW / nm"
    assert summary["color"]["2deg"]["x"]["unit"] == "dimensionless"
    assert summary["color"]["2deg"]["cct"]["unit"] == "K"
    assert summary["color"]["2deg"]["X"]["unit"] == "lm"
    assert summary["color"]["2deg"]["L_star"]["unit"] == "dimensionless"
    assert summary["color"]["2deg"]["u_star"]["unit"] == "dimensionless"


def test_spectrum_summary_units_returns_pint_units():
    spectrum = Spectrum([400, 500, 600], [1, 2, 1], value_unit="count")

    result = spectrum.summary_units()

    assert result["power"]["integral"] == units.ureg.count * units.ureg.nm
    assert result["peaks"]["peak_wavelength"] == units.ureg.nm
    assert result["statistics"]["max"] == units.ureg.count


def test_profiles_from_analyses():
    import numpy as np
    import pytest

    from gsmath import Profile, Spectrum, from_analysis
    from gsmath.units import parse

    wl = np.arange(400.0, 701.0)
    led = Spectrum(wl, np.exp(-4 * np.log(2) * ((wl - 550) / 30) ** 2), value_unit="W/sr/m^2/nm")
    profile = Profile("LED", {
        "peaks": from_analysis("peaks", {"wavelength": "peak_wavelength", "width": "fwhm"}),
        "colour": from_analysis("color", {"Y": "Y", "cct": "planck.cct"},
                                options={"observer": "10deg"}, relative=["Y"]),
    })
    summary = led.summary(profile)
    assert summary["peaks"]["wavelength"] == pytest.approx(550, abs=0.5)
    assert summary["peaks"]["width"] == pytest.approx(30, abs=0.5)
    assert summary["colour"]["cct"] == led.color(observer="10deg").planck.cct
    units = led.summary_units(profile)
    assert units["peaks"]["width"] == parse("nm") and units["colour"]["cct"] == parse("K")
    assert profile.fields["colour"]["Y"].mode == "relative"
    assert profile.fields["peaks"]["width"].mode == "difference"
    with pytest.raises(KeyError, match="fields not taken"):
        from_analysis("peaks", ["fwhm"], relative=["Y"])
