import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gsmath import (
    CMFS, Calibration, ColorMatrixFit, Image, Measurement, Rectangle, SpectralSensitivity,
    Spectrum, fit_color_matrix, site_means,
)
from gsmath.mosaic import site_masks
from gsmath.units import Q_, parse

WL = np.arange(380.0, 781.0, 1.0)
RADIANCE = "W/sr/m^2/nm"


def gauss(mu, fwhm):
    return np.exp(-4 * np.log(2) * ((WL - mu) / fwhm) ** 2)


def light(values):
    return Spectrum(WL, np.asarray(values) * 1e-3, value_unit=RADIANCE)


def camera():
    """A plausible Bayer camera: not a mix of the colour-matching functions."""
    return SpectralSensitivity(
        WL, np.column_stack([gauss(600, 70) + 0.05 * gauss(440, 40), gauss(530, 95),
                             gauss(462, 75)]) * 50, ("R", "G", "B"), "count/ms / (W/sr/m^2)")


def luther_camera():
    """A camera whose channels are an exact mix of the CIE 1931 functions."""
    cmf = np.column_stack([np.interp(WL, CMFS["2deg"].wavelength, CMFS["2deg"].values[:, k])
                           for k in range(3)])
    mix = np.array([[0.9, 0.4, -0.1], [-0.2, 1.1, 0.1], [0.05, -0.1, 1.2]])
    return SpectralSensitivity(WL, cmf @ mix.T * 40, ("R", "G", "B"), "count/ms / (W/sr/m^2)")


def XYZ(spectra):
    return np.array([s.color().XYZ for s in spectra])


PRIMARIES = [gauss(630, 30), gauss(530, 35), gauss(455, 22)]
LED_PEAKS = [405, 430, 450, 470, 500, 525, 550, 575, 595, 615, 635, 660]


def led_mixes(n, seed):
    rng = np.random.default_rng(seed)
    channels = [gauss(mu, 22) for mu in LED_PEAKS]
    return [light(sum(w * c for w, c in zip(rng.uniform(0, 1, len(channels)), channels)))
            for _ in range(n)]


def test_monochromator_scan_gives_the_sensitivity():
    truth = camera()
    band = np.linspace(2.0, 3.0, WL.size)  # band radiance, W/sr/m^2
    signal = truth.values * band[:, None]  # counts/ms
    measured = SpectralSensitivity.from_monochromator(
        WL, signal, band, signal_unit="count/ms", reference_unit="W/sr/m^2")
    np.testing.assert_allclose(measured.values, truth.values)
    assert measured.value_unit == parse("count/ms / (W/sr/m^2)")
    response = measured.response(light(PRIMARIES[1]))
    assert response.units == parse("count/ms")  # per (W/sr/m^2) x W/sr/m^2/nm x nm
    assert measured.channel("G").value_unit == measured.value_unit
    with pytest.raises(ValueError, match="positive"):
        SpectralSensitivity.from_monochromator(WL, signal, -band)


def test_response_integrates_the_sensitivity_against_the_light():
    sensitivity = SpectralSensitivity(WL, np.ones((WL.size, 1)), ("V",))
    flat = Spectrum([300, 900], [2.0, 2.0])  # wider than the sensitivity
    assert sensitivity.response(flat)[0] == pytest.approx(2.0 * 400)
    narrow = Spectrum([500, 600], [1.0, 1.0])  # zero outside its own range
    assert sensitivity.response(narrow)[0] == pytest.approx(100, abs=1)


def test_a_luther_camera_is_exact_for_any_light():
    cam = luther_camera()
    training = [light(g) for g in PRIMARIES] + led_mixes(5, 1)
    fit = fit_color_matrix(cam.responses(training), XYZ(training))
    held = led_mixes(20, 2) + [light(gauss(mu, 10)) for mu in range(420, 680, 20)]
    report = fit.validate(cam.responses(held), XYZ(held))
    assert report["delta_uv_prime"]["max"] < 1e-6 and report["delta_Y"]["max"] < 1e-6


def test_a_displays_own_primaries_give_an_exact_matrix():
    cam = camera()
    training = [light(g) for g in PRIMARIES] + [light(sum(PRIMARIES))]
    fit = fit_color_matrix(cam.responses(training), XYZ(training))
    rng = np.random.default_rng(4)
    held = [light(sum(w * g for w, g in zip(rng.uniform(0, 1, 3), PRIMARIES))) for _ in range(30)]
    report = fit.validate(cam.responses(held), XYZ(held))
    assert report["delta_uv_prime"]["max"] < 1e-6 and report["delta_Y"]["max"] < 1e-6


def test_tunable_led_mixes_to_within_what_three_channels_allow():
    cam = camera()
    training = [light(gauss(mu, 22)) for mu in LED_PEAKS] + led_mixes(40, 5)
    held = led_mixes(60, 6)
    xyz_fit = fit_color_matrix(cam.responses(training), XYZ(training))
    report = xyz_fit.validate(cam.responses(held), XYZ(held))
    assert report["delta_uv_prime"]["mean"] < 0.012 and report["delta_Y"]["mean"] < 0.06
    uv_fit = fit_color_matrix(cam.responses(training), XYZ(training), method="uv_Y")
    assert (uv_fit.training_errors["delta_uv_prime"] ** 2).mean() < \
        (xyz_fit.training_errors["delta_uv_prime"] ** 2).mean()


def test_weighting_by_luminance_alone_is_not_the_default():
    cam = camera()
    training = [light(gauss(mu, 22)) for mu in LED_PEAKS] + led_mixes(40, 5)
    held = led_mixes(60, 6)
    measured, reference = cam.responses(training), XYZ(training)
    relative = fit_color_matrix(measured, reference)
    by_Y = fit_color_matrix(measured, reference, weighting="absolute",
                            weights=1 / reference[:, 1])
    assert relative.validate(cam.responses(held), XYZ(held))["delta_uv_prime"]["mean"] < \
        by_Y.validate(cam.responses(held), XYZ(held))["delta_uv_prime"]["mean"]


def test_preserving_a_white_exactly():
    cam = camera()
    training = [light(gauss(mu, 22)) for mu in LED_PEAKS] + [light(sum(PRIMARIES))]
    measured, reference = cam.responses(training), XYZ(training)
    white = len(training) - 1
    fit = fit_color_matrix(measured, reference, preserve=white)
    np.testing.assert_allclose(fit.predict(measured[white:white + 1])[0], reference[white],
                               rtol=1e-10)
    free = fit_color_matrix(measured, reference)
    assert abs(free.training_errors["delta_Y"][white]) > 1e-6  # not exact without it


def test_fitting_for_ciede2000():
    cam = camera()
    training = [light(gauss(mu, 22)) for mu in LED_PEAKS] + led_mixes(20, 7)
    measured, reference = cam.responses(training), XYZ(training)
    white = reference[np.argmax(reference[:, 1])] * 1.2
    xyz_fit = fit_color_matrix(measured, reference, white=white)
    de_fit = fit_color_matrix(measured, reference, method="delta_e2000", white=white)
    assert de_fit.training_errors["delta_e2000"].mean() < xyz_fit.training_errors["delta_e2000"].mean()
    with pytest.raises(ValueError, match="needs a white"):
        fit_color_matrix(measured, reference, method="delta_e2000")


def test_units_and_the_calibration_step(tmp_path):
    cam = camera()
    training = [light(g) for g in PRIMARIES] + [light(sum(PRIMARIES))]
    fit = fit_color_matrix(cam.responses(training), XYZ(training), unit="cd/m^2 / (count/ms)")
    assert isinstance(fit, ColorMatrixFit) and fit.matrix_unit == parse("cd/m^2 / (count/ms)")
    inferred = fit_color_matrix(cam.responses(training), Q_(XYZ(training), "cd/m^2"))
    assert inferred.matrix_unit == fit.matrix_unit
    rgb = Image(np.broadcast_to(cam.response(training[3]).magnitude, (4, 4, 3)),
                value_unit="count/ms", channels=("R", "G", "B"))
    xyz = rgb.apply_matrix(fit, unit=fit.matrix_unit)  # the fit works as the matrix
    np.testing.assert_allclose(xyz.values[0, 0], XYZ(training)[3], rtol=1e-9)
    assert xyz.to_value_unit("cd/m^2") is not None
    pipeline = Calibration("colorimeter", [fit.step()])
    loaded = Calibration.read(pipeline.write(tmp_path / "c.gsmath"))
    np.testing.assert_allclose(Measurement(raw=rgb).process(loaded).values, xyz.values)
    assert set(fit.summary) == {"method", "samples", "matrix", "unit", "training"}


def test_site_means_of_a_raw_patch():
    masks = site_masks("GRBG", (8, 10))
    raw = Image(np.where(masks["R"], 300.0, np.where(masks["G"], 200.0, 100.0)),
                mosaic="GRBG")
    np.testing.assert_allclose(site_means(raw), [300, 200, 100])
    np.testing.assert_allclose(site_means(raw, Rectangle(2, 2, 6, 6)), [300, 200, 100])
    with pytest.raises(ValueError, match="isn't a colour-filter mosaic"):
        site_means(Image(np.ones((4, 4))))


def test_bad_fits_are_explained():
    cam = camera()
    training = [light(g) for g in PRIMARIES]
    measured, reference = cam.responses(training), XYZ(training)
    with pytest.raises(ValueError, match="at least 3 patches"):
        fit_color_matrix(measured[:2], reference[:2])
    with pytest.raises(ValueError, match="don't span"):
        fit_color_matrix(np.ones((4, 3)), np.ones((4, 3)))
    with pytest.raises(ValueError, match="unknown method"):
        fit_color_matrix(measured, reference, method="magic")


def test_sensitivity_plot():
    ax = camera().normalized().plot()
    assert [line.get_label() for line in ax.get_lines()] == ["R", "G", "B"]
    assert ax.get_ylabel() == "Sensitivity"
