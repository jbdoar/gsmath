"""Series, and what every kind of sampled data shares (gsmath.sampled)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gsmath import Image, LineCut, PixelMap, Sampled, Series, SpectralSensitivity, Spectrum
from gsmath.units import Q_, parse

T = np.arange(5.0)


def rgb_series():
    return Series(T, np.column_stack([T, 2 * T, 3 * T]), label="rgb", x_unit="s",
                  value_unit="count", channels=("R", "G", "B"))


def test_kinds_of_sampled_data():
    spectrum = Spectrum([400, 500], [1.0, 2.0])
    assert isinstance(spectrum, Series) and isinstance(spectrum, Sampled)
    assert spectrum.axis is spectrum.wavelength and spectrum.axis_unit == parse("nm")
    assert spectrum.channels is None and spectrum.n_channels == 1
    assert isinstance(Image(np.zeros((2, 2))), Sampled) and not isinstance(Image(np.zeros((2, 2))), Series)
    cut = LineCut([0.0, 1.0], [5.0, 6.0], distance_unit="mm")
    assert cut.axis is cut.distance and cut.axis_unit == parse("mm")
    sensitivity = SpectralSensitivity([400.0, 500.0], np.ones((2, 3)))
    assert sensitivity.axis is sensitivity.wavelength and sensitivity.n_channels == 3


def test_a_series_checks_its_shape_and_channels():
    series = rgb_series()
    assert len(series) == 5 and series.n_channels == 3 and series.x_unit == parse("s")
    assert Series(T, np.ones((5, 2))).channels == ("0", "1")
    with pytest.raises(ValueError, match="one row per x"):
        Series(T, np.ones(4))
    with pytest.raises(ValueError, match="2 channel names for 3"):
        Series(T, np.ones((5, 3)), channels=("a", "b"))
    with pytest.raises(ValueError, match="one-dimensional"):
        Series(np.ones((5, 2)), np.ones(5))


def test_channels():
    series = rgb_series()
    green = series["G"]
    assert green.values.ndim == 1 and green.channels == ("G",) and type(green) is Series
    np.testing.assert_allclose(green.values, 2 * T)
    assert series[["B", "R"]].channels == ("B", "R")
    with pytest.raises(KeyError, match="no channel 'Y'"):
        series["Y"]
    with pytest.raises(TypeError, match="by channel name"):
        series[0]


def test_series_arithmetic():
    series = rgb_series()
    gain = Series(T, np.full(5, 2.0), x_unit="s")  # one channel scales them all
    doubled = series * gain
    np.testing.assert_allclose(doubled.values, 2 * series.values)
    assert doubled.channels == ("R", "G", "B") and doubled.value_unit == parse("count")
    assert (gain * series).channels == ("R", "G", "B")
    np.testing.assert_allclose((series - series).values, 0)
    assert (series / Q_(2, "s")).value_unit == parse("count/s")
    with pytest.raises(ValueError, match="X arrays must match"):
        series + Series(T + 1, np.ones(5), x_unit="s")
    with pytest.raises(ValueError, match="X units must match"):
        series + Series(T, np.ones(5), x_unit="ms")
    with pytest.raises(ValueError, match="Channels must match"):
        series + series[["R", "G", "B"]].copy(channels=("X", "Y", "Z"))
    with pytest.raises(TypeError):
        Spectrum(T, np.ones(5)) + Series(T, np.ones(5))  # different axes


def test_integer_data_becomes_float_in_arithmetic():
    for data in (Spectrum([1, 2], np.array([1, 2], dtype=np.uint16)),
                 Series([1, 2], np.array([1, 2], dtype=np.uint16)),
                 Image(np.array([[1, 2]], dtype=np.uint16))):
        assert (data - 3).values.dtype == np.float64
        assert np.all((data - 3).values < 0)


def test_copy_none_removes_an_optional_field_everywhere():
    spectrum = Spectrum([1, 2], [1.0, 2.0], value_unit="W")
    image = Image(np.ones((2, 2)), value_unit="W")
    series = rgb_series()
    for data in (spectrum, image, series):
        assert data.copy().value_unit == parse(data.value_unit)
        assert data.copy(value_unit=None).value_unit is None
        assert data.copy(label=None).label == data.label  # required fields keep
    np.testing.assert_array_equal(spectrum.copy(wavelength=None).wavelength, [1, 2])
    assert series.copy(channels=None, values=np.ones(5)).channels is None
    with pytest.raises(TypeError, match="unknown field"):
        spectrum.copy(channels=("a",))


def test_trim_and_interpolate_keep_a_line_cuts_positions():
    image = Image(np.arange(30.0).reshape(3, 10), pixel_map=PixelMap.linear((3, 10), 0.5,
                                                                             unit="mm"))
    cut = image.row(1)
    part = cut.trim(1.0, 3.0)
    np.testing.assert_allclose(part.distance, [1.0, 1.5, 2.0, 2.5, 3.0])
    np.testing.assert_allclose(part.x, cut.x[2:7])
    assert isinstance(part, LineCut) and part.distance_unit == parse("mm")
    finer = cut.interpolate([0.25, 0.75])
    np.testing.assert_allclose(finer.values, [10.5, 11.5])
    np.testing.assert_allclose(finer.x, cut.x[0] + np.array([0.25, 0.75]))
    np.testing.assert_allclose(rgb_series().interpolate([0.5, 9])["G"].values, [1.0, np.nan])


def test_a_line_cut_finds_a_spots_width():
    # A Gaussian spot, 0.6 deg FWHM, centred at x = 0.3 deg.
    pixel_map = PixelMap.linear((21, 201), 0.02, unit="deg")
    x, y = pixel_map.grid()
    spot = np.exp(-4 * np.log(2) * ((x - 0.3) ** 2 + y ** 2) / 0.6 ** 2)
    image = Image(np.stack([spot, 2 * spot, spot], axis=-1), value_unit="cd/m^2",
                  channels=("X", "Y", "Z"), pixel_map=pixel_map)
    cut = image.line((-2, 0), (2, 0), coords=True)
    peaks = cut["Y"].peaks()
    assert peaks.fwhm == pytest.approx(0.6, abs=0.005)
    assert peaks.unit("fwhm") == parse("deg")
    assert cut.x[np.argmax(cut["Y"].values)] == pytest.approx(0.3, abs=0.01)
    assert cut["Y"].stats().max == pytest.approx(2.0, abs=0.01)
    with pytest.raises(ValueError, match="pick one of"):
        cut.peaks()
    assert cut.peaks("Y").fwhm == peaks.fwhm


def test_sampled_data_is_immutable_and_copies_share_arrays():
    series = Series(T, np.array([0.0, 1, 3, 1, 0]))
    assert series.peaks() is series.peaks()  # cached for good
    with pytest.raises(ValueError, match="read-only"):
        series.values[:] = 0
    with pytest.raises(AttributeError, match="immutable"):
        series.label = "renamed"
    assert series.copy(label="renamed").values is series.values  # shared, not duplicated
    image = Image(np.zeros((4, 4), dtype=np.uint16), mosaic="RGGB")
    with pytest.raises(ValueError, match="read-only"):
        image.values[0, 0] = 1
    with pytest.raises(AttributeError, match="immutable"):
        image.mosaic = None
    assert image.copy(label="x").values is image.values
    assert image.stats() is image.stats()
    mine = np.ones((2, 2))
    Image(mine)
    mine[0, 0] = 5  # the caller's own array stays writable


def test_spectral_sensitivity_is_a_series():
    sensitivity = SpectralSensitivity([400.0, 500.0, 600.0], np.eye(3), value_unit="count/J")
    assert isinstance(sensitivity.channel("G"), Spectrum)
    assert sensitivity.channel(["G", "B"]).channels == ("G", "B")
    doubled = sensitivity * 2
    assert isinstance(doubled, SpectralSensitivity) and doubled.channels == ("R", "G", "B")
    np.testing.assert_allclose(doubled.values, 2 * np.eye(3))
    with pytest.raises(ValueError, match="wavelengths must increase"):
        SpectralSensitivity([500.0, 400.0], np.ones((2, 3)))


def test_series_plot():
    ax = rgb_series().plot()
    assert ax.get_xlabel() == "x (s)" and len(ax.get_lines()) == 3
    assert [line.get_linestyle() for line in ax.get_lines()][:2] == ["-", "--"]
    sensitivity = SpectralSensitivity([400.0, 500.0], np.ones((2, 3)), value_unit="count/J")
    ax = sensitivity.plot()
    assert ax.get_xlabel() == "Wavelength (nm)" and ax.get_ylabel().startswith("Sensitivity (")


def test_characterisation_curves_are_series():
    from gsmath import Linearity

    t = np.arange(1.0, 6.0)
    linearity = Linearity.fit(Q_(t, "ms"), np.column_stack([10 * t, 20 * t]), channels=("R", "G"))
    curve = linearity.curve
    assert isinstance(curve, Series) and curve.channels == ("R", "G")
    assert curve.x_unit == parse("ms") and curve.plot().get_xlabel() == "Stimulus (ms)"
    np.testing.assert_allclose(curve["G"].values, 20 * t)


def spot_cut(width, label):
    x = np.linspace(-2, 2, 401)
    spot = np.exp(-4 * np.log(2) * x ** 2 / width ** 2)
    return LineCut(x - x[0], np.column_stack([spot, 2 * spot]), label=label, distance_unit="deg",
                   value_unit="cd/m^2", channels=("X", "Y"))


def test_series_summarise_compare_and_check():
    cut = spot_cut(0.6, "a")
    summary = cut.summary()
    assert set(summary) == {"X", "Y"}
    assert summary["Y"]["fwhm"] == pytest.approx(0.6, abs=0.01)
    assert summary["Y"]["peak"] == pytest.approx(2.0, abs=0.01)
    assert summary["Y"]["peak_value"] == pytest.approx(2.0)
    units = cut.summary_units()
    assert units["Y"]["fwhm"] == parse("deg") and units["Y"]["max"] == parse("cd/m^2")
    assert cut.check({"Y": {"fwhm": {"maximum": 0.5}}}).failed == ("Y.fwhm",)
    assert cut.compare(spot_cut(0.66, "ref")).records["Y.fwhm"].difference == pytest.approx(
        -0.06, abs=0.01)
    single = Series(T, np.array([0.0, 1, 3, 1, 0]), x_unit="s")
    assert single.summary()["peak"] == pytest.approx(2.0)
    assert Spectrum([500.0, 510], [1.0, 2.0]).profile().name == "spectrum"


def test_collections_of_any_kind():
    from gsmath import Collection, to_dataframe
    from gsmath.comparison import check, compare

    cuts = Collection([spot_cut(0.6, "a"), spot_cut(0.8, "b")])
    frame = to_dataframe(cuts, include_units=True)
    assert list(frame["label"]) == ["a", "b"] and "Y.fwhm [deg]" in frame.columns
    limits = cuts[0].profile().with_limits({"Y": {"fwhm": {"maximum": 0.7}}})
    assert [c.passed for c in check(cuts, limits)] == [True, False]
    tolerance = cuts[0].profile().with_limits({"Y": {"fwhm": {"tolerance": 0.1}}})
    assert [c.passed for c in compare(cuts, spot_cut(0.62, "ref"), profile=tolerance)] \
        == [True, False]
    sensitivities = Collection([
        SpectralSensitivity([400.0, 500.0, 600.0], np.eye(3) * k, label=f"camera {k}")
        for k in (1, 2)])
    assert to_dataframe(sensitivities)["G.peak_value"].tolist() == [1.0, 2.0]


def test_collection_convenience_methods():
    from gsmath import Collection

    cuts = Collection([spot_cut(0.6, "a"), spot_cut(0.8, "b")])
    assert cuts.summary()["b"]["Y"]["fwhm"] == pytest.approx(0.8, abs=0.01)
    assert cuts.summary_units()["a"]["Y"]["fwhm"] == parse("deg")
    checks = cuts.check({"Y": {"fwhm": {"maximum": 0.7}}})
    assert checks.passed is False and checks.failed == ("b",)
    assert checks["a"].passed and checks["b"].failed == ("Y.fwhm",)
    assert list(checks.to_dataframe()["passed"]) == [True, False]
    compared = cuts.compare(spot_cut(0.62, "ref"), {"Y": {"fwhm": {"tolerance": 0.1}}})
    assert compared.failed == ("b",)
    assert cuts.compare(spot_cut(0.62, "ref")).passed is None  # nothing judged
    paired = cuts.compare(Collection([spot_cut(0.8, "b"), spot_cut(0.6, "a")]),
                          {"Y": {"fwhm": {"tolerance": 0.01}}})
    assert paired.passed is True  # paired by label, not position
    frame = cuts.to_dataframe(include_units=True)
    assert "Y.fwhm [deg]" in frame.columns and cuts.to_dict()["a"]["X"]["peak_value"] == 1.0


def test_collections_of_results_and_spectra():
    from gsmath import Collection

    wl = np.arange(500.0, 600)
    spectra = Collection([Spectrum(wl, np.exp(-4 * np.log(2) * ((wl - 550) / w) ** 2),
                                   label=f"{w} nm", value_unit="W/nm") for w in (20, 30)])
    assert spectra.check({"peaks": {"fwhm": {"maximum": 25}}}).failed == ("30 nm",)
    images = Collection([
        Image(np.full((60, 80, 3), level), label=f"{level}", value_unit="cd/m^2",
              channels=("X", "Y", "Z")) for level in (100.0, 80.0)])
    uniformity = Collection([image.uniformity() for image in images])
    assert uniformity.check({"luminance": {"mean": {"minimum": 90}}}).failed == ("80.0",)
    assert images.check({"Y": {"mean": {"minimum": 90}}}).failed == ("80.0",)
    with pytest.raises(ValueError, match="empty collection"):
        Collection([]).profile()
    with pytest.raises(TypeError, match="no default profile"):
        Collection([uniformity["100.0"].check({"luminance": {"mean": {}}})]).profile()


def test_series_resample_like_spectra():
    x = np.linspace(0.0, 4.0, 9)
    curves = Series(x, np.column_stack([x ** 2, 2 * x ** 2]), x_unit="mm", channels=("a", "b"))
    cubic = curves.interpolate(np.array([0.3, 1.7]), spline_order=3)
    np.testing.assert_allclose(cubic.values, np.column_stack([[0.09, 2.89], [0.18, 5.78]]))
    regular = curves.interpolate(step=0.25, start=1.0, stop=2.0)
    np.testing.assert_allclose(regular.x, [1.0, 1.25, 1.5, 1.75, 2.0])
    assert regular.label.endswith("[interpolated]") and regular.channels == ("a", "b")
    np.testing.assert_allclose(curves(2.0), [4.0, 8.0])           # one value per channel
    np.testing.assert_allclose(curves([3.0, 1.0])[:, 1], [18.0, 2.0])  # any order
    assert np.isnan(curves(5.0)).all() and curves(5.0, extrapolate=True)[0] > 16
    with pytest.raises(ValueError, match="strictly increasing"):
        curves.interpolate([2.0, 1.0])
    assert curves.trim(1.0, 2.0).x.tolist() == [1.0, 1.5, 2.0]
