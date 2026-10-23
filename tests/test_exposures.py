import numpy as np
import pytest

from gsmath import (
    Calibration, ExposureSeries, Measurement, PerExposure, Responsivity, Spectrum, StrayLight,
    SubtractDark, read,
)
from gsmath.straylight import StrayLightCorrection
from gsmath.units import Q_

WAVELENGTH = np.array([400.0, 401.0, 402.0, 403.0])
OFFSET = 100.0
FULL_SCALE = 1000.0
RATE = np.array([0.5, 2.0, 5.0, 20.0])  # counts per ms


def counts(values, label="raw"):
    return Spectrum(WAVELENGTH, np.asarray(values, dtype=float), value_unit="count", label=label)


def exposure(ms, rate=RATE):
    """A reading at `ms`: the offset plus rate x time, clipped at full scale."""
    raw = np.minimum(OFFSET + rate * ms, FULL_SCALE)
    return Measurement(raw=counts(raw), dark=counts(np.full(4, OFFSET), "dark"),
                       exposure=Q_(ms, "ms"))


def series(*times, rate=RATE):
    return ExposureSeries([exposure(t, rate) for t in times], saturation=FULL_SCALE,
                          label="hdr", settings={"multiple": 2})


@pytest.fixture
def calibration():
    return Calibration("test", [SubtractDark(), PerExposure(), Responsivity(3.0)])


def test_combined_rate_is_the_exposure_weighted_mean_of_unsaturated_exposures(calibration):
    # Raw reaches 1000 from 1800/450/180/45 ms on: the last pixel saturates at 160 ms.
    hdr = series(10, 40, 160)
    spectrum = hdr.process(calibration)
    np.testing.assert_allclose(spectrum.values, RATE * 3.0)
    report = spectrum.metadata["processing"]["reports"]["ExposureSeries"]
    assert report["elements_by_exposures_used"] == {"0": 0, "1": 0, "2": 1, "3": 3}
    assert report["saturated_in_all"] == 0
    assert report["exposures"] == [10.0, 40.0, 160.0] and report["exposure_unit"] == "ms"
    used = hdr.exposures_used()  # the map, shaped like the data
    assert isinstance(used, Spectrum) and used.value_unit is None
    np.testing.assert_array_equal(used.values, [3, 3, 3, 2])


def test_weighting_is_summed_counts_over_summed_exposure(calibration):
    noisy = ExposureSeries([
        Measurement(raw=counts([110, 110, 110, 110]), dark=counts(np.zeros(4)), exposure=Q_(10, "ms")),
        Measurement(raw=counts([430, 430, 430, 430]), dark=counts(np.zeros(4)), exposure=Q_(40, "ms")),
    ], saturation=FULL_SCALE)
    spectrum = noisy.process(Calibration("rate", [SubtractDark(), PerExposure()]))
    np.testing.assert_allclose(spectrum.values, (110 + 430) / 50)


def test_a_pixel_saturated_everywhere_takes_the_shortest_exposure_and_is_flagged(calibration):
    rate = np.array([0.5, 2.0, 5.0, 200.0])  # the last pixel saturates even at 10 ms
    hdr = series(10, 40, rate=rate)
    spectrum = hdr.process(calibration)
    report = spectrum.metadata["processing"]["reports"]["ExposureSeries"]
    assert report["saturated_in_all"] == 1
    np.testing.assert_array_equal(hdr.exposures_used().values, [2, 2, 2, 0])
    assert spectrum.values[3] == pytest.approx((FULL_SCALE - OFFSET) / 10 * 3.0)


def test_without_a_saturation_level_every_exposure_is_used(calibration):
    unbounded = ExposureSeries([exposure(10), exposure(20)])  # a spectrum records no white level
    assert unbounded.saturation == "auto" and unbounded.saturation_level is None
    spectrum = unbounded.process(calibration)
    report = spectrum.metadata["processing"]["reports"]["ExposureSeries"]
    assert report["elements_by_exposures_used"]["2"] == 4 and report["saturation"] is None
    assert ExposureSeries([exposure(10)], saturation=None).saturation_level is None


def test_exposures_combine_before_stray_light_sees_saturation():
    # A matrix that moves every pixel's light into pixel 0: fed a saturated
    # reading, pixel 0 would come out low; fed the combined one, exactly right.
    mixing = np.zeros((4, 4))
    mixing[0, :] = 1.0
    correction = StrayLightCorrection(WAVELENGTH, mixing)
    cal = Calibration("slc", [SubtractDark(), StrayLight(correction), PerExposure()])
    spectrum = series(10, 160).process(cal)
    assert spectrum.values[0] == pytest.approx(RATE.sum())


def test_combined_data_is_counts_at_the_longest_exposure(calibration):
    combined = series(160, 10, 40).process(calibration, until="SubtractDark")
    np.testing.assert_allclose(combined.values, RATE * 160)
    assert combined.metadata["processing"]["applied"] == ["SubtractDark"]
    assert combined.metadata["processing"]["complete"] is False


def test_processing_record_reads_as_one_pass(calibration):
    spectrum = series(10, 40).process(calibration)
    processing = spectrum.metadata["processing"]
    assert processing["applied"] == ["SubtractDark", "PerExposure", "Responsivity"]
    assert processing["complete"] is True
    assert processing["combined_after"] == "SubtractDark"
    assert processing["fingerprint"] == calibration.fingerprint
    assert spectrum.label == "hdr"


def test_a_skipped_step_before_combining_is_recorded():
    no_dark = ExposureSeries([Measurement(raw=counts(RATE * t), exposure=Q_(t, "ms"))
                              for t in (10, 20)])
    cal = Calibration("optional", [SubtractDark(optional=True), PerExposure()])
    spectrum = no_dark.process(cal)
    np.testing.assert_allclose(spectrum.values, RATE)
    assert spectrum.metadata["processing"]["skipped"] == ["SubtractDark"]
    assert spectrum.metadata["processing"]["complete"] is False


def test_process_checks_its_step_names(calibration):
    with pytest.raises(ValueError, match="no step 'FlatField' to combine after"):
        series(10).process(calibration, combine_after="FlatField")
    with pytest.raises(ValueError, match="does not come"):
        series(10).process(Calibration("c", [PerExposure(), SubtractDark()]), until="PerExposure")


def test_series_check_their_measurements():
    with pytest.raises(ValueError, match="at least one"):
        ExposureSeries([])
    with pytest.raises(ValueError, match="exposure time"):
        ExposureSeries([Measurement(raw=counts(RATE))])
    with pytest.raises(ValueError, match="same shape"):
        ExposureSeries([exposure(10), Measurement(raw=Spectrum([1.0, 2.0], [1.0, 2.0]),
                                                  exposure=Q_(1, "ms"))])


def test_write_and_read_back(tmp_path, calibration):
    original = series(10, 40, 160)
    path = original.write(tmp_path / "hdr.gsmath")
    for loaded in (ExposureSeries.read(path), read(path)):
        assert isinstance(loaded, ExposureSeries)
        assert loaded.saturation == FULL_SCALE and loaded.settings == {"multiple": 2}
        assert [float(e.to("ms").magnitude) for e in loaded.exposures] == [10, 40, 160]
        np.testing.assert_allclose(loaded.process(calibration).values,
                                   original.process(calibration).values)


def test_calibration_process_can_start_after_a_step(calibration):
    reading = Measurement(raw=counts([100, 200, 300, 400]), exposure=Q_(100, "ms"))
    spectrum = calibration.process(reading, after="SubtractDark")
    np.testing.assert_allclose(spectrum.values, [3.0, 6.0, 9.0, 12.0])
    assert spectrum.metadata["processing"]["applied"] == ["PerExposure", "Responsivity"]
    assert spectrum.metadata["processing"]["after"] == "SubtractDark"
    assert spectrum.metadata["processing"]["complete"] is False
    with pytest.raises(ValueError, match="no step"):
        calibration.process(reading, after="FlatField")
    with pytest.raises(ValueError, match="does not come after"):
        calibration.process(reading, after="PerExposure", until="SubtractDark")


# Images: the same combining, on a camera's raw frames.
FRAME_RATE = np.array([[10.0, 100.0], [1000.0, 5000.0]])  # counts per ms
WHITE = 4095.0


def raw_frame(ms, rate=FRAME_RATE):
    from gsmath import Image
    from gsmath.mosaic import Mosaic

    return Image(np.minimum(rate * ms + 50, WHITE), value_unit="count",
                 mosaic=Mosaic("RGGB", white_level=WHITE))


def camera_series(*times, rate=FRAME_RATE):
    return ExposureSeries([Measurement(raw=raw_frame(t, rate), exposure=Q_(t, "ms"))
                           for t in times])


def camera_calibration():
    from gsmath import DarkModel, Image, MaskSaturated

    dark = Image(np.full((2, 2), 50.0), value_unit="count")
    model = DarkModel.fit([dark, dark], [Q_(1, "ms"), Q_(2, "ms")])
    return Calibration("camera", [MaskSaturated(), model.step(), PerExposure()])


def test_raw_frames_combine_like_spectra():
    hdr = camera_series(0.5, 2, 8)
    assert hdr.saturation_level == pytest.approx(0.98 * WHITE)  # from the raw file
    rate = hdr.process(camera_calibration())  # combines after SubtractDarkModel
    np.testing.assert_allclose(rate.values, FRAME_RATE)
    assert rate.mosaic.pattern == "RGGB"
    assert rate.metadata["processing"]["combined_after"] == "SubtractDarkModel"
    np.testing.assert_array_equal(hdr.exposures_used().values, [[3, 3], [2, 1]])


def test_pixels_masked_as_saturated_are_left_out_not_spread():
    # MaskSaturated makes the clipped readings NaN before combining: the
    # combination skips them rather than turning the pixel NaN.
    rate = camera_series(0.5, 2, 8).process(camera_calibration())
    assert np.all(np.isfinite(rate.values))
    # Saturated in every exposure: the shortest one's value, NaN once masked.
    hot = FRAME_RATE.copy()
    hot[1, 1] = 1e5
    result = camera_series(0.5, 2, rate=hot).process(camera_calibration())
    assert np.isnan(result.values[1, 1]) and np.isfinite(result.values[0, 0])
    report = result.metadata["processing"]["reports"]["ExposureSeries"]
    assert report["saturated_in_all"] == 1


def test_the_report_does_not_grow_with_the_frame():
    from gsmath import Image
    import json

    big = ExposureSeries([Measurement(raw=Image(np.full((300, 400), 100.0), value_unit="count"),
                                      dark=Image(np.zeros((300, 400)), value_unit="count"),
                                      exposure=Q_(t, "ms")) for t in (1, 2)])
    result = big.process(Calibration("c", [SubtractDark(), PerExposure()]))
    report = result.metadata["processing"]["reports"]["ExposureSeries"]
    assert len(json.dumps(report)) < 300
    assert big.exposures_used().shape == (300, 400)


def test_combine_after_defaults_to_the_dark_step():
    with pytest.raises(ValueError, match="no dark subtraction"):
        series(10).process(Calibration("c", [PerExposure()]))
    with pytest.raises(TypeError, match="raw value, None or 'auto'"):
        ExposureSeries([exposure(10)], saturation="high")


def test_image_series_write_and_read_back(tmp_path):
    hdr = camera_series(0.5, 2, 8)
    path = hdr.write(tmp_path / "hdr.gsmath")
    loaded = read(path)
    assert loaded.saturation == "auto" and loaded.measurements[0].raw.mosaic.pattern == "RGGB"
    np.testing.assert_allclose(loaded.process(camera_calibration()).values, FRAME_RATE)


def test_any_step_that_removes_the_dark_is_where_a_series_combines():
    from gsmath.calibration import Step

    class SubtractPedestal(Step):  # a lab's own dark step, by any name
        removes_dark = True

        def apply(self, data, measurement):
            return data - Q_(OFFSET, "count")

    cal = Calibration("own", [SubtractPedestal(), PerExposure()])
    spectrum = series(10, 40, 160).process(cal)
    np.testing.assert_allclose(spectrum.values, RATE)
    assert spectrum.metadata["processing"]["combined_after"] == "SubtractPedestal"
