import numpy as np
import pytest

from gsmath import (
    Calibration, Collection, Measurement, PerExposure, Responsivity, Spectrum, StrayLight,
    SubtractDark,
)
from gsmath.straylight import StrayLightCorrection
from gsmath.units import Q_, parse

WAVELENGTH = np.array([400.0, 401.0, 402.0, 403.0])


def counts(values, label="raw"):
    return Spectrum(WAVELENGTH, np.asarray(values, dtype=float), value_unit="count", label=label)


@pytest.fixture
def measurement():
    return Measurement(raw=counts([110, 210, 310, 410]), dark=counts([10, 10, 10, 10], "dark"),
                       exposure=Q_(100, "ms"), label="ch01_l10",
                       settings={"averages": 1}, instrument={"model": "example-spectrometer"})


@pytest.fixture
def calibration():
    responsivity = Spectrum(WAVELENGTH, [2.0, 2.0, 4.0, 4.0], value_unit="uW/cm^2/sr/nm * ms / count")
    return Calibration("test", [SubtractDark(), PerExposure(), Responsivity(responsivity)])


def test_process_applies_the_steps_in_order_with_units(measurement, calibration):
    spectrum = measurement.process(calibration)
    np.testing.assert_allclose(spectrum.values, [2.0, 4.0, 12.0, 16.0])  # (raw-dark)/100 ms x r
    assert spectrum.to_value_unit("uW/cm^2/sr/nm") is not None
    assert spectrum.label == "ch01_l10"
    processing = spectrum.metadata["processing"]
    assert processing["applied"] == ["SubtractDark", "PerExposure", "Responsivity"]
    assert processing["complete"] and processing["fingerprint"] == calibration.fingerprint


def test_a_missing_input_raises_unless_the_step_is_optional(calibration):
    no_dark = Measurement(raw=counts([1, 2, 3, 4]), exposure=Q_(1, "ms"))
    with pytest.raises(ValueError, match="SubtractDark needs a dark"):
        no_dark.process(calibration)
    optional = Calibration("optional dark", [SubtractDark(optional=True), PerExposure()])
    spectrum = no_dark.process(optional)
    assert spectrum.metadata["processing"]["skipped"] == ["SubtractDark"]
    assert spectrum.metadata["processing"]["complete"] is False


def test_until_stops_partway(measurement, calibration):
    partial = measurement.process(calibration, until="SubtractDark")
    np.testing.assert_allclose(partial.values, [100, 200, 300, 400])
    assert partial.metadata["processing"]["complete"] is False
    with pytest.raises(ValueError, match="no step"):
        measurement.process(calibration, until="FlatField")


def test_stray_light_step():
    correction = StrayLightCorrection(WAVELENGTH, np.eye(4) * 0.5)
    spectrum = Measurement(raw=counts([2, 4, 6, 8])).process(Calibration("slc", [StrayLight(correction)]))
    np.testing.assert_allclose(spectrum.values, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="wavelength grid"):
        StrayLight(StrayLightCorrection(WAVELENGTH + 1, np.eye(4))).apply(counts([1, 1, 1, 1]), None)


def test_fingerprint_follows_the_contents():
    a = Calibration("same", [Responsivity(2.0)])
    assert a.fingerprint == Calibration("same", [Responsivity(2.0)]).fingerprint
    assert a.fingerprint != Calibration("same", [Responsivity(2.5)]).fingerprint
    assert a.fingerprint != Calibration("same", [SubtractDark(), Responsivity(2.0)]).fingerprint


def test_fingerprint_updates_after_step_mutation():
    calibration = Calibration("mutable", [Responsivity(2.0)])
    before = calibration.fingerprint
    calibration.steps[0].responsivity = 3.0
    assert calibration.fingerprint != before
    assert calibration.fingerprint == Calibration("mutable", [Responsivity(3.0)]).fingerprint


def test_measurements_check_their_inputs():
    with pytest.raises(TypeError, match="same kind"):
        Measurement(raw=counts([1, 2, 3, 4]), dark=np.zeros(4))
    with pytest.raises(TypeError, match="quantity"):
        Measurement(raw=counts([1, 2, 3, 4]), exposure=100)


def test_write_and_read_round_trip(tmp_path, measurement, calibration):
    path = measurement.write(tmp_path / "m.gsmath")
    again = Measurement.read(path)
    assert again.exposure == Q_(100, "ms") and again.label == "ch01_l10"
    assert again.settings == {"averages": 1}
    assert again.instrument == {"model": "example-spectrometer"}
    np.testing.assert_array_equal(again.process(calibration).values,
                                  measurement.process(calibration).values)
    with pytest.raises(ValueError, match="Not a 'gsmath.spectrum' file"):
        Spectrum.read(path)


def test_a_collection_of_measurements_processes_to_a_collection_of_spectra(calibration):
    runs = Collection([Measurement(raw=counts([110 + i, 210, 310, 410]), dark=counts([10] * 4),
                                   exposure=Q_(100, "ms"), label=f"m{i}") for i in range(3)])
    spectra = runs.map(lambda m: m.process(calibration))
    assert list(spectra.labels) == ["m0", "m1", "m2"]
    assert isinstance(spectra[0], Spectrum)


def test_a_calibration_saves_and_loads_with_its_data(tmp_path, calibration):
    full = Calibration("full", [
        SubtractDark(optional=True),
        StrayLight(StrayLightCorrection(WAVELENGTH, np.eye(4, dtype=np.float32) * 0.5)),
        PerExposure(),
        *calibration.steps[2:],
        Responsivity(Q_(3, "1/sr")),
    ], metadata={"sources": {"cal": "Amplitude.cal"}})
    path = full.write(tmp_path / "calibrations" / f"{full.fingerprint}.gsmath")
    again = Calibration.read(path)
    assert again.fingerprint == full.fingerprint and again.name == "full"
    assert [s.name for s in again.steps] == [s.name for s in full.steps]
    assert again.steps[0].optional and again.metadata == full.metadata
    m = Measurement(raw=counts([110, 210, 310, 410]), dark=counts([10] * 4), exposure=Q_(100, "ms"))
    np.testing.assert_array_equal(m.process(again).values, m.process(full).values)
    assert m.process(again).value_unit == m.process(full).value_unit


def test_a_tampered_calibration_file_is_refused(tmp_path, calibration):
    path = calibration.write(tmp_path / "c.gsmath")
    path.write_text(path.read_text().replace('"name": "test"', '"name": "edited"'))
    with pytest.raises(ValueError, match="don't match the saved fingerprint"):
        Calibration.read(path)


def test_custom_steps_register_themselves(tmp_path):
    from gsmath.calibration import STEP_TYPES, Step

    class Halve(Step):
        def apply(self, data, measurement):
            return data / 2

    assert STEP_TYPES["Halve"] is Halve
    path = Calibration("halve", [Halve()]).write(tmp_path / "h.gsmath")
    spectrum = Measurement(raw=counts([2, 4, 6, 8])).process(Calibration.read(path))
    np.testing.assert_allclose(spectrum.values, [1, 2, 3, 4])


def test_a_measurement_refers_to_its_calibration(tmp_path, measurement, calibration):
    tagged = Measurement(raw=measurement.raw, dark=measurement.dark, exposure=measurement.exposure,
                         calibration=calibration.reference)
    again = Measurement.read(tagged.write(tmp_path / "m.gsmath"))
    assert again.calibration == {"name": "test", "fingerprint": calibration.fingerprint}


def test_gsmath_read_returns_whatever_a_file_holds(tmp_path, measurement, calibration):
    import gsmath

    spectrum_path = counts([1, 2, 3, 4]).write(tmp_path / "s.gsmath")
    measurement_path = measurement.write(tmp_path / "m.gsmath")
    calibration_path = calibration.write(tmp_path / "c.gsmath")
    assert isinstance(gsmath.read(spectrum_path), Spectrum)
    assert isinstance(gsmath.read(measurement_path), Measurement)
    loaded = gsmath.read(calibration_path)
    assert isinstance(loaded, Calibration) and loaded.fingerprint == calibration.fingerprint

    counts([1, 2, 3, 4]).write(tmp_path / "s.csv")
    assert isinstance(gsmath.read(tmp_path / "s.csv"), Spectrum)

    (tmp_path / "x.gsmath").write_text('{"format": "gsmath.hologram", "version": 1}')
    with pytest.raises(ValueError, match="unknown gsmath record 'gsmath.hologram'"):
        gsmath.read(tmp_path / "x.gsmath")


def test_sensor_temperature_and_per_channel_exposure(tmp_path):
    from gsmath import Image

    frames = Image(np.stack([np.full((2, 3), 100.0), np.full((2, 3), 400.0),
                             np.full((2, 3), 200.0)], axis=-1),
                   value_unit="count", channels=("X", "Y", "Z"))
    measurement = Measurement(raw=frames, exposure=Q_([10.0, 40.0, 20.0], "ms"),
                              temperature=Q_(31.5, "degC"))
    rate = measurement.process(Calibration("rate", [PerExposure()]))
    np.testing.assert_allclose(rate.values[0, 0], [10.0, 10.0, 10.0])
    assert rate.value_unit == parse("count/ms")
    again = Measurement.read(measurement.write(tmp_path / "m.gsmath"))
    np.testing.assert_allclose(again.exposure.to("ms").magnitude, [10, 40, 20])
    assert again.temperature.to("K").magnitude == pytest.approx(304.65)
    assert Measurement.read(Measurement(raw=counts([1, 2, 3, 4])).write(
        tmp_path / "plain.gsmath")).temperature is None
    with pytest.raises(ValueError, match="as many channels"):
        Measurement(raw=frames, exposure=Q_([10.0, 40.0], "ms")).process(
            Calibration("rate", [PerExposure()]))
    with pytest.raises(TypeError, match="temperature quantity"):
        Measurement(raw=frames, temperature=Q_(5, "ms"))
    with pytest.raises(TypeError, match="temperature quantity"):
        Measurement(raw=frames, temperature=31.5)
