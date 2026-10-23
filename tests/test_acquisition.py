import matplotlib

matplotlib.use("Agg")

import warnings

import numpy as np
import pytest

from gsmath import (
    Calibration, DarkModel, FlatField, Image, Linearity, Linearize, MaskSaturated, Measurement,
    Rectangle, build_flat, site_means,
)
from gsmath.mosaic import Mosaic, site_masks
from gsmath.units import Q_, parse

SHAPE = (8, 10)
BIAS = 100 + np.arange(80, dtype=float).reshape(SHAPE) % 7
RATE = np.full(SHAPE, 20.0)
RATE[2, 3] = 5000.0  # a hot pixel


def dark_frame(ms, noise=0.0, seed=0):
    values = BIAS + RATE * ms / 1000 + np.random.default_rng(seed).normal(0, noise, SHAPE)
    return Image(values, value_unit="count", mosaic="RGGB")


def test_a_dark_model_recovers_bias_and_rate():
    exposures = [Q_(t, "ms") for t in (10, 50, 200)]
    model = DarkModel.fit([dark_frame(t.magnitude) for t in exposures], exposures)
    np.testing.assert_allclose(model.bias.values, BIAS)
    np.testing.assert_allclose(model.rate.values, RATE)
    assert model.rate.value_unit == parse("count/s") and model.residual_rms < 1e-9
    np.testing.assert_allclose(model.dark(Q_(0.1, "s")).values, BIAS + RATE * 0.1)
    assert model.hot_pixels().array.sum() == 1 and model.hot_pixels().array[2, 3]
    assert model.read_noise is None  # no repeated frames
    assert model.bias.mosaic.pattern == "RGGB"


def test_read_noise_from_repeated_frames():
    frames, exposures = [], []
    for t in (10, 100):
        for k in range(40):
            frames.append(dark_frame(t, noise=3.0, seed=100 * t + k))
            exposures.append(Q_(t, "ms"))
    assert DarkModel.fit(frames, exposures).read_noise == pytest.approx(3.0, rel=0.1)


def test_measurements_carry_their_exposures_and_the_step_subtracts(tmp_path):
    runs = [Measurement(raw=dark_frame(t), exposure=Q_(t, "ms")) for t in (10, 40, 160)]
    model = DarkModel.fit(runs)
    light = Measurement(raw=dark_frame(80) + Q_(500, "count"), exposure=Q_(80, "ms"))
    pipeline = Calibration("dark", [model.step()])
    np.testing.assert_allclose(light.process(pipeline).values, 500.0)
    loaded = Calibration.read(pipeline.write(tmp_path / "dark.gsmath"))
    np.testing.assert_allclose(light.process(loaded).values, 500.0)
    with pytest.raises(ValueError, match="needs an exposure"):
        Measurement(raw=dark_frame(10)).process(pipeline)


def test_dark_model_inputs_are_checked():
    with pytest.raises(ValueError, match="two or more exposures"):
        DarkModel.fit([dark_frame(10), dark_frame(10)], [Q_(10, "ms")] * 2)
    with pytest.raises(TypeError, match="time quantities"):
        DarkModel.fit([dark_frame(10), dark_frame(20)], [10, 20])


SLOPE = 0.4 + 0.1 * (np.arange(80).reshape(SHAPE) % 3)  # counts/K


def warm_dark(ms, celsius, noise=0.0, seed=0, doubling=7.0):
    """A dark at a sensor temperature: bias drifting linearly, the rate
    doubling every `doubling` K, both from their values at 25 degC."""
    rate = RATE * 2 ** ((celsius - 25) / doubling)
    values = BIAS + SLOPE * (celsius - 25) + rate * ms / 1000
    values = values + np.random.default_rng(seed).normal(0, noise, SHAPE)
    return Measurement(raw=Image(values, value_unit="count", mosaic="RGGB"),
                       exposure=Q_(ms, "ms"), temperature=Q_(celsius, "degC"))


def temperature_series(noise=0.0, repeats=1):
    # The temperature drifts a little within each set, as on an uncooled camera.
    return [warm_dark(t, T + 0.1 * k, noise, seed=1000 * T + 10 * t + k)
            for T in (20, 27, 34, 41) for t in (10, 100, 400) for k in range(repeats)]


def test_a_dark_model_across_temperatures_recovers_the_doubling():
    model = DarkModel.fit(temperature_series(), reference=Q_(25, "degC"))
    assert model.temperature_dependent and model.doubling == pytest.approx(7.0, rel=1e-4)
    np.testing.assert_allclose(model.bias.values, BIAS, atol=1e-5)
    np.testing.assert_allclose(model.rate.values, RATE, rtol=1e-5)
    np.testing.assert_allclose(model.bias_slope.values, SLOPE, atol=1e-6)
    assert model.bias_slope.value_unit == parse("count/K")
    assert model.reference_temperature == 25 and model.temperatures == (20, 41.0)
    assert model.residual_rms < 1e-6
    expected = warm_dark(250, 37.5).raw.values
    np.testing.assert_allclose(model.dark(Q_(250, "ms"), Q_(37.5, "degC")).values, expected)
    assert model.rate_at(Q_(32, "degC")) == pytest.approx(40.0, rel=1e-4)
    assert [T for T, _, _ in model.by_temperature] == [20, 27, 34, 41]
    assert model.by_temperature[0][1] == pytest.approx(20 * 2 ** (-5 / 7), rel=0.02)
    assert model.summary["doubling"] == pytest.approx(7.0, rel=1e-4)
    with pytest.raises(ValueError, match="depends on temperature"):
        model.dark(Q_(10, "ms"))


def test_by_default_the_reference_is_the_darks_mean_temperature():
    model = DarkModel.fit(temperature_series())
    assert model.reference_temperature == pytest.approx(30.5)
    np.testing.assert_allclose(model.dark(Q_(100, "ms"), Q_(22, "degC")).values,
                               warm_dark(100, 22).raw.values)


def test_noisy_darks_across_temperatures():
    model = DarkModel.fit(temperature_series(noise=3.0, repeats=4))
    assert model.doubling == pytest.approx(7.0, rel=0.05)
    assert model.read_noise == pytest.approx(3.0, rel=0.1)
    assert model.residual_rms == pytest.approx(3.0, rel=0.1)
    assert model.hot_pixels().array[2, 3] and model.hot_pixels().array.sum() == 1


def test_the_step_uses_each_measurements_exposure_and_temperature(tmp_path):
    model = DarkModel.fit(temperature_series())
    pipeline = Calibration("dark", [model.step()])
    loaded = Calibration.read(pipeline.write(tmp_path / "dark.gsmath"))
    assert loaded.fingerprint == pipeline.fingerprint
    for ms, celsius in ((10, 21), (80, 30.5), (300, 40)):
        dark = warm_dark(ms, celsius)
        light = Measurement(raw=dark.raw + Q_(500, "count"), exposure=dark.exposure,
                            temperature=dark.temperature)
        np.testing.assert_allclose(light.process(pipeline).values, 500.0, atol=1e-4)
        np.testing.assert_allclose(light.process(loaded).values, 500.0, atol=1e-4)
    with pytest.raises(ValueError, match="needs a sensor temperature"):
        Measurement(raw=dark.raw, exposure=Q_(10, "ms")).process(pipeline)
    hot = warm_dark(100, 50)
    with pytest.warns(UserWarning, match="outside the dark model's 20-41 °C: the dark is "
                                         "extrapolated"):
        hot.process(pipeline)


def test_a_dark_model_from_one_temperature_warns_when_used_at_another():
    model = DarkModel.fit([warm_dark(t, 30) for t in (10, 100, 400)])
    assert not model.temperature_dependent and model.reference_temperature == 30
    step = model.step()
    assert step.arguments().keys() == {"bias", "rate", "reference_temperature",
                                       "lowest_temperature", "highest_temperature"}
    pipeline = Calibration("dark", [step])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warm_dark(50, 30.8).process(pipeline)  # within 1 K
        Measurement(raw=warm_dark(50, 30).raw, exposure=Q_(50, "ms")).process(pipeline)
    with pytest.warns(UserWarning, match="not adjusted for it"):
        warm_dark(50, 36).process(pipeline)


def test_temperature_inputs_are_checked():
    frames = temperature_series()
    with pytest.raises(ValueError, match="every frame a temperature, or none"):
        DarkModel.fit([*frames, Measurement(raw=frames[0].raw, exposure=Q_(5, "ms"))])
    with pytest.raises(TypeError, match="temperature quantities"):
        DarkModel.fit([f.raw for f in frames[:3]], [f.exposure for f in frames[:3]], [20, 21, 22])
    with pytest.raises(ValueError, match="four or more combinations"):
        DarkModel.fit([warm_dark(10, 20), warm_dark(100, 30), warm_dark(400, 40)])


def test_dark_rate_plot():
    ax = DarkModel.fit(temperature_series()).plot()
    assert ax.get_yscale() == "log" and "°C" in ax.get_xlabel()
    with pytest.raises(ValueError, match="one temperature"):
        DarkModel.fit([dark_frame(10), dark_frame(20)], [Q_(10, "ms"), Q_(20, "ms")]).plot()


def test_mask_saturated():
    raw = Image(np.array([[100.0, 4050.0], [4095.0, 10.0]]),  # 98% of 4095 is 4013
                mosaic=Mosaic("RGGB", white_level=4095))
    masked = MaskSaturated().apply(raw, None).values
    assert np.isnan(masked[0, 1]) and np.isnan(masked[1, 0]) and masked[0, 0] == 100
    assert np.isnan(MaskSaturated(50).apply(raw, None).values[0, 0])
    with pytest.raises(ValueError, match="no white level"):
        MaskSaturated().apply(Image(np.ones((2, 2))), None)


def response(stimulus, gain, compression=0.02, full=4000.0):
    linear = gain * np.asarray(stimulus, dtype=float)
    return np.minimum(linear * (1 - compression * (linear / full) ** 2), full)


def test_linearity_finds_gain_deviation_and_saturation():
    t = np.arange(1, 16, dtype=float)
    signal = np.column_stack([response(t, 300), response(t, 400)])
    clipped = signal >= 4000  # as a white level would show
    result = Linearity.fit(Q_(t, "ms"), signal, channels=("R", "G"), clipped=clipped)
    np.testing.assert_allclose(result.gain, [300, 400], rtol=0.01)
    assert result.stimulus_unit == parse("ms")
    np.testing.assert_array_equal(result.saturated, clipped)
    assert result.saturation[1] == pytest.approx(signal[~clipped[:, 1], 1].max())
    expected = signal[~clipped[:, 0], 0] / (result.gain[0] * t[~clipped[:, 0]]) - 1
    assert result.max_deviation[0] == pytest.approx(np.abs(expected).max())
    assert set(result.summary) == {"R", "G"}
    assert result.plot().get_ylabel() == "Deviation from linear (%)"


def test_a_plateau_is_saturation_even_without_a_white_level():
    t = np.arange(1, 11, dtype=float)
    signal = np.minimum(300 * t, 1800.0)  # reaches the plateau exactly at t = 6
    result = Linearity.fit(t, signal)
    assert list(result.saturated[:, 0]) == [False] * 6 + [True] * 4
    assert result.gain[0] == pytest.approx(300)


def test_linearize_undoes_the_compression(tmp_path):
    t = np.arange(1, 13, dtype=float)
    signal = np.column_stack([response(t, g, full=1e5) for g in (300, 400, 350)])
    result = Linearity.fit(t, signal, channels=("R", "G", "B"))
    step = result.step()
    rgb = Image(signal[None, :, :], channels=("R", "G", "B"))
    corrected = step.apply(rgb, None).values[0]
    np.testing.assert_allclose(corrected, t[:, None] * result.gain, rtol=1e-9)
    # on a raw mosaic, each colour's table applies to its own sites
    masks = site_masks("RGGB", (2, 2))
    raw = Image(np.where(masks["R"], signal[5, 0], np.where(masks["G"], signal[5, 1], signal[5, 2])),
                mosaic="RGGB")
    np.testing.assert_allclose(site_means(step.apply(raw, None)), t[5] * result.gain, rtol=1e-9)
    loaded = Calibration.read(Calibration("lin", [step]).write(tmp_path / "lin.gsmath"))
    np.testing.assert_allclose(loaded.steps[0].apply(rgb, None).values, rgb.values * 0 + corrected)


def test_linearity_from_frames_flags_partly_clipped_frames():
    vignetting = np.linspace(0.8, 1.0, 80).reshape(SHAPE)
    frames = []
    for t in range(1, 9):
        values = np.minimum(100 + 400 * t * vignetting, 4095)
        frames.append(Image(values, mosaic=Mosaic("RGGB", white_level=4095)))
    result = Linearity.from_frames(frames, Q_(np.arange(1, 9), "ms"),
                                   dark=Image(np.full(SHAPE, 100.0)))
    # the brightest pixels reach the white level at t = 10.2 ms, so none clip...
    assert not result.saturated.any()
    frames.append(Image(np.minimum(100 + 400 * 10 * vignetting / 0.9, 4095),
                        mosaic=Mosaic("RGGB", white_level=4095)))
    clipped = Linearity.from_frames(frames, Q_(np.arange(1, 10), "ms"),
                                    dark=Image(np.full(SHAPE, 100.0)))
    assert clipped.saturated[-1].all()  # ...but the last frame clips somewhere


def test_build_flat_and_a_mosaic_flat_field_keep_the_colour_balance():
    masks = site_masks("RGGB", (16, 20))
    rows, cols = np.mgrid[0:16, 0:20]
    falloff = 1 - 0.002 * ((rows - 7.5) ** 2 + (cols - 9.5) ** 2)
    colour = np.where(masks["R"], 0.5, np.where(masks["G"], 1.0, 0.8))  # the source's colour
    frames = [Image(100 + 1000 * colour * falloff, value_unit="count",
                    mosaic="RGGB") for _ in range(3)]
    flat = build_flat(frames, dark=Image(np.full((16, 20), 100.0)), sigma=2)
    assert flat.metadata["flat"]["frames"] == 3
    # smoothing stays within each colour's sites: R keeps its own level
    # (away from the border, where smoothing a curved flat biases it)
    inner = masks["R"] & (rows >= 2) & (rows < 14) & (cols >= 2) & (cols < 18)
    np.testing.assert_allclose(flat.values[inner] / (1000 * falloff[inner]), 0.5, rtol=0.02)
    scene = Image(np.where(masks["R"], 300.0, np.where(masks["G"], 200.0, 100.0)) * falloff,
                  mosaic="RGGB")
    corrected = FlatField(flat).apply(scene, None)
    means = site_means(corrected, Rectangle(4, 4, 16, 12))
    np.testing.assert_allclose(means / means[1], [1.5, 1.0, 0.5], rtol=0.01)  # balance kept
