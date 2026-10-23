"""Calibrate a (simulated) Bayer camera as an imaging colorimeter, end to end.

A simulated 12-bit RGGB camera -- uncooled, so its bias and dark current
(with hot pixels) drift with sensor temperature; read and shot noise,
vignetting, slight nonlinearity, clipping -- goes through the whole procedure:

1. dark frames at several exposures, as the sensor warms up
                                         -> DarkModel, temperature-dependent
2. frames of a uniform source            -> build_flat
3. an exposure series up to clipping     -> Linearity (and its correction)
4. a monochromator scan, through the same dark and flat steps
                                         -> SpectralSensitivity
5. the display's primaries and white     -> fit_color_matrix
6. a frame of a non-uniform display      -> calibrated X, Y, Z, compared
   with the truth, and its uniformity.

It also compares the dark of a long exposure on a warm sensor with what a
dark model fitted at one temperature would subtract.

Run::

    python examples/camera_colorimeter.py
"""

from __future__ import annotations

import numpy as np

from gsmath import (
    Calibration, DarkModel, Demosaic, FlatField, Image, Linearity, MaskSaturated, Measurement,
    PerExposure, SpectralSensitivity, Spectrum, build_flat, fit_color_matrix, grid, site_means,
)
from gsmath import xyz_to_uv_prime
from gsmath.mosaic import Mosaic, site_masks
from gsmath.units import Q_

WL = np.arange(380.0, 781.0, 1.0)
RADIANCE = "W/sr/m^2/nm"


def gauss(mu, fwhm):
    return np.exp(-4 * np.log(2) * ((WL - mu) / fwhm) ** 2)


class SimulatedCamera:
    """A 12-bit RGGB camera; signals in counts (one electron per count)."""

    white_level = 4095.0

    def __init__(self, shape=(96, 128), seed=0):
        self.shape = shape
        self.rng = np.random.default_rng(seed)
        rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
        r2 = ((rows - (shape[0] - 1) / 2) ** 2 + (cols - (shape[1] - 1) / 2) ** 2) \
            / ((shape[0] / 2) ** 2 + (shape[1] / 2) ** 2)
        self.vignetting = (1 - 0.25 * r2) ** 2
        self.temperature = 25.0                                # sensor, degC
        self.bias = 100 + self.rng.normal(0, 2, shape)        # fixed pattern, at 25 degC
        self.bias_slope = 0.5 + self.rng.normal(0, 0.05, shape)  # counts/K
        self.dark_rate = np.full(shape, 20.0)                  # counts/s at 25 degC,
        self.doubling = 6.5                                    # doubling every 6.5 K
        hot = self.rng.choice(self.dark_rate.size, 12, replace=False)
        self.dark_rate.flat[hot] = 800.0
        self.read_noise = 3.0
        # Sensitivity, counts per ms per (W/sr/m^2) -- not a mix of the CMFs.
        self.sensitivity = SpectralSensitivity(
            WL, np.column_stack([gauss(600, 70) + 0.05 * gauss(440, 40), gauss(530, 95),
                                 gauss(462, 75)]) * 2.2e3,
            ("R", "G", "B"), "count/ms / (W/sr/m^2)")
        self.masks = site_masks("RGGB", shape)

    def capture(self, scene, exposure_ms):
        """A raw frame. ``scene`` is [(weight map or 1, Spectrum)]: radiance at
        each pixel is the weighted sum of the spectra."""
        signal = np.zeros(self.shape)
        for weight, spectrum in scene:
            rgb = self.sensitivity.response(spectrum).to("count/ms").magnitude
            per_site = sum(np.where(self.masks[c], rgb[k], 0.0) for k, c in enumerate("RGB"))
            signal += np.broadcast_to(weight, self.shape) * per_site
        electrons = signal * self.vignetting * exposure_ms
        electrons = electrons + self.rng.normal(0, np.sqrt(np.maximum(electrons, 0)))
        electrons *= 1 - 0.02 * (electrons / self.white_level) ** 2   # slight compression
        warmer = self.temperature - 25
        dark = self.bias + self.bias_slope * warmer \
            + self.dark_rate * 2 ** (warmer / self.doubling) * exposure_ms / 1e3
        counts = electrons + dark + self.rng.normal(0, self.read_noise, self.shape)
        counts = np.clip(np.round(counts), 0, self.white_level)
        return Image(counts.astype(np.uint16), label="raw", value_unit="count",
                     mosaic=Mosaic("RGGB", white_level=self.white_level))

    def measure(self, scene, exposure_ms, frames=1):
        """A Measurement of the mean of ``frames`` raw frames, with the
        sensor temperature as read out (to 0.1 degC)."""
        captured = [self.capture(scene, exposure_ms) for _ in range(frames)]
        mean = np.mean([frame.values for frame in captured], axis=0)
        return Measurement(raw=captured[0].copy(values=mean), exposure=Q_(exposure_ms, "ms"),
                           temperature=Q_(round(self.temperature, 1), "degC"))


def display_primaries(scale=1.0):
    """A display's red, green and blue spectra (radiance per unit drive)."""
    return [Spectrum(WL, scale * g * 1e-3, value_unit=RADIANCE)
            for g in (gauss(630, 30), gauss(532, 35), gauss(456, 22))]


def calibrate(camera, *, frames=8):
    """Steps 1-5; returns the colorimeter Calibration and what was measured."""
    # 1. Dark frames at four exposures, three of each, at each of four sensor
    #    temperatures as the camera warms up (each Measurement records its own).
    darks = {}
    for celsius in (24, 30, 36, 42):
        camera.temperature = celsius
        darks[celsius] = [camera.measure([], t) for t in (5, 20, 80, 320) for _ in range(3)]
    dark = DarkModel.fit([m for runs in darks.values() for m in runs])
    one_temperature = DarkModel.fit(darks[24])  # for comparison: as if never warmed
    camera.temperature = 33.0

    # 2. A uniform broadband source (an integrating sphere) for the flat.
    sphere = [(1.0, Spectrum(WL, 2e-3 * (1 + 0.3 * (WL - 380) / 400), value_unit=RADIANCE))]
    flat = build_flat([camera.measure(sphere, 4) for _ in range(frames)], dark=dark, sigma=4)

    # 3. Linearity: the sphere at rising exposures, up to clipping.
    exposures = Q_(np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]), "ms")
    series = [camera.measure(sphere, t, frames=2).raw for t in exposures.magnitude]
    linearity = Linearity.from_frames(series, exposures, dark=dark,
                                      temperatures=Q_(camera.temperature, "degC"))

    # Frames are cleaned the same way from here on -- the monochromator scan
    # too, so the matrix is fitted on the signal it will be applied to.
    front = [MaskSaturated(), dark.step(), linearity.step(), FlatField(flat)]

    # 4. Monochromator scan: 5 nm-wide bands, their radiance known.
    centres = np.arange(390.0, 771.0, 5.0)
    signals, band_radiance = [], []
    for centre in centres:
        band = Spectrum(WL, 5e-3 * np.exp(-4 * np.log(2) * ((WL - centre) / 5) ** 2),
                        value_unit=RADIANCE)
        measurement = camera.measure([(1.0, band)], 30, frames=2)
        cleaned = Calibration("front", front).process(measurement)
        signals.append(site_means(cleaned) / 30)                # counts/ms
        band_radiance.append(np.trapezoid(band.values, WL))     # W/sr/m^2
    sensitivity = SpectralSensitivity.from_monochromator(
        centres, np.array(signals), np.array(band_radiance),
        signal_unit="count/ms", reference_unit="W/sr/m^2")

    # 5. The display's primaries and white, as a spectroradiometer gives them.
    primaries = display_primaries(scale=1.2)
    training = primaries + [Spectrum(WL, sum(p.values for p in primaries), value_unit=RADIANCE)]
    fit = fit_color_matrix(sensitivity.responses(training),
                           [s.color().XYZ for s in training], preserve=3,
                           unit="cd/m^2 / (count/ms)")
    calibration = Calibration("colorimeter", [*front, Demosaic(), PerExposure(), fit.step()])
    return calibration, {"dark": dark, "one_temperature_dark": one_temperature,
                         "flat": flat, "linearity": linearity,
                         "sensitivity": sensitivity, "fit": fit}


def display_scene(camera, primaries):
    """A non-uniform display: each primary falls off differently, so both
    luminance and colour vary. Returns the scene and the true X, Y, Z."""
    rows, cols = np.mgrid[0:camera.shape[0], 0:camera.shape[1]].astype(float)
    r2 = ((rows / (camera.shape[0] - 1) - 0.5) ** 2 + (cols / (camera.shape[1] - 1) - 0.5) ** 2)
    weights = [1 - 0.3 * r2, 1 - 0.5 * r2, 1 - 0.8 * r2]
    scene = list(zip(weights, primaries))
    XYZ = sum(w[..., None] * p.color().XYZ for w, p in scene)
    return scene, XYZ


def run(shape=(96, 128), frames=8, seed=0):
    camera = SimulatedCamera(shape, seed)
    calibration, measured = calibrate(camera, frames=frames)
    primaries = display_primaries(scale=1.2)
    scene, truth = display_scene(camera, primaries)
    camera.temperature = 38.0  # warmer than during the calibration
    xyz = camera.measure(scene, 20, frames=frames).process(calibration)
    inner = (slice(2, -2), slice(2, -2))  # demosaic edges
    Y_error = xyz.values[..., 1][inner] / truth[..., 1][inner] - 1
    duv = np.hypot(*np.moveaxis(xyz_to_uv_prime(xyz.values[inner])
                                - xyz_to_uv_prime(truth[inner]), -1, 0))
    measured_uniformity = xyz.uniformity(points=grid(3, 3, inset=0.1), spot=0.1)
    true_uniformity = Image(truth, channels=("X", "Y", "Z")).uniformity(
        points=grid(3, 3, inset=0.1), spot=0.1)
    # A long dark exposure on a warm sensor: what's left after subtracting
    # each model's dark (the one-temperature model can't follow the drift).
    camera.temperature = 40.0
    long_dark = camera.measure([], 250, frames=frames)
    left = {name: float(np.median(long_dark.raw.values
                                  - model.dark(long_dark.exposure, long_dark.temperature).values))
            for name, model in (("temperature model", measured["dark"]),
                                ("fitted at 24 degC", measured["one_temperature_dark"]))}
    return {
        "measured": measured, "xyz": xyz, "truth": truth, "dark_left": left,
        "Y_error_mean": float(np.nanmean(Y_error)), "Y_error_rms": float(np.sqrt(np.nanmean(Y_error ** 2))),
        "delta_uv_mean": float(np.nanmean(duv)), "delta_uv_p99": float(np.nanpercentile(duv, 99)),
        "uniformity": measured_uniformity, "true_uniformity": true_uniformity,
    }


def main():
    result = run()
    measured = result["measured"]
    print("Dark model:", {k: v for k, v in measured["dark"].summary.items() if k != "exposures"})
    print("Dark left at 40 degC, 250 ms (median counts):",
          {name: round(v, 2) for name, v in result["dark_left"].items()})
    print("Linearity:", measured["linearity"].summary)
    print("Matrix fit (training):", measured["fit"].summary["training"]["delta_uv_prime"])
    print(f"Per-pixel luminance error: mean {result['Y_error_mean'] * 100:+.2f}%, "
          f"rms {result['Y_error_rms'] * 100:.2f}%")
    print(f"Per-pixel du'v': mean {result['delta_uv_mean']:.4f}, "
          f"99th percentile {result['delta_uv_p99']:.4f}")
    for name in ("min_max_ratio", "max_deviation_from_centre"):
        print(f"Uniformity {name}: measured {result['uniformity'].luminance[name]:.4f}, "
              f"true {result['true_uniformity'].luminance[name]:.4f}")
    print(f"Uniformity max du'v' from centre: measured "
          f"{result['uniformity'].color['max_delta_uv_prime_from_centre']:.4f}, true "
          f"{result['true_uniformity'].color['max_delta_uv_prime_from_centre']:.4f}")


if __name__ == "__main__":
    main()
