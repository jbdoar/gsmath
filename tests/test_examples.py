"""Smoke tests for examples that compose several public gsmath APIs."""

import importlib.util
from pathlib import Path

import numpy as np

from gsmath import Spectrum


def test_compare_spectral_collections_manifest(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples/compare_spectral_collections.py"
    spec = importlib.util.spec_from_file_location("compare_spectral_collections", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    wavelength = np.arange(360., 831.)
    for name, center in (("reference", 550.), ("measured", 551.)):
        values = np.exp(-0.5 * ((wavelength - center) / 30.) ** 2)
        Spectrum(wavelength, values, value_unit="uW / cm^2 / sr / nm").write(
            tmp_path / f"{name}.gsmath"
        )
    manifest = tmp_path / "pairs.csv"
    manifest.write_text("label,left,right\ngreen,reference.gsmath,measured.gsmath\n",
                        encoding="utf-8")
    report = tmp_path / "reports" / "comparison.csv"
    example.main([str(manifest), "--format", "gsmath", "--output", str(report),
                  "--no-plots"])
    assert report.is_file()
    assert "green" in report.read_text(encoding="utf-8")


def test_camera_colorimeter_end_to_end():
    """A simulated Bayer camera calibrated as a colorimeter measures a
    non-uniform display to within its noise."""
    path = Path(__file__).resolve().parents[1] / "examples/camera_colorimeter.py"
    spec = importlib.util.spec_from_file_location("camera_colorimeter", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    result = example.run(shape=(64, 96), frames=4)
    measured = result["measured"]
    assert measured["dark"].hot_pixels().array.sum() == 12
    assert abs(measured["dark"].read_noise - 3.0) < 0.3
    assert abs(measured["dark"].doubling - 6.5) < 0.3
    assert abs(result["dark_left"]["temperature model"]) < 0.5
    assert result["dark_left"]["fitted at 24 degC"] > 20
    assert (measured["linearity"].max_deviation < 0.02).all()
    assert abs(result["Y_error_mean"]) < 0.005
    assert result["delta_uv_mean"] < 0.003
    sampled, true = result["uniformity"], result["true_uniformity"]
    # At this small size the measured ratio runs about 0.006 low for every
    # seed (0.002-0.010): the smoothed flat's bias near the edges, where
    # the outer spots sit. The full-size example agrees to about 0.004.
    assert abs(sampled.luminance["min_max_ratio"] - true.luminance["min_max_ratio"]) < 0.012
    assert abs(sampled.color["max_delta_uv_prime_from_centre"]
               - true.color["max_delta_uv_prime_from_centre"]) < 0.001
