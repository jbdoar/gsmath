import json

import numpy as np
import pytest

from gsmath import Spectrum


@pytest.fixture
def spectrum():
    return Spectrum(
        wavelength=np.array([400, 500, 600], dtype=np.int32),
        values=np.array([1.25, np.nan, np.inf], dtype=np.float64),
        label="Measured spectrum",
        wavelength_unit="nm",
        value_unit="uW / cm^2 / sr / nm",
        metadata={"instrument": "example", "averages": np.int64(3)},
    )


def test_native_round_trip_preserves_spectrum(spectrum, tmp_path):
    path = tmp_path / "nested" / "measurement.gsmath"

    result_path = spectrum.write(path)
    result = Spectrum.read(path)

    assert result_path == path
    np.testing.assert_array_equal(result.wavelength, spectrum.wavelength)
    np.testing.assert_array_equal(result.values, spectrum.values)
    assert result.wavelength.dtype == spectrum.wavelength.dtype
    assert result.values.dtype == spectrum.values.dtype
    assert result.label == spectrum.label
    assert result.wavelength_unit == spectrum.wavelength_unit
    assert result.value_unit == spectrum.value_unit
    assert result.metadata == {"instrument": "example", "averages": 3}


def test_native_format_can_override_extension(spectrum, tmp_path):
    path = tmp_path / "measurement.data"
    spectrum.write(path, format="native")
    result = Spectrum.read(path, format="gsmath")
    assert result.label == spectrum.label


def test_csv_round_trip(spectrum, tmp_path):
    path = tmp_path / "measurement.csv"
    spectrum.write(path)

    result = Spectrum.read(
        path,
        wavelength_unit=spectrum.wavelength_unit,
        value_unit=spectrum.value_unit,
    )

    np.testing.assert_allclose(result.wavelength, spectrum.wavelength)
    np.testing.assert_allclose(result.values, spectrum.values)
    assert result.label == "measurement"


def test_csv_optional_header(spectrum, tmp_path):
    path = tmp_path / "measurement.csv"
    spectrum.write(path, header=True)
    result = Spectrum.read(path, skip_header=1)
    np.testing.assert_allclose(result.values, spectrum.values)


def test_unknown_or_ambiguous_format_is_rejected(spectrum, tmp_path):
    path = tmp_path / "measurement.dat"
    with pytest.raises(ValueError, match="Cannot infer"):
        spectrum.write(path)
    with pytest.raises(ValueError, match="Unknown spectrum format"):
        spectrum.write(path, format="vendor-x")


def test_lighttouch_calibration_round_trip(tmp_path):
    spectrum = Spectrum(
        [400.0, 500.0, 600.0], [1.25, 2.5, 5.0],
        label="Amplitude response",
        value_unit="uW / cm^2 / sr / nm / count",
    )
    path = spectrum.write(tmp_path / "amplitude.cal", integration_time_ms=125.5)

    restored = Spectrum.read(path)

    assert path.read_text(encoding="utf-8").splitlines()[:8] == [
        "2.0 [Gamma Scientific Spectral File]",
        "LightTouch Amplitude Cal (1500 nm,0 ms,0 ms)",
        "Radiance (µW/cm²/sr)  ",
        "1",
        "125.5 [Integration Time ms]",
        "400.00000000",
        "3",
        "600.00000000",
    ]
    np.testing.assert_allclose(restored.wavelength, spectrum.wavelength, atol=5e-7)
    np.testing.assert_allclose(restored.values, spectrum.values, rtol=1e-8)


def test_lighttouch_spectral_file_saved_from_a_spreadsheet(tmp_path):
    # Header lines end with a comma, the header block is padded with empty
    # fields, and the text is cp1252.
    from gsmath.amplitude import lighttouch_unit

    lines = ["2.0 [Gamma Scientific Spectral File],", "Spectral Scan,", "Radiance (µW/cm²/sr)  ,",
             "3,", ",", ",", ",", "300.0,", "3,", "310.0,",
             "300.0,1.5e-02", "305.0,2.5e-02", "310.0,3.5e-02"]
    path = tmp_path / "standard.csv"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("cp1252"))

    spectrum = Spectrum.from_lighttouch(path)

    np.testing.assert_allclose(spectrum.wavelength, [300.0, 305.0, 310.0])
    np.testing.assert_allclose(spectrum.values, [1.5e-2, 2.5e-2, 3.5e-2])
    assert spectrum.metadata["header"][2] == "Radiance (µW/cm²/sr)"
    assert lighttouch_unit(spectrum) is not None


def test_native_rejects_unknown_version(spectrum, tmp_path):
    path = tmp_path / "measurement.gsmath"
    spectrum.write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported native spectrum version"):
        Spectrum.read(path)


def test_native_rejects_non_json_metadata(tmp_path):
    spectrum = Spectrum([400], [1], metadata={"object": object()})
    with pytest.raises(TypeError, match="metadata must be JSON-compatible"):
        spectrum.write(tmp_path / "measurement.gsmath")
