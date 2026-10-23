"""Native .gsmath files keep reading, and keep being written the same way.

The fixtures in tests/data/native were written before the codec-registry
refactoring (see native_fixtures.py). Reading each must give the right kind
of object; writing that object again must reproduce the file: byte for byte
for JSON records, and the same record and arrays for containers (whose zip
timestamps differ).
"""

import json
import zipfile

import numpy as np
import pytest

import gsmath
from gsmath import Calibration, Image, Measurement, Spectrum
from gsmath.straylight import StrayLightCorrection, StrayLightDistribution

from native_fixtures import DIRECTORY

EXPECTED = {
    "spectrum.gsmath": Spectrum,
    "measurement_spectrum.gsmath": Measurement,
    "calibration_spectral.gsmath": Calibration,
    "straylight_distribution.gsmath": StrayLightDistribution,
    "straylight_correction.gsmath": StrayLightCorrection,
    "image.gsmath": Image,
    "measurement_image.gsmath": Measurement,
    "calibration_image.gsmath": Calibration,
}


def contents(path):
    """A container's record text and arrays, or a JSON file's bytes."""
    if not zipfile.is_zipfile(path):
        return path.read_bytes()
    with zipfile.ZipFile(path) as container:
        record = container.read("record.json")
        arrays = {}
        for name in sorted(container.namelist()):
            if name.endswith(".npy"):
                with container.open(name) as member:
                    arrays[name] = np.lib.format.read_array(member, allow_pickle=False)
    return record, arrays


@pytest.mark.parametrize("name", EXPECTED)
def test_fixture_reads_and_writes_identically(name, tmp_path):
    fixture = DIRECTORY / name
    obj = gsmath.read(fixture)
    assert isinstance(obj, EXPECTED[name])
    written = obj.write(tmp_path / name) or tmp_path / name
    before, after = contents(fixture), contents(written)
    if isinstance(before, bytes):
        assert after == before
    else:
        assert after[0] == before[0]
        assert list(after[1]) == list(before[1])
        for key in before[1]:
            np.testing.assert_array_equal(after[1][key], before[1][key])
            assert after[1][key].dtype == before[1][key].dtype


def test_fixtures_are_what_the_records_say():
    record = json.loads((DIRECTORY / "calibration_spectral.gsmath").read_text())
    kinds = [argument["kind"] for step in record["calibration"]["steps"]
             for argument in step["arguments"].values()]
    assert set(kinds) == {"stray_light", "wavelength_calibration", "spectrum", "quantity",
                          "number", "array"}


def test_a_registered_codec_makes_a_new_value_savable(tmp_path):
    from dataclasses import dataclass

    from gsmath.calibration import Step
    from gsmath.io import native

    @dataclass(frozen=True)
    class Gain:
        value: float

    class ApplyGain(Step):
        def __init__(self, gain, *, optional=False):
            super().__init__(optional=optional)
            self.gain = gain

        def apply(self, data, measurement):
            return data * self.gain.value

        def parameters(self):
            return {"gain": self.gain.value}

        def arguments(self):
            return {"gain": self.gain}

    calibration = Calibration("gain", [ApplyGain(Gain(2.5))])
    with pytest.raises(TypeError, match="can't save a value of type Gain"):
        calibration.write(tmp_path / "before.gsmath")
    codec = native.Codec("gain", lambda v: isinstance(v, Gain),
                         lambda v: {"value": v.value}, lambda p: Gain(p["value"]))
    native.CODECS.insert(0, codec)
    try:
        loaded = Calibration.read(calibration.write(tmp_path / "gain.gsmath"))
        assert loaded.steps[0].gain == Gain(2.5)
        assert loaded.fingerprint == calibration.fingerprint
    finally:
        native.CODECS.remove(codec)


def test_records_and_formats_are_looked_up_in_one_place(tmp_path):
    from gsmath.io import formats, native

    assert native.record_for(Spectrum([1, 2], [3, 4])).format == "gsmath.spectrum"
    with pytest.raises(TypeError, match="no native record stores"):
        native.record_for(object())
    assert formats.resolve("a.TIF").name == "tiff"
    assert formats.resolve("x", "jpg", kind="image").name == "jpeg"
    with pytest.raises(ValueError, match="Unknown spectrum format 'png'"):
        formats.resolve("x", "png", kind="spectrum")
    with pytest.raises(ValueError, match="Cannot infer image format from suffix '.cal'"):
        formats.resolve("x.cal", kind="image")
    path = Spectrum([1, 2], [3, 4]).write(tmp_path / "s.gsmath")
    with pytest.raises(ValueError, match="Not a 'gsmath.image' file"):
        native.read_record(path, record="gsmath.image")
    assert native._value_payload is native.encode_value  # older name kept for linefit


def test_raw_frames_saved_with_the_mosaic_in_metadata_still_read():
    measurement = gsmath.read(DIRECTORY / "legacy_mosaic_in_metadata.gsmath")
    assert measurement.raw.mosaic.pattern == "RGGB"
    assert measurement.raw.mosaic.black_level == (64.0,) * 4
    assert measurement.raw.mosaic.white_level == 4095.0
    assert "mosaic" not in measurement.raw.metadata
    assert measurement.dark.mosaic == measurement.raw.mosaic


def test_large_arrays_go_in_a_container_small_ones_stay_json(tmp_path):
    import zipfile

    import numpy as np

    from gsmath import Calibration, Spectrum, StrayLight, read
    from gsmath.io.native import LARGE_ARRAY_BYTES
    from gsmath.straylight import StrayLightCorrection

    wl = np.arange(400.0, 912.0)  # 512 x 512 float64: 2 MiB
    big = StrayLightCorrection(wl, np.eye(wl.size) * 0.5)
    assert big.values.nbytes > LARGE_ARRAY_BYTES
    path = big.write(tmp_path / "big.gsmath")
    assert zipfile.is_zipfile(path)
    np.testing.assert_array_equal(read(path).values, big.values)
    calibration = Calibration("with a big matrix", [StrayLight(big)])
    path = calibration.write(tmp_path / "cal.gsmath")
    assert zipfile.is_zipfile(path) and read(path).fingerprint == calibration.fingerprint
    small = Spectrum(wl, np.ones(wl.size)).write(tmp_path / "small.gsmath")
    assert not zipfile.is_zipfile(small) and small.read_text(encoding="utf-8").startswith("{")
