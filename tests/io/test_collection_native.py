"""Collections as single native files: gsmath.collection records."""

import json
import zipfile

import numpy as np
import pytest

import gsmath
from gsmath import Collection, ExposureSeries, Image, Measurement, Spectrum, compare
from gsmath.profile import SPECTRUM_PROFILE
from gsmath.units import Q_


def led(label, peak=500.0, scale=1.0):
    x = np.linspace(380.0, 780.0, 81)
    return Spectrum(x, scale * np.exp(-0.5 * ((x - peak) / 10.0) ** 2), label=label,
                    value_unit="uW/cm^2/sr/nm", metadata={"serial": label})


def test_a_collection_of_spectra_is_one_readable_json_file(tmp_path):
    leds = Collection([led("450nm", 450), led("520nm", 520), led("620nm", 620)],
                      label="SN-2066", metadata={"operator": "lab", "nested": {"a": [1, 2]}})
    path = leds.write(tmp_path / "leds.gsmath")
    assert not zipfile.is_zipfile(path)
    record = json.loads(path.read_text())
    assert (record["format"], record["version"]) == ("gsmath.collection", 1)
    assert [item["format"] for item in record["collection"]["items"]] == ["gsmath.spectrum"] * 3

    loaded = gsmath.read(path)
    assert isinstance(loaded, Collection)
    assert (loaded.label, loaded.metadata) == ("SN-2066", leds.metadata)
    assert loaded.labels == ("450nm", "520nm", "620nm")
    for before, after in zip(leds, loaded):
        np.testing.assert_array_equal(after.values, before.values)
        np.testing.assert_array_equal(after.wavelength, before.wavelength)
        assert after.value_unit == before.value_unit
        assert after.metadata == before.metadata
    assert Collection.read(path).labels == loaded.labels


def test_mixed_records_and_images_go_in_a_container(tmp_path):
    raw = Spectrum([400.0, 500.0], [900.0, 950.0], label="raw", value_unit="count")
    dark = Spectrum([400.0, 500.0], [100.0, 100.0], label="dark", value_unit="count")
    frame = Image(np.arange(12, dtype=np.uint16).reshape(3, 4), label="frame", value_unit="count")
    series = ExposureSeries([Measurement(raw, dark, exposure=Q_(t, "ms")) for t in (10, 20)],
                            saturation=None, label="hdr")
    mixed = Collection([led("lamp"), Measurement(raw, dark, exposure=Q_(5, "ms"), label="scan"),
                        series, frame, Collection([led("inner")], label="nested")], label="mixed")
    path = mixed.write(tmp_path / "mixed.gsmath")
    assert zipfile.is_zipfile(path)
    loaded = Collection.read(path)
    assert loaded.labels == ("lamp", "scan", "hdr", "frame", "nested")
    assert isinstance(loaded["scan"], Measurement)
    np.testing.assert_array_equal(loaded["scan"].dark.values, dark.values)
    assert isinstance(loaded["hdr"], ExposureSeries)
    assert loaded["hdr"].exposures == [Q_(10, "ms"), Q_(20, "ms")]
    np.testing.assert_array_equal(loaded["frame"].values, frame.values)
    assert loaded["frame"].values.dtype == np.uint16
    assert loaded["nested"].labels == ("inner",)


def test_compare_of_two_collection_files_matches_comparing_the_collections(tmp_path):
    measured = Collection([led("450nm", 451), led("520nm", 520, 1.2)])
    reference = Collection([led("520nm", 520), led("450nm", 450)])
    a = measured.write(tmp_path / "a.gsmath")
    b = reference.write(tmp_path / "b.gsmath")
    from_files = compare(gsmath.read(a), gsmath.read(b), profile=SPECTRUM_PROFILE)
    direct = compare(measured, reference, profile=SPECTRUM_PROFILE)
    for left, right in zip(from_files, direct):
        assert left.label == right.label
        for path, record in left.records.items():
            assert record.difference == pytest.approx(right.records[path].difference)


def test_what_a_collection_file_cannot_hold_is_refused(tmp_path):
    class Note:
        label = "note"

    with pytest.raises(TypeError, match="no native record stores a Note"):
        Collection([Note()]).write(tmp_path / "notes.gsmath")

    path = Collection([led("a"), led("b")]).write(tmp_path / "two.gsmath")
    record = json.loads(path.read_text())
    record["collection"]["items"][1]["spectrum"]["label"] = "a"
    duplicated = tmp_path / "duplicated.gsmath"
    duplicated.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="labels must be unique"):
        gsmath.read(duplicated)

    record = json.loads(path.read_text())
    record["collection"]["items"][0]["format"] = "gsmath.nothing"
    unknown = tmp_path / "unknown.gsmath"
    unknown.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unknown gsmath record 'gsmath.nothing'"):
        gsmath.read(unknown)

    record = json.loads(path.read_text())
    record["collection"]["items"][0]["version"] = 99
    newer = tmp_path / "newer.gsmath"
    newer.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="item 0: unsupported spectrum version 99"):
        gsmath.read(newer)


def test_an_empty_collection_round_trips(tmp_path):
    empty = Collection(label="nothing yet")
    loaded = Collection.read(empty.write(tmp_path / "empty.gsmath"))
    assert (len(loaded), loaded.label) == (0, "nothing yet")
