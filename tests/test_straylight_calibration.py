import csv

import numpy as np
import pytest

from gsmath import Measurement, Spectrum, units
from gsmath.scripts import straylight as script
from gsmath.straylight import (
    LineSpreadScan, StrayLightCorrection, build_distribution, characterize_scan,
    evaluate_correction, read_line_scans, stray_light_calibration,
)

X = np.arange(300.0, 901.0, 1.0)


def line_scan(center, *, fwhm=5.0, stray=2e-3, peak=40000.0):
    """A Gaussian line with a broad, weak stray-light pedestal, in counts."""
    sigma = fwhm / 2.3548
    line = np.exp(-0.5 * ((X - center) / sigma) ** 2)
    pedestal = stray * np.exp(-0.5 * ((X - center) / 150.0) ** 2)
    return Spectrum(wavelength=X.copy(), values=peak * (line + pedestal), label=f"{center:g}",
                    wavelength_unit="nm", value_unit="count")


def scans(centers=range(340, 861, 20)):
    return [LineSpreadScan(line_scan(c), float(c), label=f"StrayLight_{c}.csv") for c in centers]


def test_characterize_accepts_a_clean_line_and_rejects_bad_ones():
    good = characterize_scan(scans([500])[0])
    assert good.accepted and good.reason is None
    assert good.center_nm == pytest.approx(500, abs=0.1)
    assert good.fwhm_nm == pytest.approx(5, abs=0.5)

    off = characterize_scan(LineSpreadScan(line_scan(510), 500.0), max_center_error_nm=2.5)
    assert off.status == "rejected" and off.reason.startswith("center error")
    assert off.center_nm == pytest.approx(510, abs=0.1)  # measured even when rejected

    edge = characterize_scan(LineSpreadScan(line_scan(305), 305.0))
    assert edge.status == "rejected" and edge.reason == "insufficient edge margin"


def test_correction_removes_most_of_the_out_of_band_signal():
    result = stray_light_calibration(scans(), min_accepted=10)
    assert len(result.accepted) == len(result.scans) == 27
    assert result.distribution.values.shape == (len(X), len(X))
    assert np.all(result.distribution.values >= 0)
    summary = result.summary
    assert summary["evaluation"]["median_suppression_ratio"] < 0.2
    assert 0.98 < summary["evaluation"]["in_band_ratio_min"] <= summary["evaluation"]["in_band_ratio_max"] < 1.02
    assert summary["edge_extrapolation"]  # centres 340-860 on a 300-900 detector
    assert 1 <= summary["condition_number"] < 2


def test_build_distribution_ignores_rejected_scans_and_needs_two():
    checked = [characterize_scan(s) for s in scans([400, 500, 600])]
    rejected = characterize_scan(LineSpreadScan(line_scan(700), 650.0))
    d = build_distribution(checked + [rejected])
    assert d.values.shape == (len(X), len(X))
    with pytest.raises(ValueError, match="two accepted"):
        build_distribution(checked[:1] + [rejected])
    with pytest.raises(ValueError, match="4 scans accepted|1 of"):
        stray_light_calibration(scans([500]) + [LineSpreadScan(line_scan(700), 650.0)], min_accepted=3)


def test_identity_correction_changes_nothing():
    checked = [characterize_scan(s) for s in scans([400, 600])]
    evaluation = evaluate_correction(StrayLightCorrection(X.copy(), np.eye(len(X))), checked)
    for row in evaluation.rows():
        assert row["suppression_ratio"] == pytest.approx(1)
        assert row["in_band_ratio"] == pytest.approx(1)
        assert row["negative_area"] == 0
    assert "Set nm" in evaluation.table()


def test_result_writes_matrices_reports_and_plots(tmp_path):
    result = stray_light_calibration(scans(range(360, 841, 40)))
    matrix = result.write(tmp_path / "out")
    assert matrix == tmp_path / "out" / "matrices" / "C.gsmath"
    restored = StrayLightCorrection.read(matrix)
    np.testing.assert_array_equal(restored.values, result.correction.values)
    legacy = StrayLightCorrection.read(tmp_path / "out" / "matrices" / "C.bin", X)
    np.testing.assert_allclose(legacy.values, result.correction.values, rtol=1e-6, atol=1e-7)
    for name in ("scans.csv", "evaluation.csv", "summary.json", "plots/fwhm.png",
                 "plots/distribution.png", "plots/correction.png"):
        assert (tmp_path / "out" / name).is_file(), name


def test_script_reads_a_station_manifest_of_native_measurements(tmp_path):
    """The station's run: native measurements (raw + dark) listed in manifest.csv."""
    run = tmp_path / "scans"
    run.mkdir()
    rows = []
    for center in range(360, 841, 40):
        signal = line_scan(center)
        dark = signal.copy(values=np.full(len(X), 200.0), label="dark")
        raw = signal.copy(values=signal.values + 200.0)
        name = f"line_{center:04d}nm.gsmath"
        Measurement(raw=raw, dark=dark, exposure=units.Q_(100, "ms")).write(run / name)
        rows.append((name, center))
    with (run / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("file", "set_wavelength_nm"))
        writer.writerows(rows)

    read = read_line_scans(run)
    assert [s.set_wavelength_nm for s in read] == [r[1] for r in rows]
    np.testing.assert_allclose(read[0].spectrum.values, line_scan(360).values)

    assert script.main(["calibrate", str(run), "--output", str(tmp_path / "out"), "--plot-every", "0"]) == 0
    assert script.main(["evaluate", str(run), "--matrix", str(tmp_path / "out" / "matrices" / "C.gsmath"),
                        "--output", str(tmp_path / "eval"), "--plot-every", "0"]) == 0
    assert (tmp_path / "eval" / "evaluation.csv").is_file()


def test_script_reads_named_two_column_csvs(tmp_path):
    for center in (400, 500, 600):
        spectrum = line_scan(center)
        np.savetxt(tmp_path / f"StrayLight_{center}.0.csv",
                   np.column_stack((spectrum.wavelength, spectrum.values)), delimiter=",",
                   header="wavelength_nm,counts", comments="")
    read = read_line_scans(tmp_path)
    assert [s.set_wavelength_nm for s in read] == [400, 500, 600]
    assert script.main(["calibrate", str(tmp_path / "missing")]) == 2
