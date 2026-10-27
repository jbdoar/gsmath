import numpy as np
import pytest

from gsmath import (Collection, EmissionLine, LineSpreadFunction, Spectrum, evaluate_wavelength,
                    fit_wavelength, lamp_lines, line_center)

N = 1044
TRUTH = np.array([-4.0e-9, 2.45e-5, 0.883, 204.9])  # descending, nm per pixel index
SHAPE = LineSpreadFunction((1.5, 1.3, 2.0), asymmetry=0.3, slit_width_px=0.5)


def truth_pixel(wavelength):
    axis = np.polyval(TRUTH, np.arange(N))
    return np.interp(wavelength, axis, np.arange(N))


def lamp_scan(*, shape=SHAPE, noise=2.0, seed=4):
    """A HgAr scan whose line means sit exactly on the TRUTH mapping."""
    rng = np.random.default_rng(seed)
    lines = [line.wavelength_nm for line in lamp_lines()]
    centers = truth_pixel(lines)
    inside = (centers > 5) & (centers < N - 6)
    areas = rng.uniform(2e3, 2e4, inside.sum())
    x = np.arange(N)
    y = shape.profiles(x, centers[inside], N) @ areas + 50 + 0.02 * x
    return y + rng.normal(0, noise, N)


def anchored_error(evaluation):
    """Largest |fitted - true| wavelength between the outermost measured lines."""
    pixels = [line.pixel for line in evaluation.measured]
    x = np.linspace(min(pixels), max(pixels), 500)
    return np.max(abs(evaluation.calibration.wavelength(x) - np.polyval(TRUTH, x)))


@pytest.fixture(scope="module")
def scan():
    return lamp_scan()


@pytest.fixture(scope="module")
def fitted(scan):
    return fit_wavelength(scan, direction="increasing")


def test_bundled_lamp_lines_are_labelled_and_sorted():
    lines = lamp_lines("hgar")
    assert isinstance(lines, Collection)
    wavelengths = [line.wavelength_nm for line in lines]
    assert wavelengths == sorted(wavelengths)
    assert lines["Hg I 546.074"].wavelength_nm == 546.074
    assert lines["Ar I 1046.7177"].species == "Ar I"


def test_line_list_csv_with_comments_and_unused_lines(tmp_path):
    path = tmp_path / "lamp.csv"
    path.write_text("# a comment\nwavelength_nm,species,use\n546.074,Hg I,yes\n365.4836,Hg I,no\n")
    lines = lamp_lines(path)
    assert [line.label for line in lines] == ["Hg I 365.4836", "Hg I 546.074"]
    assert [line.use for line in lines] == [False, True]
    path.write_text("nm\n546.074\n")
    with pytest.raises(ValueError, match="wavelength_nm"):
        lamp_lines(path)


def test_fit_recovers_the_mean_centred_mapping(fitted):
    assert fitted.passed
    assert len(fitted.measured) >= 15
    assert anchored_error(fitted) < 0.05
    assert fitted.summary["max_abs_holdout_nm"] < 0.1


def test_close_pairs_are_reported_as_blends_not_judged(fitted):
    status = {line.label: line.status for line in fitted.lines}
    for label in ("Hg I 576.96", "Hg I 579.066", "Ar I 772.376", "Ar I 772.421",
                  "Ar I 1046.7177", "Ar I 1047.0054"):
        assert status[label] == "blend"
    judged = set(fitted.comparison.labels)
    assert not judged & {"Hg I 576.96", "Ar I 772.376"}


def test_peak_maxima_are_offset_from_line_means(scan, fitted):
    # The asymmetric line's maximum is not its mean: a calibration through the
    # maxima is offset from the truth by more than one through fitted means.
    peak = fit_wavelength(scan, method="peak", direction="increasing")
    assert anchored_error(peak) > 2 * anchored_error(fitted)


def test_evaluation_is_a_profile_comparison_per_line(fitted):
    measured = {line.label for line in fitted.measured}
    assert set(fitted.comparison.labels) == measured
    record = fitted.comparison["Hg I 546.074"].records["wavelength_nm"]
    assert record.field.tolerance == 0.5
    assert record.difference == pytest.approx(fitted.lines[3].error_nm)


def test_evaluate_passes_true_and_fails_shifted_calibrations(scan):
    good = evaluate_wavelength(scan, TRUTH)
    assert good.passed and good.summary["max_abs_error_nm"] < 0.1
    shifted = evaluate_wavelength(scan, TRUTH + [0, 0, 0, 0.8])
    assert not shifted.passed
    assert shifted.summary["lines_failed"] == len(shifted.measured)
    lost = evaluate_wavelength(scan, TRUTH + [0, 0, 0, 5.0])
    assert not lost.passed
    assert all(line.status in ("not found", "blend") for line in lost.lines)


def test_evaluate_uses_the_spectrums_own_axis_by_default(scan):
    spectrum = Spectrum(wavelength=np.polyval(TRUTH, np.arange(N)), values=scan, label="lamp")
    assert evaluate_wavelength(spectrum).passed
    with pytest.raises(ValueError, match="wavelength axis"):
        evaluate_wavelength(scan)


def test_too_few_lines_cannot_pass(scan):
    lines = Collection([EmissionLine(546.074, "Hg I"), EmissionLine(696.543, "Ar I")])
    evaluation = evaluate_wavelength(scan, TRUTH, lines)
    assert len(evaluation.measured) == 2
    assert all(c.passed for c in evaluation.comparison)
    assert not evaluation.passed
    assert "At least 5 measured lines" in evaluation.table()


def test_table_lists_every_line_with_its_outcome(fitted):
    text = fitted.table()
    assert text.splitlines()[0].split()[:3] == ["Line", "Reference", "nm"]
    assert "Holdout nm" in text
    assert "blend: unresolved from" in text
    assert any(row.startswith("PASS: ") for row in text.splitlines())
    assert len(fitted.rows()) == sum(line.use for line in lamp_lines())


@pytest.mark.parametrize("method", ["fit", "centroid", "peak"])
def test_line_center_methods_agree_on_a_symmetric_line(method):
    shape = LineSpreadFunction((2.0, 2.0, 2.0), 0.0, 0.0)
    y = 10 + 1000 * shape.profiles(np.arange(60), [30.3], 60)[:, 0]
    center, _ = line_center(y, (20, 40), method=method, peak_pixel=30.3, fwhm_px=4.7)
    assert center == pytest.approx(30.3, abs=0.02)


def test_line_center_rejects_short_windows():
    with pytest.raises(ValueError, match="7 valid samples"):
        line_center(np.ones(50), (10, 14), method="centroid")


def test_cli_calibrate_then_evaluate_round_trip(tmp_path, scan, capsys):
    from gsmath.scripts.wavelength import main
    path = tmp_path / "lamp.csv"
    np.savetxt(path, np.column_stack((np.arange(N), scan)), delimiter=",",
               header="pixel,signal", comments="")
    output = tmp_path / "out"
    assert main(["calibrate", str(path), "--direction", "increasing", "--output", str(output)]) == 0
    printed = capsys.readouterr().out
    assert "PASS:" in printed and "Coefficients c0..c3 (ascending)" in printed
    for name in ("lines.csv", "summary.json", "table.txt", "wavelength.png",
                 "coefficients.json", "calibration.gsmath", "calibrated_scan.gsmath"):
        assert (output / name).exists()
    assert main(["evaluate", str(path), "--calibration", str(output / "coefficients.json")]) == 0
    assert main(["evaluate", str(path), "--calibration", str(output / "calibration.gsmath")]) == 0
    shifted = ",".join(str(c) for c in (TRUTH + [0, 0, 0, 0.8])[::-1])
    assert main(["evaluate", str(path), "--coefficients", shifted]) == 1
    assert main(["evaluate", str(path), "--coefficients", "1,x"]) == 2


def test_isolated_lines_pass_the_crowding_cross_check(fitted):
    clean = [line for line in fitted.measured if line.label in
             ("Hg I 404.656", "Hg I 546.074", "Ar I 696.543", "Ar I 826.452", "Ar I 866.794")]
    assert len(clean) == 5
    assert all(not line.crowded and abs(line.cross_check_nm) < 0.1 for line in clean)


def test_unlisted_neighbour_flags_a_line_as_crowded(scan):
    center = truth_pixel(546.074)
    shape = LineSpreadFunction((1.5, 1.5, 1.5), 0.3, 0.5)
    crowded = scan + 1e4 * shape.profiles(np.arange(N), [center + 7.0], N)[:, 0]
    evaluation = evaluate_wavelength(crowded, TRUTH)
    line = next(line for line in evaluation.lines if line.label == "Hg I 546.074")
    assert line.status == "measured" and line.crowded
    assert "crowded: centroid differs" in evaluation.table()
    assert evaluation.summary["lines_crowded"] >= 1
