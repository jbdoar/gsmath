import numpy as np
import pytest

from gsmath import Spectrum, delta_e_2000


@pytest.mark.parametrize(
    ("Lab1", "Lab2", "expected"),
    [
        ([50.0000, 2.6772, -79.7751], [50.0000, 0.0000, -82.7485], 2.0425),
        ([50.0000, 3.1571, -77.2803], [50.0000, 0.0000, -82.7485], 2.8615),
        ([50.0000, 2.8361, -74.0200], [50.0000, 0.0000, -82.7485], 3.4412),
        ([50.0000, -1.3802, -84.2814], [50.0000, 0.0000, -82.7485], 1.0000),
        ([50.0000, -1.1848, -84.8006], [50.0000, 0.0000, -82.7485], 1.0000),
        ([50.0000, -0.9009, -85.5211], [50.0000, 0.0000, -82.7485], 1.0000),
        ([50.0000, 0.0000, 0.0000], [50.0000, -1.0000, 2.0000], 2.3669),
        ([50.0000, -1.0000, 2.0000], [50.0000, 0.0000, 0.0000], 2.3669),
        ([50.0000, 2.4900, -0.0010], [50.0000, -2.4900, 0.0009], 7.1792),
        ([50.0000, 2.4900, -0.0010], [50.0000, -2.4900, 0.0010], 7.1792),
    ],
)
def test_delta_e_2000_matches_sharma_reference_pairs(Lab1, Lab2, expected):
    assert delta_e_2000(Lab1, Lab2) == pytest.approx(expected, abs=5e-5)


def test_delta_e_2000_is_zero_for_identical_colors():
    Lab = [53.2, 80.1, 67.2]

    assert delta_e_2000(Lab, Lab) == pytest.approx(0.0)


def test_delta_e_2000_is_symmetric():
    Lab1 = [60.0, 30.0, -40.0]
    Lab2 = [20.0, -10.0, 50.0]

    assert delta_e_2000(Lab1, Lab2) == pytest.approx(
        delta_e_2000(Lab2, Lab1)
    )


def test_delta_e_2000_broadcasts_over_color_arrays():
    colors = np.array([
        [50.0000, 2.6772, -79.7751],
        [50.0000, 0.0000, 0.0000],
    ])
    references = np.array([
        [50.0000, 0.0000, -82.7485],
        [50.0000, -1.0000, 2.0000],
    ])

    np.testing.assert_allclose(
        delta_e_2000(colors, references),
        [2.0425, 2.3669],
        atol=5e-5,
    )


def test_delta_e_2000_accepts_application_weighting_factors():
    Lab1 = [40.0, 10.0, 20.0]
    Lab2 = [60.0, 10.0, 20.0]

    unweighted = delta_e_2000(Lab1, Lab2)
    weighted = delta_e_2000(Lab1, Lab2, k_L=2)

    assert weighted == pytest.approx(unweighted / 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k_L": 0},
        {"k_C": -1},
        {"k_H": np.nan},
    ],
)
def test_delta_e_2000_rejects_invalid_weighting_factors(kwargs):
    with pytest.raises(ValueError, match="finite and positive"):
        delta_e_2000([50, 0, 0], [50, 1, 1], **kwargs)


def test_delta_e_2000_validates_input_shapes():
    with pytest.raises(ValueError, match="Lab1"):
        delta_e_2000([50, 0], [50, 1, 1])
    with pytest.raises(ValueError, match="broadcast-compatible"):
        delta_e_2000(np.zeros((2, 3)), np.zeros((3, 3)))


def test_delta_e_2000_propagates_nonfinite_values():
    assert np.isnan(delta_e_2000([np.nan, 0, 0], [50, 1, 1]))


def test_color_analysis_compares_against_another_analysis():
    measured = Spectrum([400, 500, 600], [0.0, 2.0, 1.0]).color()
    reference = Spectrum([400, 500, 600], [0.0, 1.0, 2.0]).color()

    assert measured.delta_e_2000(reference) == pytest.approx(
        delta_e_2000(measured.Lab, reference.Lab)
    )


def test_color_analysis_compares_against_lab_triple():
    color = Spectrum([400, 500, 600], [0.0, 2.0, 1.0]).color()
    reference = [50.0, 0.0, 0.0]

    assert color.delta_e_2000(reference) == pytest.approx(
        delta_e_2000(color.Lab, reference)
    )


def test_color_analysis_requires_matching_observers():
    spectrum = Spectrum([400, 500, 600], [0.0, 2.0, 1.0])

    with pytest.raises(ValueError, match="matching colorimetric observers"):
        spectrum.color(observer="2deg").delta_e_2000(
            spectrum.color(observer="10deg")
        )


def test_color_analysis_requires_matching_reference_whites():
    spectrum = Spectrum([400, 500, 600], [0.0, 2.0, 1.0])

    with pytest.raises(ValueError, match="matching reference whites"):
        spectrum.color(white="D65").delta_e_2000(
            spectrum.color(white="D50")
        )
