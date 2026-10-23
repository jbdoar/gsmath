import numpy as np
import pytest

import gsmath
from gsmath.straylight import StrayLightCorrection, StrayLightDistribution


def kernel(x):
    t = (x - 500) / 100
    return .001 * (1 + .2*t[:, None] + .5*t[None, :] + .1*t[:, None]*t[None, :])


def test_regrid_kernel_spacing_orientation_and_correction():
    old = np.array([400., 415., 450., 490., 540., 590., 650., 700.])
    new = np.array([420., 435., 480., 535., 610., 680.])
    d = StrayLightDistribution(old, kernel(old)*np.gradient(old)[None, :])
    expected = kernel(new)*np.gradient(new)[None, :]
    np.testing.assert_allclose(d.regrid(new).values, expected, atol=1e-15)
    truth = np.array([2., 10., 4., 20., 6., 1.])
    measured = (np.eye(len(new)) + expected) @ truth
    np.testing.assert_allclose(d.correction().regrid(new).apply_matrix(measured), truth, atol=1e-12)


def test_identity_grid_is_exact_independent_copy():
    x = np.arange(400., 406.)
    values = np.arange(36, dtype=float).reshape(6, 6)/1000
    values[0, 1] = -.001
    for cls in (StrayLightDistribution, StrayLightCorrection):
        original = cls(x, values)
        result = original.regrid(x)
        np.testing.assert_array_equal(result.values, values)
        assert not np.shares_memory(result.values, values)
        assert not np.shares_memory(result.wavelength, x)


def test_distribution_roundtrip_and_zero_scattering():
    x = np.linspace(400, 700, 31)
    d = StrayLightDistribution(x, kernel(x)*10)
    np.testing.assert_allclose(d.correction().distribution().values, d.values, atol=1e-14)
    result = StrayLightCorrection(x, np.eye(len(x))).regrid(x+1, bounds="edge")
    np.testing.assert_array_equal(result.values, np.eye(len(x)))


@pytest.mark.parametrize("new", [[1], [1, 1], [2, 1], [1, np.nan], [[1, 2]]])
def test_bad_grids(new):
    for cls in (StrayLightDistribution, StrayLightCorrection):
        with pytest.raises(ValueError, match="wavelength"):
            cls(np.arange(3.), np.eye(3)).regrid(new)


def test_bounds_and_matrix_validation():
    x = np.array([400., 500., 600.])
    for cls in (StrayLightDistribution, StrayLightCorrection):
        m = cls(x, np.eye(3))
        with pytest.raises(ValueError, match="outside"):
            m.regrid(x+1)
        with pytest.raises(ValueError, match="bounds"):
            m.regrid(x, bounds="guess")
        with pytest.raises(ValueError, match="matrix"):
            cls(x, np.eye(2)).regrid(x)
        with pytest.raises(ValueError, match="matrix"):
            cls(x, np.full((3, 3), np.nan)).regrid(x)
    d = StrayLightDistribution(x, kernel(x)*100)
    new = np.array([390., 500., 620.])
    expected = kernel(np.clip(new, 400, 600))*np.gradient(new)[None, :]
    np.testing.assert_allclose(d.regrid(new, bounds="edge").values, expected)


def test_singular_correction_cannot_be_recovered():
    with pytest.raises(np.linalg.LinAlgError):
        StrayLightCorrection(np.arange(3.), np.zeros((3, 3))).distribution()


@pytest.mark.parametrize("cls", [StrayLightDistribution, StrayLightCorrection])
def test_native_matrix_retains_grid_dtype_and_kind(cls, tmp_path):
    grid = np.array([400., 450., 500.], dtype=np.float32)
    values = np.eye(3, dtype=np.float64) * 0.25
    matrix = cls(grid, values)
    path = matrix.write(tmp_path / "nested" / "matrix.gsmath")
    restored = cls.read(path)
    assert isinstance(gsmath.read(path), cls)
    assert restored.wavelength.dtype == grid.dtype
    assert restored.values.dtype == values.dtype
    np.testing.assert_array_equal(restored.wavelength, grid)
    np.testing.assert_array_equal(restored.values, values)
    with pytest.raises(ValueError, match="wavelength grid differs"):
        cls.read(path, grid + 1)
    other = StrayLightCorrection if cls is StrayLightDistribution else StrayLightDistribution
    with pytest.raises(ValueError, match="Not a"):
        other.read(path)


def test_legacy_binary_requires_explicit_grid(tmp_path):
    grid = np.array([400., 450., 500.])
    matrix = StrayLightCorrection(grid, np.eye(3))
    path = matrix.write(tmp_path / "matrix.bin")
    with pytest.raises(ValueError, match="require a wavelength grid"):
        StrayLightCorrection.read(path)
    restored = StrayLightCorrection.read(path, grid)
    assert restored.values.dtype == np.float32
    np.testing.assert_array_equal(restored.values, np.eye(3))
