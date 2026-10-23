import numpy as np
import pytest

from gsmath._utils import get_nested, map2_nested, map_nested, safe_divide, set_nested, to_builtin
from gsmath.export import to_dict


def test_to_builtin_numpy_values():
    assert to_builtin(np.float64(1.5)) == 1.5
    assert to_builtin(np.array([1, 2, 3])) == [1, 2, 3]


def test_to_builtin_recurses_through_containers():
    value = {"array": np.array([1, 2]), "scalar": np.int64(3)}
    assert to_builtin(value) == {"array": [1, 2], "scalar": 3}


def test_get_and_set_nested():
    mapping = {"color": {"xy": {"y": 0.4}}}
    set_nested(mapping, "color.xy.x", 0.3)
    assert get_nested(mapping, "color.xy.x") == 0.3
    assert mapping == {"color": {"xy": {"x": 0.3, "y": 0.4}}}


@pytest.mark.parametrize("path", ["color.uv", "color.xy.x.value"])
def test_get_nested_missing_path(path):
    with pytest.raises(KeyError, match=path):
        get_nested({"color": {"xy": {"x": 0.3}}}, path)


def test_map_nested_receives_paths_and_values():
    result = map_nested(
        {"color": {"x": 0.3, "y": 0.4}},
        lambda path, value: (".".join(path), value * 2),
    )
    assert result == {
        "color": {"x": ("color.x", 0.6), "y": ("color.y", 0.8)}
    }


def test_map2_nested_combines_matching_trees():
    result = map2_nested(
        {"a": {"b": 3}},
        {"a": {"b": 2}},
        lambda path, left, right: (".".join(path), left - right),
    )
    assert result == {"a": {"b": ("a.b", 1)}}


def test_map2_nested_rejects_key_mismatch():
    with pytest.raises(ValueError, match="Key mismatch"):
        map2_nested({"a": 1}, {"b": 1}, lambda *_: None)


def test_to_dict_flat():
    value = {"color": {"xy": {"x": np.float64(0.3)}}}
    assert to_dict(value, flat=True) == {"color.xy.x": 0.3}


def test_safe_divide():
    result = safe_divide(np.array([4.0, 2.0]), np.array([2.0, 0.0]))
    assert result[0] == 2.0
    assert np.isnan(result[1])


def test_safe_divide_scalars():
    assert safe_divide(6, 3) == pytest.approx(2.0)
