"""Analysis objects created by :class:`gsmath.Spectrum` and :class:`gsmath.Image`.

Feature modules are imported on first access; resolved exports are cached.
"""

from importlib import import_module as _import_module


_LAZY_IMPORTS = {
    'AnalysisResult': ('.base', 'AnalysisResult'),
    'PeakAnalysis': ('.peaks', 'PeakAnalysis'),
    'PowerAnalysis': ('.power', 'PowerAnalysis'),
    'ColorAnalysis': ('.color', 'ColorAnalysis'),
    'Colorimetry': ('.colorimetry', 'Colorimetry'),
    'reference_white_XYZ': ('.colorimetry', 'reference_white_XYZ'),
    'xyz_to_cct': ('.colorimetry', 'xyz_to_cct'),
    'xyz_to_lab': ('.colorimetry', 'xyz_to_lab'),
    'xyz_to_luv': ('.colorimetry', 'xyz_to_luv'),
    'xyz_to_uv': ('.colorimetry', 'xyz_to_uv'),
    'xyz_to_uv_prime': ('.colorimetry', 'xyz_to_uv_prime'),
    'xyz_to_xy': ('.colorimetry', 'xyz_to_xy'),
    'StatisticsAnalysis': ('.statistics', 'StatisticsAnalysis'),
    'Histogram': ('.image', 'Histogram'),
    'ImageStatistics': ('.image', 'ImageStatistics'),
    'ImageColor': ('.image_color', 'ImageColor'),
    'TristimulusColor': ('.image_color', 'TristimulusColor'),
    'EdgeMTF': ('.mtf', 'EdgeMTF'),
    'edge_mtf': ('.mtf', 'edge_mtf'),
    'FullFieldUniformity': ('.uniformity', 'FullFieldUniformity'),
    'Uniformity': ('.uniformity', 'Uniformity'),
    'grid': ('.uniformity', 'grid'),
    'delta_e_2000': ('.delta_e', 'delta_e_2000'),
}


__all__ = ['AnalysisResult', 'PeakAnalysis', 'PowerAnalysis', 'ColorAnalysis', 'xyz_to_lab', 'xyz_to_luv', 'xyz_to_xy', 'xyz_to_uv', 'xyz_to_uv_prime', 'xyz_to_cct', 'reference_white_XYZ', 'Colorimetry', 'StatisticsAnalysis', 'ImageStatistics', 'Histogram', 'ImageColor', 'TristimulusColor', 'EdgeMTF', 'edge_mtf', 'Uniformity', 'FullFieldUniformity', 'grid', 'delta_e_2000']


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(_import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _LAZY_IMPORTS.keys())
