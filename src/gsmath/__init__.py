"""Public API for spectral mathematics, analysis, and reference data.

Feature modules are imported on first access; resolved exports are cached.
"""

from importlib import import_module as _import_module

from .bracket import Bracketed, bracket
from .report import report


_LAZY_IMPORTS = {
    'Sampled': ('.sampled', 'Sampled'),
    'Series': ('.series', 'Series'),
    'Spectrum': ('.spectrum', 'Spectrum'),
    'Image': ('.image', 'Image'),
    'PixelMap': ('.pixelmap', 'PixelMap'),
    'LineCut': ('.linecut', 'LineCut'),
    'ROI': ('.roi', 'ROI'),
    'Rectangle': ('.roi', 'Rectangle'),
    'Circle': ('.roi', 'Circle'),
    'Polygon': ('.roi', 'Polygon'),
    'Mask': ('.roi', 'Mask'),
    'Quad': ('.perspective', 'Quad'),
    'find_display': ('.perspective', 'find_display'),
    'homography': ('.perspective', 'homography'),
    'Collection': ('.collection', 'Collection'),
    'Measurement': ('.measurement', 'Measurement'),
    'ExposureSeries': ('.exposures', 'ExposureSeries'),
    'Bracketed': ('.bracket', 'Bracketed'),
    'Calibration': ('.calibration', 'Calibration'),
    'Step': ('.calibration', 'Step'),
    'SubtractDark': ('.calibration', 'SubtractDark'),
    'FlatField': ('.calibration', 'FlatField'),
    'ColorMatrix': ('.calibration', 'ColorMatrix'),
    'PerExposure': ('.calibration', 'PerExposure'),
    'Responsivity': ('.calibration', 'Responsivity'),
    'StrayLight': ('.calibration', 'StrayLight'),
    'WavelengthAxis': ('.calibration', 'WavelengthAxis'),
    'AdaptiveSmooth': ('.cleaning', 'AdaptiveSmooth'),
    'Despike': ('.cleaning', 'Despike'),
    'BlackLevel': ('.mosaic', 'BlackLevel'),
    'Demosaic': ('.mosaic', 'Demosaic'),
    'Mosaic': ('.mosaic', 'Mosaic'),
    'demosaic': ('.mosaic', 'demosaic'),
    'site_means': ('.mosaic', 'site_means'),
    'ColorMatrixFit': ('.colorimeter', 'ColorMatrixFit'),
    'SpectralSensitivity': ('.colorimeter', 'SpectralSensitivity'),
    'fit_color_matrix': ('.colorimeter', 'fit_color_matrix'),
    'DarkModel': ('.acquisition', 'DarkModel'),
    'Linearity': ('.acquisition', 'Linearity'),
    'Linearize': ('.acquisition', 'Linearize'),
    'MaskSaturated': ('.acquisition', 'MaskSaturated'),
    'SubtractDarkModel': ('.acquisition', 'SubtractDarkModel'),
    'build_flat': ('.acquisition', 'build_flat'),
    'HGAR_LINES_NM': ('.wavelength', 'HGAR_LINES_NM'),
    'CalibrationPeak': ('.wavelength', 'CalibrationPeak'),
    'LineMatch': ('.wavelength', 'LineMatch'),
    'WavelengthCalibration': ('.wavelength', 'WavelengthCalibration'),
    'detect_calibration_peaks': ('.wavelength', 'detect_calibration_peaks'),
    'identify_calibration_lines': ('.wavelength', 'identify_calibration_lines'),
    'calibrate_wavelength': ('.wavelength', 'calibrate_wavelength'),
    'LineSpreadFunction': ('.linefit', 'LineSpreadFunction'),
    'JointLineFit': ('.linefit', 'JointLineFit'),
    'FactoryLineModel': ('.linefit', 'FactoryLineModel'),
    'fit_lamp_lines': ('.linefit', 'fit_lamp_lines'),
    'fit_factory_lines': ('.linefit', 'fit_factory_lines'),
    'HeldOutLine': ('.wavelength_validation', 'HeldOutLine'),
    'WavelengthValidation': ('.wavelength_validation', 'WavelengthValidation'),
    'validate_factory_lines': ('.wavelength_validation', 'validate_factory_lines'),
    'Comparison': ('.comparison', 'Comparison'),
    'ComparisonRecord': ('.comparison', 'ComparisonRecord'),
    'compare': ('.comparison', 'compare'),
    'SpectrumGraph': ('.plotting', 'SpectrumGraph'),
    'PeakAnalysis': ('.analysis', 'PeakAnalysis'),
    'ColorAnalysis': ('.analysis', 'ColorAnalysis'),
    'xyz_to_lab': ('.analysis', 'xyz_to_lab'),
    'xyz_to_luv': ('.analysis', 'xyz_to_luv'),
    'xyz_to_xy': ('.analysis', 'xyz_to_xy'),
    'xyz_to_uv': ('.analysis', 'xyz_to_uv'),
    'xyz_to_uv_prime': ('.analysis', 'xyz_to_uv_prime'),
    'xyz_to_cct': ('.analysis', 'xyz_to_cct'),
    'reference_white_XYZ': ('.analysis', 'reference_white_XYZ'),
    'delta_e_2000': ('.analysis', 'delta_e_2000'),
    'PowerAnalysis': ('.analysis', 'PowerAnalysis'),
    'StatisticsAnalysis': ('.analysis', 'StatisticsAnalysis'),
    'ImageStatistics': ('.analysis', 'ImageStatistics'),
    'Histogram': ('.analysis', 'Histogram'),
    'ImageColor': ('.analysis', 'ImageColor'),
    'TristimulusColor': ('.analysis', 'TristimulusColor'),
    'EdgeMTF': ('.analysis', 'EdgeMTF'),
    'edge_mtf': ('.analysis', 'edge_mtf'),
    'Uniformity': ('.analysis', 'Uniformity'),
    'FullFieldUniformity': ('.analysis', 'FullFieldUniformity'),
    'grid': ('.analysis', 'grid'),
    'Field': ('.profile', 'Field'),
    'Profile': ('.profile', 'Profile'),
    'from_analysis': ('.profile', 'from_analysis'),
    'ColorMatchingFunction': ('.reference', 'ColorMatchingFunction'),
    'CMFS': ('.reference', 'CMFS'),
    'Illuminant': ('.reference', 'Illuminant'),
    'ILLUMINANTS': ('.reference', 'ILLUMINANTS'),
    'to_dict': ('.export', 'to_dict'),
    'to_dataframe': ('.export', 'to_dataframe'),
    'print_result': ('.export', 'print_result'),
    'StrayLightCorrection': ('.straylight', 'StrayLightCorrection'),
    'StrayLightDistribution': ('.straylight', 'StrayLightDistribution'),
    'read': ('.io.formats', 'read_any'),
}


__all__ = ['read', 'Sampled', 'Series', 'Spectrum', 'Image', 'PixelMap', 'LineCut', 'ROI', 'Rectangle', 'Circle', 'Polygon', 'Mask', 'Quad', 'find_display', 'homography', 'Collection', 'Measurement', 'ExposureSeries', 'bracket', 'Bracketed', 'Calibration', 'Step', 'SubtractDark', 'FlatField', 'ColorMatrix', 'Despike', 'AdaptiveSmooth', 'Demosaic', 'BlackLevel', 'demosaic', 'site_means', 'Mosaic', 'SpectralSensitivity', 'ColorMatrixFit', 'DarkModel', 'SubtractDarkModel', 'MaskSaturated', 'Linearity', 'Linearize', 'build_flat', 'PerExposure', 'Responsivity', 'StrayLight', 'WavelengthAxis', 'HGAR_LINES_NM', 'CalibrationPeak', 'LineMatch', 'WavelengthCalibration', 'detect_calibration_peaks', 'identify_calibration_lines', 'calibrate_wavelength', 'LineSpreadFunction', 'JointLineFit', 'FactoryLineModel', 'fit_lamp_lines', 'fit_factory_lines', 'HeldOutLine', 'WavelengthValidation', 'validate_factory_lines', 'PeakAnalysis', 'ColorAnalysis', 'xyz_to_lab', 'xyz_to_luv', 'xyz_to_xy', 'xyz_to_uv', 'xyz_to_uv_prime', 'xyz_to_cct', 'reference_white_XYZ', 'delta_e_2000', 'PowerAnalysis', 'StatisticsAnalysis', 'ImageStatistics', 'Histogram', 'ImageColor', 'TristimulusColor', 'fit_color_matrix', 'EdgeMTF', 'edge_mtf', 'Uniformity', 'FullFieldUniformity', 'grid', 'Field', 'from_analysis', 'Profile', 'Comparison', 'ComparisonRecord', 'compare', 'ColorMatchingFunction', 'CMFS', 'Illuminant', 'ILLUMINANTS', 'SpectrumGraph', 'to_dict', 'to_dataframe', 'print_result', 'report']


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
