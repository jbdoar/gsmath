"""Public API for spectral mathematics, analysis, and reference data."""

from .sampled import Sampled
from .series import Series
from .spectrum import Spectrum
from .image import Image
from .pixelmap import PixelMap
from .linecut import LineCut
from .roi import ROI, Rectangle, Circle, Polygon, Mask
from .perspective import Quad, find_display, homography
from .collection import Collection
from .measurement import Measurement
from .exposures import ExposureSeries
from .bracket import Bracketed, bracket
from .calibration import (
    Calibration,
    Step,
    SubtractDark,
    FlatField,
    ColorMatrix,
    PerExposure,
    Responsivity,
    StrayLight,
    WavelengthAxis,
)
from .cleaning import AdaptiveSmooth, Despike
from .mosaic import BlackLevel, Demosaic, Mosaic, demosaic, site_means
from .colorimeter import ColorMatrixFit, SpectralSensitivity, fit_color_matrix
from .acquisition import (DarkModel, Linearity, Linearize, MaskSaturated,
                          SubtractDarkModel, build_flat)
from .wavelength import (
    HGAR_LINES_NM, CalibrationPeak, LineMatch, WavelengthCalibration,
    detect_calibration_peaks, identify_calibration_lines, calibrate_wavelength,
)
from .linefit import (LineSpreadFunction, JointLineFit, FactoryLineModel,
                      fit_lamp_lines, fit_factory_lines)
from .wavelength_validation import HeldOutLine, WavelengthValidation, validate_factory_lines
from .wavelength_lines import (EmissionLine, MeasuredLine, WavelengthEvaluation, lamp_lines,
                               line_center, measure_lines, evaluate_wavelength,
                               fit_wavelength, plot_evaluation)

from .comparison import (
    Comparison,
    ComparisonRecord,
    compare,
)

from .plotting import SpectrumGraph

from .analysis import (
    PeakAnalysis,
    ColorAnalysis,
    xyz_to_lab,
    xyz_to_luv,
    xyz_to_xy,
    xyz_to_uv,
    xyz_to_uv_prime,
    xyz_to_cct,
    reference_white_XYZ,
    delta_e_2000,
    PowerAnalysis,
    StatisticsAnalysis,
    ImageStatistics,
    Histogram,
    ImageColor,
    TristimulusColor,
    EdgeMTF,
    edge_mtf,
    Uniformity,
    FullFieldUniformity,
    grid,
)

from .profile import (
    Field,
    Profile,
    from_analysis,
)

from .reference import (
    ColorMatchingFunction,
    CMFS,
    Illuminant,
    ILLUMINANTS,
)

from .export import (
    to_dict,
    to_dataframe,
    print_result,
)

from .straylight import StrayLightCorrection, StrayLightDistribution

from .report import report

from .io.formats import read_any as read

__all__ = [
    # Core
    "read",
    "Sampled",
    "Series",
    "Spectrum",
    "Image",
    "PixelMap",
    "LineCut",
    "ROI",
    "Rectangle",
    "Circle",
    "Polygon",
    "Mask",
    "Quad",
    "find_display",
    "homography",
    "Collection",
    "Measurement",
    "ExposureSeries",
    "bracket",
    "Bracketed",
    "Calibration",
    "Step",
    "SubtractDark",
    "FlatField",
    "ColorMatrix",
    "Despike",
    "AdaptiveSmooth",
    "Demosaic",
    "BlackLevel",
    "demosaic",
    "site_means",
    "Mosaic",
    "SpectralSensitivity",
    "ColorMatrixFit",
    "DarkModel",
    "SubtractDarkModel",
    "MaskSaturated",
    "Linearity",
    "Linearize",
    "build_flat",
    "PerExposure",
    "Responsivity",
    "StrayLight",
    "WavelengthAxis",
    "HGAR_LINES_NM",
    "CalibrationPeak",
    "LineMatch",
    "WavelengthCalibration",
    "detect_calibration_peaks",
    "identify_calibration_lines",
    "calibrate_wavelength",
    "EmissionLine",
    "MeasuredLine",
    "WavelengthEvaluation",
    "lamp_lines",
    "line_center",
    "measure_lines",
    "evaluate_wavelength",
    "fit_wavelength",
    "plot_evaluation",
    "LineSpreadFunction",
    "JointLineFit",
    "FactoryLineModel",
    "fit_lamp_lines",
    "fit_factory_lines",
    "HeldOutLine",
    "WavelengthValidation",
    "validate_factory_lines",

    # Analysis
    "PeakAnalysis",
    "ColorAnalysis",
    "xyz_to_lab",
    "xyz_to_luv",
    "xyz_to_xy",
    "xyz_to_uv",
    "xyz_to_uv_prime",
    "xyz_to_cct",
    "reference_white_XYZ",
    "delta_e_2000",
    "PowerAnalysis",
    "StatisticsAnalysis",
    "ImageStatistics",
    "Histogram",
    "ImageColor",
    "TristimulusColor",
    "fit_color_matrix",
    "EdgeMTF",
    "edge_mtf",
    "Uniformity",
    "FullFieldUniformity",
    "grid",

    # Profiles
    "Field",
    "from_analysis",
    "Profile",

    # Comparison
    "Comparison",
    "ComparisonRecord",
    "compare",

    # Reference data
    "ColorMatchingFunction",
    "CMFS",
    "Illuminant",
    "ILLUMINANTS",

    # Plotting
    "SpectrumGraph",

    # Export
    "to_dict",
    "to_dataframe",
    "print_result",

    # Report
    "report",
]
