"""Analysis objects created by :class:`gsmath.Spectrum` and :class:`gsmath.Image`."""

from .base import AnalysisResult
from .peaks import PeakAnalysis
from .power import PowerAnalysis
from .color import ColorAnalysis
from .colorimetry import (Colorimetry, reference_white_XYZ, xyz_to_cct, xyz_to_lab,
                          xyz_to_luv, xyz_to_uv, xyz_to_uv_prime, xyz_to_xy)
from .statistics import StatisticsAnalysis
from .image import Histogram, ImageStatistics
from .image_color import ImageColor, TristimulusColor
from .mtf import EdgeMTF, edge_mtf
from .uniformity import FullFieldUniformity, Uniformity, grid
from .delta_e import delta_e_2000

__all__ = [
    "AnalysisResult",
    "PeakAnalysis",
    "PowerAnalysis",
    "ColorAnalysis",
    "xyz_to_lab",
    "xyz_to_luv",
    "xyz_to_xy",
    "xyz_to_uv",
    "xyz_to_uv_prime",
    "xyz_to_cct",
    "reference_white_XYZ",
    "Colorimetry",
    "StatisticsAnalysis",
    "ImageStatistics",
    "Histogram",
    "ImageColor",
    "TristimulusColor",
    "EdgeMTF",
    "edge_mtf",
    "Uniformity",
    "FullFieldUniformity",
    "grid",
    "delta_e_2000",
]
