"""Paths to reference data shipped inside the Python package."""

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CMFS_DIR = DATA_DIR / "color_matching_functions"
ILLUMINANTS_DIR = DATA_DIR / "illuminants"
ACTION_SPECTRA_DIR = DATA_DIR / "action_spectra"

ROBERTSON_TABLE = DATA_DIR / "robertson_table.csv"
