"""Analytic line shapes used by fitting routines."""

import numpy as np


def gaussian(x, amp, mu, sigma, offset):
    """Evaluate a Gaussian with amplitude, center, width, and baseline."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

# lorentzian
# voigt
# etc
