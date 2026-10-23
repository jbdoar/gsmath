"""Perceptual color-difference calculations."""

from __future__ import annotations

import numpy as np


def delta_e_2000(Lab1, Lab2, *, k_L=1.0, k_C=1.0, k_H=1.0):
    """Return the CIEDE2000 color difference between CIELAB values.

    Inputs may be individual ``(L*, a*, b*)`` triples or broadcast-compatible
    arrays whose final dimension has length three. The weighting factors use
    the CIE reference conditions by default.
    """
    Lab1 = np.asarray(Lab1, dtype=float)
    Lab2 = np.asarray(Lab2, dtype=float)

    if Lab1.ndim == 0 or Lab1.shape[-1] != 3:
        raise ValueError("Lab1 must have a final dimension of length three.")
    if Lab2.ndim == 0 or Lab2.shape[-1] != 3:
        raise ValueError("Lab2 must have a final dimension of length three.")

    try:
        Lab1, Lab2 = np.broadcast_arrays(Lab1, Lab2)
    except ValueError as exc:
        raise ValueError("Lab inputs must have broadcast-compatible shapes.") from exc

    weights = np.asarray([k_L, k_C, k_H], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("k_L, k_C, and k_H must be finite and positive.")

    L1, a1, b1 = np.moveaxis(Lab1, -1, 0)
    L2, a2, b2 = np.moveaxis(Lab2, -1, 0)

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2
    C_bar_7 = C_bar**7
    G = 0.5 * (1 - np.sqrt(C_bar_7 / (C_bar_7 + 25**7)))

    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    C1_prime = np.hypot(a1_prime, b1)
    C2_prime = np.hypot(a2_prime, b2)
    h1_prime = _hue_degrees(a1_prime, b1)
    h2_prime = _hue_degrees(a2_prime, b2)

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    delta_h_prime = h2_prime - h1_prime
    zero_chroma = C1_prime * C2_prime == 0
    delta_h_prime = np.where(zero_chroma, 0.0, delta_h_prime)
    delta_h_prime = np.where(delta_h_prime > 180, delta_h_prime - 360, delta_h_prime)
    delta_h_prime = np.where(delta_h_prime < -180, delta_h_prime + 360, delta_h_prime)
    delta_H_prime = (
        2
        * np.sqrt(C1_prime * C2_prime)
        * np.sin(np.radians(delta_h_prime / 2))
    )

    L_bar_prime = (L1 + L2) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2
    h_sum = h1_prime + h2_prime
    h_difference = np.abs(h1_prime - h2_prime)
    h_bar_prime = np.where(
        zero_chroma,
        h_sum,
        np.where(
            h_difference <= 180,
            h_sum / 2,
            np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2),
        ),
    )

    T = (
        1
        - 0.17 * np.cos(np.radians(h_bar_prime - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3 * h_bar_prime + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar_prime - 63))
    )
    delta_theta = 30 * np.exp(-((h_bar_prime - 275) / 25) ** 2)
    C_bar_prime_7 = C_bar_prime**7
    R_C = 2 * np.sqrt(C_bar_prime_7 / (C_bar_prime_7 + 25**7))

    lightness_offset = L_bar_prime - 50
    S_L = 1 + 0.015 * lightness_offset**2 / np.sqrt(
        20 + lightness_offset**2
    )
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T
    R_T = -np.sin(np.radians(2 * delta_theta)) * R_C

    L_term = delta_L_prime / (k_L * S_L)
    C_term = delta_C_prime / (k_C * S_C)
    H_term = delta_H_prime / (k_H * S_H)
    result = np.sqrt(
        np.maximum(
            0.0,
            L_term**2 + C_term**2 + H_term**2 + R_T * C_term * H_term,
        )
    )

    return result.item() if result.ndim == 0 else result


def _hue_degrees(a_prime, b):
    """Return hue angles in the CIEDE2000 interval from zero to 360."""
    hue = np.mod(np.degrees(np.arctan2(b, a_prime)), 360.0)
    return np.where((a_prime == 0) & (b == 0), 0.0, hue)


__all__ = ["delta_e_2000"]
