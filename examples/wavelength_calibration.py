"""Inspect a LightTouch lamp scan: python examples/wavelength_calibration.py FILE.

Outputs a residual table and a diagnostic PNG alongside the requested output
prefix. The input scan is never modified. Bounds are optional instrument priors.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gsmath import Calibration, Spectrum, calibrate_wavelength


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scan', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/wlcal/result'))
    parser.add_argument('--wavelength-bounds', nargs=2, type=float)
    parser.add_argument('--dispersion-bounds', nargs=2, type=float)
    parser.add_argument('--direction', choices=['either', 'increasing', 'decreasing'], default='either')
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--degree', type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument('--tolerance', type=float, default=1.0)
    args = parser.parse_args()
    scan = Spectrum.from_lighttouch(args.scan)
    result = calibrate_wavelength(scan, degree=args.degree, iterations=args.iterations, tolerance_nm=args.tolerance,
        wavelength_bounds_nm=args.wavelength_bounds, dispersion_bounds=args.dispersion_bounds,
        direction=args.direction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Calibration('HgAr wavelength', [result.to_step()]).write(
        args.output.with_suffix('.gsmath'))
    scan.calibrate_wavelength(result).write(
        args.output.with_name(args.output.name + '_calibrated').with_suffix('.gsmath'))
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['peak_index', 'pixel', 'reference_nm', 'predicted_nm', 'residual_nm'])
        for m in result.matches:
            writer.writerow([m.peak_index, m.pixel, m.wavelength_nm, m.predicted_nm, m.residual_nm])
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(np.arange(len(scan.values)), scan.values, lw=0.8)
    for m in result.matches:
        axes[0].axvline(m.pixel, color='tab:orange', alpha=0.4)
        axes[0].text(m.pixel, 0.95, f'{m.wavelength_nm:.3f}', rotation=90,
                     va='top', transform=axes[0].get_xaxis_transform(), fontsize=7)
    axes[0].set_ylabel('Signal')
    axes[0].set_title(f'{len(result.matches)}/{len(result.peaks)} peaks matched; RMS {result.rms_nm:.3f} nm')
    axes[1].scatter([m.pixel for m in result.matches], [m.residual_nm for m in result.matches])
    axes[1].axhline(0, color='gray', lw=0.8)
    axes[1].set(xlabel='Sample index (pixel)', ylabel='Residual (nm)')
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(f'{len(result.matches)}/{len(result.peaks)} matches; RMS {result.rms_nm:.4f} nm; '
          f'max {result.max_error_nm:.4f} nm; pixel coverage {result.matched_pixel_fraction:.1%}')
    print('Descending-order coefficients:', result.coefficients)


if __name__ == '__main__':
    main()
