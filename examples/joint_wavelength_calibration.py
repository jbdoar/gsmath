"""Fit a factory lamp pair and save a reusable field model and diagnostics.

python examples/joint_wavelength_calibration.py NARROW.csv WORKING.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gsmath import Calibration, Spectrum, calibrate_wavelength, fit_factory_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('narrow', type=Path)
    parser.add_argument('working', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/wlcal/joint_factory'))
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--max-nfev', type=int, default=500)
    args = parser.parse_args()
    narrow, working = [Spectrum.from_lighttouch(p) for p in (args.narrow, args.working)]
    initial = calibrate_wavelength(narrow, iterations=args.iterations)
    print(f'RANSAC initialized {len(initial.matches)} matches; final order {len(initial.coefficients)-1}', flush=True)
    factory, result = fit_factory_lines(narrow, working, initial=initial, max_nfev=args.max_nfev,
        metadata={'narrow_source': args.narrow.name, 'working_source': args.working.name,
                  'reference_status': 'provisional HgAr list', 'status': 'candidate for review'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    factory.write(args.output.with_suffix('.gsmath'))
    for tag, calibration in zip(['narrow', 'working'], result.calibrations):
        Calibration(f'Joint HgAr {tag}', [calibration.to_step()]).write(
            args.output.with_name(args.output.name+'_'+tag).with_suffix('.gsmath'))
    report = {'normalized_signal_rms': result.normalized_rms,
              'warnings': result.warnings, 'nfev': result.nfev,
              'calibrations': [c.report for c in result.calibrations]}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez(args.output.with_suffix('.npz'), narrow_model=result.models[0], working_model=result.models[1],
             narrow_residual=result.residuals[0], working_residual=result.residuals[1],
             narrow_components=result.components[0], working_components=result.components[1],
             narrow_background=result.backgrounds[0], working_background=result.backgrounds[1],
             reference_lines=result.reference_lines, relative_areas=result.relative_areas)
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    for s, (scan, label) in enumerate(zip([narrow, working], ['Narrow slit', 'Working slit'])):
        x = np.arange(len(scan.values))
        axes[0, s].plot(x, scan.values, label='Measured', lw=1)
        axes[0, s].plot(x, result.models[s], label='Joint model', lw=1)
        axes[0, s].set(title=f'{label}: normalized RMS {result.normalized_rms[s]:.2%}', ylabel='Signal')
        axes[0, s].legend()
        axes[1, s].plot(x, result.residuals[s], lw=1)
        axes[1, s].axhline(0, color='gray', lw=.5)
        axes[1, s].set(xlabel='Sample index', ylabel='Measured minus modeled')
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for s, scan in enumerate([narrow, working]):
        wave = result.calibrations[s].wavelength(np.arange(len(scan.values)))
        for ax, (lo, hi) in zip(axes[s], [(571, 584), (745, 758), (794, 806)]):
            mask = (wave >= lo) & (wave <= hi)
            ax.plot(wave[mask], scan.values[mask], '.-', label='Measured')
            ax.plot(wave[mask], result.models[s][mask], label='Total fit')
            for j, line in enumerate(result.reference_lines):
                if lo < line < hi:
                    ax.plot(wave[mask], result.components[s][mask, j]+result.backgrounds[s][mask],
                            '--', label=f'{line:.3f} nm')
            ax.set(xlabel='Joint-fit wavelength (nm)', ylabel='Signal',
                   title=('Narrow' if s == 0 else 'Working') + ' slit')
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output.with_name(args.output.name+'_blends').with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(json.dumps({'normalized_signal_rms': result.normalized_rms,
                      'warnings': result.warnings, 'nfev': result.nfev}, indent=2), flush=True)


if __name__ == '__main__':
    main()
