"""Compare held-out predictions from peak-only and joint fourth-order fits.

python examples/validate_wavelength_calibration.py NARROW.csv WORKING.csv \
    --holdout 404.656 546.074
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gsmath import Calibration, Spectrum, calibrate_wavelength, validate_factory_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('narrow', type=Path)
    parser.add_argument('working', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/wlcal/validation'))
    parser.add_argument('--narrow-initial', type=Path)
    parser.add_argument('--working-initial', type=Path)
    parser.add_argument('--holdout', type=float, nargs='+', required=True,
                        help='Reviewed, resolvable reference wavelengths to withhold in turn')
    parser.add_argument('--max-nfev', type=int, default=500)
    parser.add_argument('--max-error-nm', type=float, default=0.5)
    parser.add_argument('--max-rms-nm', type=float)
    parser.add_argument('--working-range-nm', type=float, nargs=2)
    args = parser.parse_args()
    scans = [Spectrum.from_lighttouch(p) for p in (args.narrow, args.working)]
    initials = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tag, scan, path in zip(['narrow', 'working'], scans, [args.narrow_initial, args.working_initial]):
        initial = Calibration.read(path).steps[0].solution if path else calibrate_wavelength(scan)
        initials.append(initial)
        print(f'{tag}: {len(initial.matches)} initial peak identities', flush=True)
        Calibration(f'{tag} peak identities', [initial.to_step()]).write(
            args.output.with_name(args.output.name+'_'+tag+'_initial').with_suffix('.gsmath'))
    result = validate_factory_lines(*scans, args.holdout, narrow_initial=initials[0],
        working_initial=initials[1], max_nfev=args.max_nfev,
        progress=lambda message: print(message, flush=True))
    payload = result.to_dict()
    payload['acceptance'] = result.assess(max_error_nm=args.max_error_nm, max_rms_nm=args.max_rms_nm,
                                        working_range_nm=args.working_range_nm)
    payload['limits_nm'] = {'maximum': args.max_error_nm, 'rms': args.max_rms_nm,
                           'working_range': args.working_range_nm}
    args.output.with_suffix('.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    if result.records:
        with args.output.with_suffix('.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(result.records[0])))
            writer.writeheader()
            writer.writerows(asdict(row) for row in result.records)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, scan in zip(axes, ['narrow', 'working']):
        rows = [r for r in result.records if r.scan == scan]
        for label, attr, offset, color in [('Peak only', 'peak_error_nm', -1.5, 'tab:blue'),
                                         ('Joint profiles', 'joint_error_nm', 1.5, 'tab:orange')]:
            for extrapolated, marker in [(False, 'o'), (True, 'X')]:
                selected = [r for r in rows if r.extrapolated == extrapolated]
                ax.scatter([r.wavelength_nm+offset for r in selected], [getattr(r, attr) for r in selected],
                           label=label+(' (extrapolation)' if extrapolated else ''), marker=marker, color=color)
        ax.axhline(0, color='gray', lw=.8)
        ax.set(title=scan.title()+' slit: held-out errors', xlabel='Reference wavelength (nm)', ylabel='Error (nm)')
        ax.legend(fontsize=8)
    fig.suptitle(f'Fourth-order fits; {len(result.failures)} failed folds retained in JSON report')
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(json.dumps({'all': result.summary(), 'interpolation': result.summary(interpolation_only=True),
                      'failures': result.failures}, indent=2), flush=True)


if __name__ == '__main__':
    main()
