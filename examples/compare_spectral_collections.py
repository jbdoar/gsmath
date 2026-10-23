"""Compare paired spectral scans listed in a CSV manifest.

Manifest columns: label,left,right. Paths are relative to the manifest file.
Each row represents one pair of measurements with the same label.

Run::

    python examples/compare_spectral_collections.py pairs.csv \
        --left-label instrument-a --right-label instrument-b
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

import gsmath as gs
from gsmath import Profile, from_analysis


def comparison_profile(observer="2deg"):
    """Compare peak positions, luminance, and chromaticity."""
    peaks = ["peak_wavelength_polyfit", "centroid_wavelength", "center_wavelength"]
    return Profile(
        name="spectral comparison",
        fields={
            "peaks": from_analysis("peaks", peaks),
            "color": {observer: from_analysis("color", ["Y", "x", "y"],
                                              options={"observer": observer},
                                              relative=["Y"])},
        },
    ).with_limits({
        "peaks": {name: {"tolerance": 1} for name in peaks},
        "color": {observer: {"Y": {"tolerance": 0.025}, "x": {"tolerance": 0.005},
                             "y": {"tolerance": 0.005}}},
    })


def load_pairs(manifest, *, spectrum_format="lighttouch", left_label="left",
               right_label="right"):
    """Build label-matched collections from a manifest of input paths."""
    manifest = Path(manifest)
    left, right = [], []
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"label", "left", "right"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("manifest must have label,left,right columns")
        for row_number, row in enumerate(reader, start=2):
            label = row["label"].strip()
            if not label or not row["left"] or not row["right"]:
                raise ValueError(f"manifest row {row_number} needs label,left,right")
            paths = [manifest.parent / row[side].strip() for side in ("left", "right")]
            left.append(gs.Spectrum.read(paths[0], format=spectrum_format).copy(label=label))
            right.append(gs.Spectrum.read(paths[1], format=spectrum_format).copy(label=label))
    if not left:
        raise ValueError("manifest contains no scan pairs")
    return gs.Collection(left, label=left_label), gs.Collection(right, label=right_label)


def write_plots(left, right, directory):
    """Write linear and logarithmic overlays for each matched pair."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index, (first, second) in enumerate(zip(left, right), start=1):
        stem = f"{index:03d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', first.label)}"
        graph = gs.SpectrumGraph(title=first.label)
        graph.add(first, label=left.label)
        graph.add(second, label=right.label)
        graph.figure.savefig(directory / f"{stem}_linear.png")
        graph.set_yscale("log")
        graph.figure.savefig(directory / f"{stem}_log.png")
        plt.close(graph.figure)


def parser():
    p = argparse.ArgumentParser(
        description="Compare labeled pairs of spectra from a CSV manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("manifest", type=Path, help="CSV with label,left,right columns")
    p.add_argument("--left-label", default="left", help="Name for the first collection")
    p.add_argument("--right-label", default="right", help="Name for the second collection")
    p.add_argument("--format", choices=["lighttouch", "gsmath", "csv"],
                   default="lighttouch", help="Input spectrum format")
    p.add_argument("--observer", choices=["2deg", "10deg"], default="2deg")
    p.add_argument("-o", "--output", type=Path,
                   default=Path("artifacts/compare_spectral_collections/report.csv"),
                   help="Comparison report CSV")
    p.add_argument("--no-plots", action="store_true", help="Skip overlay plots")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    left, right = load_pairs(
        args.manifest, spectrum_format=args.format,
        left_label=args.left_label, right_label=args.right_label,
    )
    result = gs.compare(left, right, profile=comparison_profile(args.observer))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gs.report(result).to_csv(args.output, index=False)
    if not args.no_plots:
        write_plots(left, right, args.output.parent / "plots")
    print(f"Compared {len(left)} pairs; report: {args.output}")


if __name__ == "__main__":
    main()
