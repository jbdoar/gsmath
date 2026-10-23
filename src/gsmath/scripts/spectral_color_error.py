import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

from gsmath import Spectrum
from gsmath.reference.cmfs import CMFS

def analyze_color_error(s1, s2):
    
    s1 = (
        s1.interpolate(startwl=360, stopwl=830, step=1)
        .normalize(by="photometric", value=1)
    )
    
    s2 = (
        s2.interpolate(startwl=360, stopwl=830, step=1)
        .normalize(by="photometric", value=1)
    )

    delta_s = s2 - s1

    cmfs = CMFS["2deg"] 
    xbar, ybar, zbar = cmfs.xbar, cmfs.ybar, cmfs.zbar

    # delta_X = delta_s * xbar
    # delta_Y = delta_s * ybar
    # delta_Z = delta_s * zbar

    # X, Y, Z = s1.color().XYZ

    X = np.trapezoid(s1.values * xbar, s1.wavelength)
    Y = np.trapezoid(s1.values * ybar, s1.wavelength)
    Z = np.trapezoid(s1.values * zbar, s1.wavelength)
    
    T = X + Y + Z

    sensitivity_x = (xbar * T - X * (xbar + ybar + zbar)) / T**2
    sensitivity_y = (ybar * T - Y * (xbar + ybar + zbar)) / T**2

    delta_x = Spectrum(
        wavelength=delta_s.wavelength,
        values=delta_s.values * sensitivity_x,
        label="dx",
    )

    delta_y = Spectrum(
        wavelength=delta_s.wavelength,
        values=delta_s.values * sensitivity_y,
        label="dy",
    )

    dx_integrated = np.trapezoid(delta_x.values, delta_x.wavelength)
    dy_integrated = np.trapezoid(delta_y.values, delta_y.wavelength)

    x1, y1 = s1.color().xy
    x2, y2 = s2.color().xy

    dx_actual = x2 - x1
    dy_actual = y2 - y1

    print(f"dx: integrated={dx_integrated:.6g}, actual={dx_actual:.6g}")
    print(f"dy: integrated={dy_integrated:.6g}, actual={dy_actual:.6g}")

    return delta_x, delta_y


def parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two LightTouch spectra and plot how their spectral "
            "difference contributes to chromaticity error."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "spectrum1",
        metavar="REFERENCE",
        help="Path to the reference spectrum in LightTouch CSV format.",
    )
    parser.add_argument(
        "spectrum2",
        metavar="COMPARISON",
        help="Path to the comparison spectrum in LightTouch CSV format.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default="artifacts/spectral_color_error/spectral_color_error.png",  # str: same help text on every OS; type converts it
        metavar="PATH",
        help="Path at which to save the analysis plot.",
    )
    return parser


def main(argv=None):
    args = parser().parse_args(argv)

    s1 = Spectrum.from_lighttouch(args.spectrum1)
    s2 = Spectrum.from_lighttouch(args.spectrum2)

    delta_x, delta_y = analyze_color_error(s1, s2)
    
    # plt.plot(delta_x.wavelength, delta_x.values, label="dx")
    # plt.plot(delta_y.wavelength, delta_y.values, label="dy")
    # plt.xlabel("Wavelength [nm]")
    # plt.ylabel("Chromaticity error contribution / nm")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(args.output, dpi=150)
    # plt.show()

    fig, axes = plt.subplots(
        4, 1,
        figsize=(9, 10),
        sharex=True,
    )

    wl = delta_x.wavelength

    # 1. Normalized spectral_color_error

    s1n = (
        s1.interpolate(startwl=360, stopwl=830, step=1)
        .normalize(by="photometric", value=1)
    )

    s2n = (
        s2.interpolate(startwl=360, stopwl=830, step=1)
        .normalize(by="photometric", value=1)
    )

    axes[0].plot(wl, s1n.values, label="Spectrum 1")
    axes[0].plot(wl, s2n.values, label="Spectrum 2")
    axes[0].set_ylabel("Normalized SPD")
    axes[0].legend()

    # 2. Spectral residual
    delta_s = s2n - s1n
    axes[1].plot(wl, delta_s.values)
    axes[1].axhline(0, linewidth=0.5)
    axes[1].set_ylabel("dS")
    
    # 3. Local chromaticity-error contribution
    axes[2].plot(wl, delta_x.values, label="dx")
    axes[2].plot(wl, delta_y.values, label="dy")
    axes[2].axhline(0, linewidth=0.5)
    axes[2].set_ylabel("Contribution / nm")
    axes[2].legend()

    # 4. Cumulative chromaticity error
    cum_dx = cumulative_trapezoid(delta_x.values, wl, initial=0)
    cum_dy = cumulative_trapezoid(delta_y.values, wl, initial=0)

    axes[3].plot(wl, cum_dx, label="cumulative_dx")
    axes[3].plot(wl, cum_dy, label="cumulative_dy")
    axes[3].axhline(0, linewidth=0.5)
    axes[3].set_ylabel("Cumulative error")
    axes[3].set_xlabel("Wavelength [nm]")
    axes[3].legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    
if __name__ == "__main__":
    main()
    
