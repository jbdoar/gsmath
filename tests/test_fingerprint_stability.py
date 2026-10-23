"""Calibration fingerprints of established step types must not change.

A saved calibration stores its fingerprint, and reading it back refuses the
file if the recomputed fingerprint differs -- so changing how an existing
step type fingerprints would lock users out of their saved calibrations.
These values were computed before the step refactoring of 2026-10-20.
"""

import numpy as np
import pytest

from gsmath import (
    AdaptiveSmooth, Calibration, Despike, PerExposure, Responsivity, Spectrum, StrayLight,
    SubtractDark,
)
from gsmath.straylight import StrayLightCorrection
from gsmath.units import Q_

WL = np.array([400.0, 401.0, 402.0, 403.0])

GOLDEN = {
    "dark+exposure": ([SubtractDark(), PerExposure()], "9ad8bfded22512c1"),
    "optional dark": ([SubtractDark(optional=True)], "dfadfcd619376adb"),
    "responsivity float": ([Responsivity(2.0)], "a41fa050c826fa57"),
    "responsivity int": ([Responsivity(2)], "a41fa050c826fa57"),
    "responsivity quantity": ([Responsivity(Q_(3, "1/sr"))], "fa3fc950ef09eb81"),
    "responsivity array": ([Responsivity(np.array([1.0, 2.0, 3.0]))], "76b0fd444dbed0a5"),
    "responsivity spectrum": (
        [Responsivity(Spectrum(WL, [2.0, 2.0, 4.0, 4.0],
                               value_unit="uW/cm^2/sr/nm * ms / count"))],
        "879db2a39bc665a5"),
    "stray light": ([StrayLight(StrayLightCorrection(WL, np.eye(4) * 0.5))], "8aa799fcbd760893"),
    "despike default": ([Despike()], "6245de134f10f34c"),
    "despike custom": ([Despike(threshold=8, window=21, relative=0.05)], "2c41bce05f25696c"),
    "adaptive smooth": ([AdaptiveSmooth()], "3be00cdb5a52177c"),
}


@pytest.mark.parametrize("name", GOLDEN)
def test_established_fingerprints_are_unchanged(name):
    steps, fingerprint = GOLDEN[name]
    assert Calibration("golden", steps).fingerprint == fingerprint


def test_new_steps_fingerprint_their_arguments_by_default():
    from gsmath.calibration import Step, fingerprint_value

    class Scale(Step):
        def __init__(self, factor, label="x", *, optional=False):
            super().__init__(optional=optional)
            self.factor, self.label = factor, label

        def apply(self, data, measurement):
            return data * self.factor

        def arguments(self):
            return {"factor": self.factor, "label": self.label}

    def fingerprint(steps):
        return Calibration("c", steps).fingerprint

    assert fingerprint([Scale(2.0)]) == fingerprint([Scale(2)])
    assert fingerprint([Scale(2.0)]) != fingerprint([Scale(3.0)])
    assert fingerprint([Scale(2.0)]) != fingerprint([Scale(2.0, "y")])
    assert fingerprint_value(None) is None and fingerprint_value(True) is True
    assert fingerprint_value("bilinear") == "bilinear"
    assert fingerprint_value(["R", 2]) == ["R", 2.0]


def test_public_unit_and_step_helpers():
    from gsmath import WavelengthCalibration, units

    assert units.divide_units("count", "s") == units.parse("count/s")
    assert units.divide_units(None, "s") == units.parse("1/s")
    assert WavelengthCalibration.to_step is WavelengthCalibration.step
