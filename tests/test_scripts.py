from pathlib import Path

import pytest

from gsmath.scripts.spectral_color_error import parser as color_error_parser
from gsmath.scripts.straylight import parser as straylight_parser


def test_color_error_cli_defaults_to_artifacts():
    args = color_error_parser().parse_args(["measured.csv", "reference.csv"])

    assert args.spectrum1 == "measured.csv"
    assert args.spectrum2 == "reference.csv"
    assert args.output == Path(
        "artifacts/spectral_color_error/spectral_color_error.png"
    )


def test_color_error_cli_accepts_platform_native_output_path(tmp_path):
    output = tmp_path / "plots" / "error.png"
    args = color_error_parser().parse_args(
        ["measured.csv", "reference.csv", "--output", str(output)]
    )

    assert args.output == output


def test_color_error_help_describes_inputs_and_shows_defaults():
    help_text = color_error_parser().format_help()

    assert "reference spectrum in LightTouch CSV format" in help_text
    assert "comparison spectrum in LightTouch CSV format" in help_text
    assert "(default:" in help_text
    assert "artifacts/spectral_color_error/spectral_color_error.png" in help_text


def test_straylight_cli_defaults_to_artifacts():
    args = straylight_parser().parse_args(["run", "scans"])

    assert args.directory == Path("scans")
    assert args.output == Path("artifacts/straylight")


def test_straylight_cli_accepts_platform_native_paths(tmp_path):
    scans = tmp_path / "input scans"
    output = tmp_path / "analysis output"
    args = straylight_parser().parse_args(
        ["generate", str(scans), "--output", str(output)]
    )

    assert args.directory == scans
    assert args.output == output


@pytest.mark.parametrize("command", ["run", "generate"])
def test_straylight_generation_help_describes_options_and_defaults(
    command,
    capsys,
):
    with pytest.raises(SystemExit, match="0"):
        straylight_parser().parse_args([command, "--help"])

    help_text = capsys.readouterr().out
    assert "Directory containing the wavelength-scan CSV files" in help_text
    assert "--suppression FACTOR" in help_text
    assert "(default: StrayLight_*.csv)" in help_text
    assert "(default: 1.0)" in help_text
    assert "(default: artifacts/straylight)" in help_text


def test_straylight_evaluate_help_describes_options_and_defaults(capsys):
    with pytest.raises(SystemExit, match="0"):
        straylight_parser().parse_args(["evaluate", "--help"])

    help_text = capsys.readouterr().out
    assert "native .gsmath or legacy .bin correction" in help_text
    assert "--max-center-error NM" in help_text
    assert "(default: 2.5)" in help_text
    assert "(default: 5)" in help_text
