import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src" / "gsmath"

ABSOLUTE_WORKSTATION_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]|/(?:home|Users)/[^/]+/)"
)


def source_files():
    return SOURCE.rglob("*.py")


def test_source_has_no_workstation_specific_paths():
    offenders = []
    for path in source_files():
        if ABSOLUTE_WORKSTATION_PATH.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_installed_scripts_do_not_execute_on_import():
    offenders = []
    scripts = SOURCE / "scripts"

    for path in scripts.glob("*.py"):
        if path.name == "__init__.py":
            continue
        source = path.read_text(encoding="utf-8")
        if 'if __name__ == "__main__":' not in source:
            offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []
