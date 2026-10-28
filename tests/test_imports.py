"""Import boundaries for the public package API."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def run_python(source):
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_and_discovery_do_not_load_scientific_stack():
    run_python('''
        import sys
        import gsmath
        assert set(gsmath.__all__) <= set(dir(gsmath))
        for name in ('numpy', 'pint', 'scipy', 'matplotlib', 'pandas',
                     'gsmath.analysis', 'gsmath.reference', 'gsmath.io'):
            assert name not in sys.modules, name
        assert not hasattr(gsmath, 'missing_public_attribute')
    ''')


def test_spectrum_import_does_not_load_optional_features():
    run_python('''
        import sys
        from gsmath import Spectrum
        from gsmath.spectrum import Spectrum as implementation
        assert Spectrum is implementation
        spectrum = Spectrum([400, 500, 600], [1., 2., 1.])
        assert len(spectrum.values) == 3
        for name in ('matplotlib', 'pandas', 'scipy.optimize',
                     'gsmath.analysis', 'gsmath.reference'):
            assert name not in sys.modules, name
    ''')


def test_dataframe_dependency_is_loaded_only_on_use():
    run_python('''
        import sys
        from gsmath import to_dict, to_dataframe
        assert to_dict({'signal': 2.0}) == {'signal': 2.0}
        assert 'pandas' not in sys.modules
        assert to_dataframe({'signal': 2.0}).iloc[0]['signal'] == 2.0
        assert 'pandas' in sys.modules
    ''')


def test_public_exports_and_star_imports_resolve_and_cache():
    run_python('''
        import gsmath
        import gsmath.analysis as analysis
        for package in (gsmath, analysis):
            for name in package.__all__:
                value = getattr(package, name)
                assert value is getattr(package, name), name
                assert name in vars(package), name
            namespace = {}
            exec(f'from {package.__name__} import *', namespace)
            assert set(package.__all__) <= namespace.keys()
    ''')
