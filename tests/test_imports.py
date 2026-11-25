"""Import boundaries and public API compatibility in fresh interpreters."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def run_python(source):
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    # Use the same source checkout as this test run, including staged builds.
    import gsmath
    environment["PYTHONPATH"] = str(Path(gsmath.__file__).resolve().parents[1])
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
        try:
            gsmath.missing_public_attribute
        except AttributeError as error:
            assert 'gsmath' in str(error)
            assert 'missing_public_attribute' in str(error)
        else:
            raise AssertionError('missing attribute did not raise')
    ''')


def test_gui_core_imports_do_not_load_unused_features():
    run_python('''
        import sys
        from gsmath import (Spectrum, Image, Measurement, Calibration,
                            SubtractDark, PixelMap, Rectangle, read)
        from gsmath.spectrum import Spectrum as implementation
        assert Spectrum is implementation
        spectrum = Spectrum([400, 500, 600], [1., 2., 1.])
        assert len(spectrum.values) == 3
        for name in ('matplotlib', 'pandas', 'scipy.optimize', 'scipy.signal',
                     'scipy.ndimage', 'gsmath.analysis', 'gsmath.reference',
                     'gsmath.linefit', 'gsmath.straylight'):
            assert name not in sys.modules, name
    ''')


def test_statistics_loads_only_the_requested_analysis():
    run_python('''
        import sys
        import gsmath
        import gsmath.analysis as analysis
        assert set(analysis.__all__) <= set(dir(analysis))
        assert not hasattr(analysis, 'missing_public_attribute')
        from gsmath import StatisticsAnalysis, Spectrum
        from gsmath.analysis.statistics import StatisticsAnalysis as implementation
        assert StatisticsAnalysis is implementation is analysis.StatisticsAnalysis
        assert Spectrum([400, 500, 600], [1., 2., 3.]).stats().mean == 2.
        for name in ('matplotlib', 'pandas', 'scipy.optimize', 'scipy.signal',
                     'gsmath.analysis.peaks', 'gsmath.analysis.color', 'gsmath.reference'):
            assert name not in sys.modules, name
    ''')


def test_dictionary_export_does_not_load_pandas():
    run_python('''
        import sys
        from gsmath import to_dict, print_result, to_dataframe, report
        assert to_dict({'signal': 2.0}) == {'signal': 2.0}
        print_result({'signal': 2.0})
        assert 'pandas' not in sys.modules
        assert to_dataframe({'signal': 2.0}).iloc[0]['signal'] == 2.0
        assert 'pandas' in sys.modules
        assert report({'signal': 2.0}).iloc[0]['signal'] == 2.0
    ''')


@pytest.mark.parametrize('first_import', [
    'from gsmath import Bracketed',
    'from gsmath.bracket import Bracketed',
    'import gsmath.bracket',
    'from gsmath.report import report',
    'import gsmath.report',
    'from gsmath.analysis.delta_e import delta_e_2000',
    'import gsmath.analysis.delta_e',
])
def test_functions_remain_callable_after_submodule_imports(first_import):
    run_python(first_import + '''
import gsmath
from gsmath import bracket, report, delta_e_2000
from gsmath.bracket import bracket as bracket_impl
from gsmath.report import report as report_impl
from gsmath.analysis.delta_e import delta_e_2000 as delta_impl
assert bracket is gsmath.bracket is bracket_impl
assert report is gsmath.report is report_impl
assert delta_e_2000 is gsmath.analysis.delta_e_2000 is delta_impl
assert bracket(2, 4, 2).effect == 1
assert callable(report)
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
            for name in package.__all__:
                assert namespace[name] is getattr(package, name), name
        # These existing top-level exports are not listed in __all__.
        from gsmath import (LineSpreadScan, StrayLightCalibrationResult,
                            StrayLightCorrection, StrayLightDistribution,
                            StrayLightEvaluation, stray_light_calibration)
        from gsmath.io.formats import read_any
        assert gsmath.read is read_any
    ''')
