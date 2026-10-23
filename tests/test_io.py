from pathlib import Path

import numpy as np

from gsmath.io.raster import source


def test_source_metadata_keeps_filename_without_parent_directories():
    metadata = source(Path("private") / "session" / "frame.png", "png",
                      np.zeros((2, 2), dtype=np.uint16))

    assert metadata == {"path": "frame.png", "format": "png", "dtype": "uint16"}
