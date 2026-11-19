"""Start the test-map recorder in every Python process of a map build (scripts/test_map.py).

A map build puts this directory first on ``PYTHONPATH`` and names its output directory in
``SUPERGLM_TEST_MAP_OUT``; Python imports ``sitecustomize`` at startup, so the pytest workers and
every Python subprocess a test starts record what they run. Without the variable it does nothing.
"""

import os

if os.environ.get("SUPERGLM_TEST_MAP_OUT"):
    import test_map_recorder

    test_map_recorder.start()
