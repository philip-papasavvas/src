"""
Created on 12 June 2020
All unit tests
"""

import os
import unittest
from time import time

_ROOT = os.path.abspath(os.path.dirname(__file__))


if __name__ == '__main__':

    # tests are stored here
    EXTRA_PATHS = [
        "securityAnalysis/tests",
        "tests"
    ]
    t = time()
    for path in EXTRA_PATHS:

        loader = unittest.TestLoader()
        test_path = os.path.join(_ROOT, path)
        suite = loader.discover(test_path, pattern="test_*.py")

        runner = unittest.TextTestRunner()
        runner.run(suite)

    print(f'All unit tests finished after {time()-t: 2.2f} secs')
