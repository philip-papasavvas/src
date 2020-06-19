# Created on 12 June 2020. For collating all unit tests

import os
import unittest
from time import time

_ROOT = os.path.abspath(os.path.dirname(__file__))


if __name__ == '__main__':

    # tests are stored here
    extra_paths = [
        "securityAnalysis/tests",
        "tests"
    ]
    t = time()
    for p in extra_paths:

        loader = unittest.TestLoader()
        test_path = os.path.join(_ROOT, p)
        suite = loader.discover(test_path, pattern="test_*.py")


        runner = unittest.TextTestRunner()
        runner.run(suite)

    print(f'All unit tests finished after {time()-t: 2.2f} secs')