# Created on 20 Sep 2020

import unittest

import numpy as np

from securityAnalysis.utils_finance import calculate_relative_return

np.random.seed(1)  # set the random seed so the unit tests use synthetic data


class TestFinanceUtils(unittest.TestCase):
    def test_calculate_relative_returns(self):
        np.testing.assert_array_almost_equal(
            calculate_relative_return(np.random.random_sample(5)),
            np.array([1.72730572e+00, 1.58782352e-04, 2.64334912e+03, 4.85412106e-01])
        )


if __name__ == '__main__':
    unittest.main()
