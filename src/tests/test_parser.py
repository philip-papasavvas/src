"""
Created on 14 April 2021
Unit tests for parser.py. First test the individual functions,
and then move onto testing the Parser
"""
import unittest

import numpy as np
import pandas as pd

from parser import get_columns, rename_columns, name_columns

np.random.seed(1)


class TestParser(unittest.TestCase):

    @classmethod
    def setUp(cls) -> None:
        sample_data = {
            'a': range(10),
            'b': [x ** 2 for x in range(10)],
            'c': np.random.randint(low=0, high=100, size=10)
        }
        cls.sample_df = pd.DataFrame(sample_data)

    def test_get_columns(self):
        self.assertListEqual(
            get_columns(dataframe=self.sample_df, columns=['a', 'b']).columns.to_list(),
            ['a', 'b']
        )

    def test_rename_columns(self):
        self.assertListEqual(
            rename_columns(dataframe=self.sample_df, columns={'a': 'zero_to_four',
                                                              'b': 'squared',
                                                              'c': 'randoms'}).columns.to_list(),
            ['zero_to_four', 'squared', 'randoms']

        )

    def test_name_columns(self):
        self.assertListEqual(
            name_columns(
                dataframe=pd.DataFrame({1: [1, 2, 4]}),
                columns=['label']
            ).columns.to_list(),
            ['label']
        )


if __name__ == '__main__':
    unittest.main()
