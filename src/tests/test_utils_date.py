# Created on 24 Dec 2019

import datetime
import unittest

import numpy as np
import pandas as pd

from src.utils_date import np_dt_to_str, excel_date_to_np, datetime_to_str


class TestDateFuncs(unittest.TestCase):
    def setUp(self) -> None:
        dates = np.arange("2010-01-01", "2010-01-05", dtype='datetime64[D]')
        self.dates = dates
        self.dataframe_dates = pd.DataFrame(
            {'date': dates,
             'date_as_str': dates.astype(str)}
        )

    def test_date_to_str(self):
        np.testing.assert_equal(np_dt_to_str(np.datetime64("2020-01-01")),
                                "20200101")

    def test_x2pdate(self):
        self.assertEqual(list(excel_date_to_np(xl_date=43100)),
                         [np.datetime64('2017-12-31')],
                         "Should be 2017-12-31 in a list")

    def test_datetime_to_str(self):
        self.assertEqual(
            datetime_to_str(input_date=datetime.datetime(2020, 1, 1)),
            "20200101"
        )


if __name__ == '__main__':
    unittest.main()
