"""
Created on: 20 Dec 2019
Created by: Philip.P

Module to unit-test Utils functions
"""
# built in imports
import unittest
import pandas as pd
import numpy as np

# local imports
from utils import Utils, Date, ListMethods, DictMethods


class TestUtils(unittest.TestCase):

    def test_Utils_funcs(self):
        # self.assertIs(type(Utils.to_array([2])),generator, "Wrong type returned")

        self.assertEqual(Utils.average(2,2,5),3, "Average has not calculated correctly")
        self.assertEqual(Utils.average(*[1,2,3]),2, "Average has not calculated correctly")

        self.assertEqual(Utils.difference([3,10,9],[3,4,10]), {9}, "Difference function not working as expected")

    def test_Date_funcs(self):
        self.assertEqual(Date.datePlusTenorNew(date=np.datetime64("2019-10-12"), pillar="1M", reverse=True), \
                         np.datetime64('2019-09-12'), \
                         "Date should be 12 Sep 2019")

        self.assertEqual(Date.datePlusTenorNew(date=np.datetime64("2019-10-12"), pillar="1Y", reverse=False), \
                         np.datetime64('2020-10-12'), \
                         "Date should be 12 Oct 2020")

        dates_sample = pd.DataFrame(data=pd.date_range(start="2010-01-01", end="2010-01-05"))
        self.assertEqual(Date.previousDate(dataframe=dates_sample, date=np.datetime64("2010-01-01"), timeDifference="4W"),
                         np.datetime64("2009-12-02"),
                         "Should've returned 2009-12-02")

        self.assertEqual(list(Date.x2pdate(xldate=43100)),
                        [np.datetime64('2017-12-31')],
                         "Should be 2017-12-31 in a list")


    def test_ListMethods_funcs(self):
        self.assertEqual(ListMethods.flatten_list([1, 2, 3, [4, 5]]), \
                         [1, 2, 3, 4, 5], \
                         "Resulting list should be [1,2,3,4,5]")


if __name__ == "__main__":
    unittest.main()