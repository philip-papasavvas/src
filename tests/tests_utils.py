"""
Created on: 20 Dec 2019
Created by: Philip.P

Module to unit-test utils functions
"""
# built in imports
import unittest
import pandas as pd
import numpy as np
import datetime
import json
import os
import pandas.testing as pd_testing

# local imports
import utils


class Testutils(unittest.TestCase):

    # utilise pandas.testing module to check if dataframes are the same
    def assertDataframeEqual(self, a, b, msg):
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    def test_utils_funcs(self):
        # self.assertIs(type(utils.to_array([2])),generator, "Wrong type returned")

        self.assertEqual(utils.average(2,2,5),3, "Average has not calculated correctly")
        self.assertEqual(utils.average(*[1,2,3]),2, "Average has not calculated correctly")

        self.assertEqual(utils.difference([3,10,9],[3,4,10]), {9}, "Difference function not working as expected")

    def test_utils_date_funcs(self):
        self.assertEqual(utils.date_shift(date=datetime.datetime(year=2010, month=10, day=10), m=2),
                         datetime.datetime(2010, 8, 10, 0, 0),
                         "Date should be 10 Aug 2010")

        self.assertEqual(utils.date_shift(date=datetime.datetime(year=2010, month=10, day=10), y=1, reverse=False),
                         datetime.datetime(2011, 10, 10, 0, 0),
                         "Date should be 10 Oct 2011")

        self.assertEqual(utils.return_date_tenor_shift(date=np.datetime64("2019-10-12"), pillar="1M", reverse=True), \
                         np.datetime64('2019-09-12'), \
                         "Date should be 12 Sep 2019")

        self.assertEqual(utils.return_date_tenor_shift(date=np.datetime64("2019-10-12"), pillar="1Y", reverse=False), \
                         np.datetime64('2020-10-12'), \
                         "Date should be 12 Oct 2020")

        dates_sample = pd.DataFrame(data=pd.date_range(start="2010-01-01", end="2010-01-05"))
        self.assertEqual(utils.return_date_diff(dataframe=dates_sample, date=np.datetime64("2010-01-01"), time_diff="4W"),
                         np.datetime64("2009-12-02"),
                         "Should've returned 2009-12-02")

        self.assertEqual(list(utils.x2pdate(xldate=43100)),
                         [np.datetime64('2017-12-31')],
                         "Should be 2017-12-31 in a list")


    def test_utils_dict_funcs(self):
        self.assertEqual(utils.flatten_dict(d=
                                            {'a':'first level', 'b': {
                                                'more detail':{
                                                    'third level'
                                                },
                                                'second level':[0]}
                                             }
                                            ), \
                         {'a': 'first level', 'b.more detail': {'third level'}, 'b.second level': [0]}, \
                         "Dict should've been flatten to have two sub keys on b level: more detail, second level")

        self.assertEqual(utils.return_dict_keys(dict={'a': 1, 'b': 2, 'c': 3}), \
                         ['a', 'b', 'c'], \
                         "Should've returned ['a', 'b', 'c']")

        self.assertEqual(utils.return_dict_values(dict={'a': 1, 'b': 2, 'c': 3}), [1,2,3], \
                         "Should've returned [1,2,3]")


    def test_utils_list_funcs(self):
        self.assertEqual(utils.flatten_list([1, 2, 3, [4, 5]]), [1, 2, 3, 4, 5], \
                         "Resulting list should be [1,2,3,4,5]")

        self.assertEqual(utils.has_duplicates(lst=[1,2,3,4,4]), True, "Value 4 is repeated")

        self.assertEqual(utils.has_duplicates(lst=[1, 2, 3, 4, 5]), False, "No repeated values")

        self.assertEqual(utils.comma_sep(lst=['hello','test','list']), 'hello,test,list', \
                         "Should've returned 'hello,test,list' ")

        self.assertEqual(utils.all_unique(lst=[1,2,3,4]), True, "All elements are unique")

        self.assertEqual(utils.chunk(lst=[1,2,3,4,5,6], chunk_size=2), \
                         [[1, 2], [3, 4], [5, 6]],
                         "Resulting list should be [[1, 2], [3, 4], [5, 6]]")

        self.assertEqual(utils.count_occurences(lst=[1,2,3,4,2,2,2,2], value=2), 5, \
                         "THe number 2 appears 5 times")

        self.assertEqual(utils.flatten(lst=[1,2,[3,4,5,[6,7]]]), [1, 2, 3, 4, 5, 6, 7], \
                         "Flattened list should be [1, 2, 3, 4, 5, 6, 7]")


    def test_utils_securities_funcs(self):
        with open(os.path.join(os.getcwd(), "data", "market_data.json"), "r") as mkt_data:
            k = json.load(mkt_data)

        sample_data = pd.DataFrame(data=k)

        self.assertEqual(first=utils.log_daily_returns(data=sample_data.iloc[:3, :]), \
                         second=pd.DataFrame(data={'INDU Index': {'01/06/2011': -0.007019973807377511,
                                                                   '02/05/2011': 0.04122269078824914},
                                                    'MXWO Index': {'01/06/2011': -0.004377976690467911,
                                                                   '02/05/2011': 0.041267840353913954}
                                                   }
                                             ), \
                         msg="Dataframes not equal")


if __name__ == "__main__":
    unittest.main()

