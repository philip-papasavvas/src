"""
Created on: 20 Dec 2019
Created by: Philip.P

Module to unit-test Utils functions
"""
# built in imports
import unittest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing

# local imports
from utils import Utils, Date, List, Dict, SecuritiesAnalysis, BloombergData


class TestUtils(unittest.TestCase):

    # utilise pandas.testing module to check if dataframes are the same
    def assertDataframeEqual(self, a, b, msg):
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

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


    def test_Dict_funcs(self):
        self.assertEqual(Dict.flatten_dict(d={'a':'first level', 'b':{'more detail':{'third level'}, 'second level':[0]}}), \
                         {'a': 'first level', 'b.more detail': {'third level'}, 'b.second level': [0]}, \
                         "Dict should've been flatten to have two sub keys on b level: more detail, second level")

        self.assertEqual(Dict.return_keys(dict={'a':1, 'b':2, 'c':3}), \
                         ['a', 'b', 'c'], \
                         "Should've returned ['a', 'b', 'c']")

        self.assertEqual(Dict.return_values(dict={'a':1, 'b':2, 'c':3}), [1,2,3], \
                         "Should've returned [1,2,3]")


    def test_List_funcs(self):
        self.assertEqual(List.flatten_list([1, 2, 3, [4, 5]]), [1, 2, 3, 4, 5], \
                         "Resulting list should be [1,2,3,4,5]")

        self.assertEqual(List.has_duplicates(lst=[1,2,3,4,4]), True, "Value 4 is repeated")

        self.assertEqual(List.has_duplicates(lst=[1, 2, 3, 4, 5]), False, "No repeated values")

        self.assertEqual(List.comma_sep(lst=['hello','test','list']), 'hello,test,list', \
                         "Should've returned 'hello,test,list' ")

        self.assertEqual(List.all_unique(lst=[1,2,3,4]), True, "All elements are unique")

        self.assertEqual(List.chunk(lst=[1,2,3,4,5,6], chunk_size=2), \
                         [[1, 2], [3, 4], [5, 6]],
                         "Resulting list should be [[1, 2], [3, 4], [5, 6]]")

        self.assertEqual(List.count_occurences(lst=[1,2,3,4,2,2,2,2], value=2), 5, \
                         "THe number 2 appears 5 times")

        self.assertEqual(List.flatten(lst=[1,2,[3,4,5,[6,7]]]), [1, 2, 3, 4, 5, 6, 7], \
                         "Flattened list should be [1, 2, 3, 4, 5, 6, 7]")


    def test_Securities_funcs(self):
        sample_data = pd.DataFrame(data={'INDU Index': {'01/04/2011': 12376.72,
                                      '04/04/2011': 12400.03,
                                      '05/04/2011': 12393.9,
                                      '06/04/2011': 12426.75,
                                      '07/04/2011': 12409.49,
                                      '08/04/2011': 12380.05,
                                      '11/04/2011': 12381.11,
                                      '12/04/2011': 12263.58,
                                      '13/04/2011': 12270.99,
                                      '14/04/2011': 12285.15,
                                      '15/04/2011': 12341.83,
                                      '18/04/2011': 12201.59,
                                      '19/04/2011': 12266.75,
                                      '20/04/2011': 12453.54,
                                      '21/04/2011': 12505.99,
                                      '22/04/2011': 12505.99,
                                      '25/04/2011': 12479.88,
                                      '26/04/2011': 12595.37,
                                      '27/04/2011': 12690.96,
                                      '28/04/2011': 12763.31,
                                      '29/04/2011': 12810.54,
                                      '02/05/2011': 12807.36,
                                      '03/05/2011': 12807.51,
                                      '04/05/2011': 12723.58,
                                      '05/05/2011': 12584.17,
                                      '06/05/2011': 12638.74,
                                      '09/05/2011': 12684.68,
                                      '10/05/2011': 12760.36,
                                      '11/05/2011': 12630.03,
                                      '12/05/2011': 12695.92,
                                      '13/05/2011': 12595.75,
                                      '16/05/2011': 12548.37,
                                      '17/05/2011': 12479.58,
                                      '18/05/2011': 12560.18,
                                      '19/05/2011': 12605.32,
                                      '20/05/2011': 12512.04,
                                      '23/05/2011': 12381.26,
                                      '24/05/2011': 12356.21,
                                      '25/05/2011': 12394.66,
                                      '26/05/2011': 12402.76,
                                      '27/05/2011': 12441.58,
                                      '30/05/2011': 12441.58,
                                      '31/05/2011': 12569.79,
                                      '01/06/2011': 12290.14,
                                      '02/06/2011': 12248.55,
                                      '03/06/2011': 12151.26,
                                      '06/06/2011': 12089.96,
                                      '07/06/2011': 12070.81,
                                      '08/06/2011': 12048.94,
                                      '09/06/2011': 12124.36},
                                         'MXWO Index': {'01/04/2011': 1341.45,
                                      '04/04/2011': 1345.62,
                                      '05/04/2011': 1343.64,
                                      '06/04/2011': 1348.86,
                                      '07/04/2011': 1345.2,
                                      '08/04/2011': 1351.43,
                                      '11/04/2011': 1348.65,
                                      '12/04/2011': 1333.85,
                                      '13/04/2011': 1338.75,
                                      '14/04/2011': 1336.54,
                                      '15/04/2011': 1338.04,
                                      '18/04/2011': 1316.91,
                                      '19/04/2011': 1324.45,
                                      '20/04/2011': 1350.22,
                                      '21/04/2011': 1360.95,
                                      '22/04/2011': 1361.03,
                                      '25/04/2011': 1357.5,
                                      '26/04/2011': 1366.69,
                                      '27/04/2011': 1371.38,
                                      '28/04/2011': 1384.94,
                                      '29/04/2011': 1388.62,
                                      '02/05/2011': 1391.86,
                                      '03/05/2011': 1382.99,
                                      '04/05/2011': 1371.36,
                                      '05/05/2011': 1355.45,
                                      '06/05/2011': 1359.27,
                                      '09/05/2011': 1353.77,
                                      '10/05/2011': 1366.36,
                                      '11/05/2011': 1357.35,
                                      '12/05/2011': 1351.14,
                                      '13/05/2011': 1343.17,
                                      '16/05/2011': 1336.65,
                                      '17/05/2011': 1329.33,
                                      '18/05/2011': 1341.98,
                                      '19/05/2011': 1345.73,
                                      '20/05/2011': 1336.65,
                                      '23/05/2011': 1313.69,
                                      '24/05/2011': 1315.76,
                                      '25/05/2011': 1320.26,
                                      '26/05/2011': 1326.96,
                                      '27/05/2011': 1338.47,
                                      '30/05/2011': 1338.29,
                                      '31/05/2011': 1354.61,
                                      '01/06/2011': 1335.59,
                                      '02/06/2011': 1325.36,
                                      '03/06/2011': 1320.11,
                                      '06/06/2011': 1309.78,
                                      '07/06/2011': 1311.9,
                                      '08/06/2011': 1301.84,
                                      '09/06/2011': 1307.93}
                                         }
                                   )

        self.assertEqual(first=SecuritiesAnalysis.log_daily_returns(data=sample_data.iloc[:10, :]), \
                         second=pd.DataFrame({'INDU Index':{'01/06/2011':-0.007019973807377511,'02/05/2011':0.04122269078824914,'02/06/2011':-0.04461244303858436,'03/05/2011':0.04462415498559125,'03/06/2011':-0.05259884951001759,'04/04/2011':0.020266023849499604,'04/05/2011':0.02575807289130516,'05/04/2011':-0.026252548767045525,'05/05/2011':0.015235258795620155},'MXWO Index':{'01/06/2011':-0.004377976690467911,'02/05/2011':0.041267840353913954,'02/06/2011':-0.04895686149346368,'03/05/2011':0.04256370135034526,'03/06/2011':-0.04653275553890701,'04/04/2011':0.019139806924532543,'04/05/2011':0.018948074784320035,'05/04/2011':-0.020420599090466673,'05/05/2011':0.008751153440626602}}), \
                         msg="Dataframes not equal")


if __name__ == "__main__":
    unittest.main()

