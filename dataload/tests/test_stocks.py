"""
Created on 7 Aug 2020
Unit tests for database functions
"""

import unittest

from dataload.stocks import return_stock_data


class MyTestCase(unittest.TestCase):
    def test_dataload(self):
        # single stock in a list
        self.assertIsNotNone(return_stock_data(stocks=['TSLA']),
                             msg='This should not be none')

        # multiple stocks
        self.assertIsNotNone(return_stock_data(stocks=['TSLA', 'AMZN']),
                             msg='This should not be none')

        # single stock not in a list
        self.assertIsNotNone(return_stock_data(stocks='AMZN'),
                             msg='This should not be none')

        # return single stock data (more than just price)
        self.assertIsNotNone(return_stock_data(stocks=['GOOGL'], return_price=False),
                             msg='This should not be none')


if __name__ == '__main__':
    unittest.main()
