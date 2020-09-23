"""
Created on 7 Aug 2020
Unit tests for database functions
"""

import unittest

from src.dataload.stocks import return_stock_data


class MyTestCase(unittest.TestCase):
    def test_return_stock_data__single_stock_in_list(self):
        self.assertIsNotNone(return_stock_data(stocks=['TSLA']),
                             msg='This should not be none')

    def test_return_stock_data__single_stock_as_str(self):
        self.assertIsNotNone(return_stock_data(stocks='AMZN'),
                             msg='This should not be none')

    def test_return_stock_data__multiple_stocks(self):
        self.assertIsNotNone(return_stock_data(stocks=['TSLA', 'AMZN']),
                             msg='This should not be none')

    def test_return_stock_data_non_price(self):
        self.assertIsNotNone(return_stock_data(stocks=['GOOGL'], return_price=False),
                             msg='This should not be none')


if __name__ == '__main__':
    unittest.main()
