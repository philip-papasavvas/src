"""
Created on: 12 Aug 2020
Created by: Philip.P_adm

Test database functions for MongoDB (non-relational database)
"""
import json
import unittest

# 3rd party import
from pymongo import MongoClient

# local import
from dataload.database import (db_connect, db_keys_and_symbols)


class TestMongoDB(unittest.TestCase):
    def setUp(self) -> None:
        mongo_cfg_path = '/Users/philip_p/python/src/dataload/config/mongo_private.json'
        self.mongo_config = json.load(open(mongo_cfg_path, 'r'))
        usernane = self.mongo_config['mongo_user']
        pw = self.mongo_config['mongo_pwd']
        mongo_url = self.mongo_config['url_cluster']

        self.pymongo_connect = MongoClient(
            host="".join(["mongodb+srv://", usernane, ":", pw, "@", mongo_url]))

    def test_db_connect_general(self):
        # check that there are databases on MongoDB connection
        self.assertIsNotNone(
            db_connect(mongo_config=self.mongo_config))

    def test_db_connect_arctic(self):
        # check that it works for arctic database
        self.assertIsNotNone(
            db_connect(mongo_config=self.mongo_config,
                       is_arctic=True))

    def test_db_keys_and_symbols(self):
        self.assertIsNotNone(
            db_keys_and_symbols(is_arctic=True,
                                library_name='security_data',
                                mongo_config=self.mongo_config))


if __name__ == '__main__':
    unittest.main()
