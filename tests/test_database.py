"""
Created on: 12 Aug 2020
Created by: Philip.P_adm

Test database functions for MongoDB (non-relational database)
"""
import json
import os
import unittest

# 3rd party import
from pymongo import MongoClient

# local import
from aialpha.dataload.database import (db_connect, db_keys_and_symbols)
from aialpha.dataload import database

MONGO_CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'aialpha', 'dataload', 'config', 'mongo_private.json'
)

# These tests require a live MongoDB Atlas connection (private credentials) and
# the optional ``arctic`` dependency, so they are skipped unless both are present.
HAS_CREDENTIALS = os.path.exists(MONGO_CFG_PATH)


@unittest.skipUnless(HAS_CREDENTIALS, "mongo_private.json credentials not available")
@unittest.skipUnless(database.ARCTIC_AVAILABLE, "optional 'arctic' dependency not installed")
class TestMongoDB(unittest.TestCase):
    def setUp(self) -> None:
        with open(MONGO_CFG_PATH, 'r') as f:
            self.mongo_config = json.load(f)
        username = self.mongo_config['mongo_user']
        pw = self.mongo_config['mongo_pwd']
        mongo_url = self.mongo_config['url_cluster']

        self.pymongo_connect = MongoClient(
            host="".join(["mongodb+srv://", username, ":", pw, "@", mongo_url]))

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
