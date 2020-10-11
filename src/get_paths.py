"""
Created on 20 Dec 2019 by Philip.P
Functions to return paths of files/folders
Aim to move all data to the mongodb cloud to gather data etc
"""

import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_dataload_path(path):
    """Get the database/data path"""
    return os.path.join(_ROOT, 'dataload', path)


def get_data_path(path):
    """Get data path from data folder in src"""
    return os.path.join(_ROOT, "data", path)
