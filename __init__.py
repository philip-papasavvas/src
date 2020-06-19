"""
Init created on creation of the projects repository
Functions to return paths of files/folders

æTODO: Aim to move all data to the mongodb cloud to gather data etc
"""
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_config_path(path):
    return os.path.join(_ROOT, 'config', path)


def get_db_path(path):
    return os.path.join(_ROOT, 'dataload', path)


def get_import_path(path):
    return os.path.join(_ROOT, 'import', path)


def get_path(path):
    return os.path.join(_ROOT, path)


def get_data_path(path):
    return os.path.join(_ROOT, "data", path)


def get_custom_path(path, directory_name):
    return os.path.join(_ROOT, directory_name, path)
