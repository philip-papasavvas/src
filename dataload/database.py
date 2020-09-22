"""
Created: 13 Oct 2019
mongoDB Atlas: initialise, write, append, read to library
"""
import json
from typing import Union

import pandas as pd
from arctic import Arctic, VERSION_STORE, CHUNK_STORE, TICK_STORE
from arctic.chunkstore.chunkstore import ChunkStore
from arctic.store.version_store import VersionStore
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError


# connect to database
def db_connect(mongo_config: dict,
               is_arctic: bool = True,
               lib_name: str = None) -> Union[object, Arctic, VersionStore, ChunkStore, Collection]:
    """
    Connect to the MongoDB Atlas instance using a config containing user, password, url_cluster
    parameters

    Args:
        mongo_config: Dict-like object with the keys ["mongo_user", "mongo_pwd", "url_cluster"]
        is_arctic: If searching the Arctic database on MongoDB (for time-series data)
        lib_name: Name of library within database, default to None

    Returns:
        object:
            lib_store: arctic store
            lib_collection: pymongo collection
            db_connection: arctic.arctic.Arctic, pymongo.database.Database
    """
    user = mongo_config["mongo_user"]
    password = mongo_config["mongo_pwd"]
    mongo_url = mongo_config["url_cluster"]

    host_url = "".join(["mongodb+srv://", user, ":", password, "@", mongo_url])
    client = MongoClient(host=host_url)

    # check the connection to the database
    try:
        client.PORT
        # print(f"Listing existing database names: {client.list_database_names()}")
    except ServerSelectionTimeoutError:
        raise ServerSelectionTimeoutError('MongoDB is not hosted.')

    if is_arctic:
        db_connection = Arctic(client)
        library_list = db_connection.list_libraries()
        if lib_name is not None:
            assert lib_name in library_list, \
                f"\n Library: '{lib_name}' does not exist in Arctic database"
            lib_store = db_connection[lib_name]
            return lib_store
        else:
            print(f"List of libraries in 'Arctic' database: \n {library_list}")
            return db_connection
    # non-arctic collection
    else:
        print("Have not yet configured non-Arctic database on MongoDB Atlas")
        # Check if I want to add a non-time series dataset


def db_keys_and_symbols(is_arctic: bool,
                        library_name: str,
                        mongo_config: dict) -> list:
    """Returns list of keys in collection (arctic/non-arctic database).

    Args:
        is_arctic: True for arctic symbols, False for non_arctic keys
        library_name: Library name
        mongo_config: Dict-like object with the keys ["mongo_user", "mongo_pwd", "url_cluster"]

    Returns:
        keys / symbols: List of symbols or keys depending on arctic/non-arctic database
    """

    # Non arctic keys
    if is_arctic is False:
        cl = db_connect(is_arctic=False,
                        lib_name=library_name,
                        mongo_config=mongo_config)
        try:
            docs = cl.find_one()
            keys = []
            for x in docs:
                keys.append(x)
            return keys

        except AttributeError:
            print('\nIncorrect data type, mongodb collection should be the only acceptable type.')

    # Arctic symbols
    else:
        lib = db_connect(is_arctic=True, lib_name=library_name, mongo_config=mongo_config)
        symbols = lib.list_symbols()
        return symbols


def db_arctic_library(mongo_config: dict,
                      library: str = None) -> Union[VersionStore, ChunkStore, list]:
    """Function to return Arctic store/list of library names in Arctic database

    Args:
        mongo_config: Dict-like object with the keys ["mongo_user", "mongo_pwd", "url_cluster"]
        library: If None, returns list of available library names

    Returns
        arctic_store
    """
    store = db_connect(mongo_config=mongo_config, is_arctic=True)

    if library is None:
        return store.list_libraries()
    else:
        return store


# -----------------------
# ARCTIC HELPER FUNCTIONS
# -----------------------
def db_arctic_read(mongo_config: dict,
                   library: str,
                   symbol: str = None) -> pd.DataFrame:
    """
    Returning the data frame stored in MongoDB Arctic database.

    Args:
        mongo_config: Dict-like object with the keys ["mongo_user", "mongo_pwd", "url_cluster"]
        library: Name of library in individual mongo cluster
        symbol: Named symbol present in library

    Returns:
        pd.DataFrame

    Note
    -----
    To check on the available library names
    >>> store = db_connect(mongo_config=mongo_config, is_arctic=True)
    >>> store.list_libraries()
    """

    lib = db_connect(mongo_config=mongo_config, is_arctic=True, lib_name=library)

    if symbol is not None:
        assert lib.has_symbol(symbol), f"{symbol} not found in library: {library}"
        out = lib.read(symbol)
    else:
        raise KeyError(f"No symbol chosen from the library, the following symbols can be"
                       f" read from the {library}: {lib.list_symbols()}")

    return out.sort_index()


def db_arctic_initialise(mongo_config: dict,
                         library_name: str,
                         library_type: str) -> None:
    """
    Initialise new arctic library.

    Args:
        mongo_config: Dict-like object with the keys ["mongo_user", "mongo_pwd", "url_cluster"]
        library_name: Name of the new library to create.
        library_type: Acceptable types ["VERSION_STORE", "CHUNK_STORE", "TICK_STORE"]

    Returns:
        None
    """
    arctic_stores = [VERSION_STORE, CHUNK_STORE, TICK_STORE]

    if library_type not in arctic_stores:
        raise KeyError(f"Library store must be an arctic store type: {VERSION_STORE}, "
                       f"{CHUNK_STORE}, {TICK_STORE}")

    lib = db_connect(mongo_config=mongo_config, is_arctic=True, lib_name=None)

    lib.initialize_library(library=library_name, lib_type=library_type)

    # This is important because arctic will not show the existing
    # libraries upon creation of a new library.
    Arctic.reload_cache(lib)


def db_arctic_write(mongo_config: dict,
                    df: pd.DataFrame,
                    symbol: str,
                    library_name: str = None) -> None:
    """
    Writing to existing arctic library.

    Args:
        mongo_config: Dict-like object with the keys ["mongo_user", "mongo_pwd", "url_cluster"]
        df: Dataframe to write into library
        symbol: Name of symbol
        library_name: Name of arctic library to write on

    Returns:
        None
    """
    assert library_name is not None, "library_name must be passed in to specify library to write."

    lib = db_connect(mongo_config=mongo_config, is_arctic=True, lib_name=library_name)
    lib.write(symbol, df)


def db_arctic_append(mongo_config: dict,
                     df: pd.DataFrame,
                     symbol: str,
                     library_name: str = None) -> None:
    """
    Appending existing arctic library.

    Args:
        mongo_config: Dict-like object with the keys ["mongo_user", "mongo_pwd", "url_cluster"]
        df: pd.DataFrame
        symbol: Name of symbol
        library_name: Name of arctic library to append on

    Returns:
        None
    """

    assert library_name is not None, "lib_name must be passed in to specify library to append."
    lib = db_connect(mongo_config=mongo_config, is_arctic=True, lib_name=library_name)
    lib.append(symbol, df, upsert=True)


if __name__ == '__main__':

    # load config
    mongo_path = 'PATH-TO-PRIVATE-MONGO-DB-HERE'
    mongo_cfg = json.load(open(mongo_path, 'r'))

    # config structure
    mongo_config_example = {
        'mongo_user': 'USERNAME',
        'mongo_pwd': 'MONGO-PASSWORD-HERE',
        'url_cluster': 'MONGODB-ATLAS-CLUSTER-HERE'
    }
