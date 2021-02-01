"""
Created 17 June 2020
Generic utility methods for handling dates
"""
import datetime

import numpy as np


def np_dt_to_str(d: np.datetime64) -> str:
    """Convert from np.datetime64 to str without hyphens"""
    return d.astype(str).replace("-", "")


def excel_date_to_np(xl_date):
    """Excel date serial (as int) to numpy datetime"""
    return np.array(['1899-12-30'], dtype='datetime64[D]') + xl_date


def date_to_excel(pdate):
    """converts datetime to Excel date serial"""
    delta = pdate.to_datetime() - datetime.datetime(1899, 12, 30)
    return (delta.days.astype(float) + delta.seconds.astype(float) / 86400).astype(int)


def time_delta_to_days(td):
    """Returns the day difference of a pandas series of timedelta64[ns]"""
    return (td.values / np.timedelta64(1, 'D')).astype(int)


def datetime_to_str(input_date: datetime.datetime):
    """Method to extract date in YYYYMMDD from datetime object"""
    return datetime.datetime.strftime(input_date, format="%Y%m%d")


if __name__ == "__main__":
    pass
