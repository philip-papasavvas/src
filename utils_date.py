"""
Created 17 June 2020
Taken from utils_generic.py
"""

import datetime as dt
from typing import Union, List

import numpy as np
import pandas as pd


def date_to_str(d: np.datetime64) -> str:
    """Convert from np.datetime64 to str without hyphens"""
    return d.astype(str).replace("-", "")


def char_to_date(s: Union[pd.DataFrame, pd.Series, np.ndarray, List[str], str]
                 ) -> Union[pd.DataFrame, pd.Series]:
    """
    Turning date from object to np.datetime64

    Args:
        s: Pass in DataFrame if multi column process is needed

    Returns:
        all columns containing "date" (case in-sensitive) will be amended

    Note
        This method can handle EITHER "/" or "-" date separators but not a combination of both.
        Users should check that there are no mixtures of separators if s is an array
    """

    def findFormat(s):
        global sep
        try:
            sep = '/'
            if pd.Series(s).str.contains('-').all():
                sep = '-'
            x = pd.Series(s).str.split('/|-', expand=True).values
            x = x.astype(int)
            month_pattern = '%m'
        except ValueError:
            month_pattern = '%b'

        year_col, month_col, date_col = None, None, None
        for i in range(x.shape[-1]):
            if x[:, i].dtype != object:
                if all(x[:, i].astype(int) > 1000):
                    year_col = i
                elif all(x[:, i].astype(int) <= 12):
                    month_col = i
                elif all(x[:, i].astype(int) <= 31):
                    date_col = i
            else:
                # only month can be string and must be in the middle
                date_col, month_col, year_col = 0, 1, 2
                break

        assert year_col is not None, 'cannot find year in date string'
        try:
            year_pattern = '%Y' if (x[:, year_col].astype(int) > 1000).all() else '%y'
        except (ValueError, TypeError, IndexError):
            return None  # last resort couldn't figure format out, let pandas do it

        month_and_date = lambda m, d, month_pattern: sep.join(
            ('%d', '%s' % month_pattern)) if m > d else sep.join(
            ('%s' % month_pattern, '%d'))

        if year_col == 0:
            if month_col is not None and date_col is not None:
                fmt = sep.join((year_pattern, month_and_date(month_col, date_col, month_pattern)))
            else:
                fmt = sep.join(
                    (year_pattern, '%s' % month_pattern, '%d'))  # default to non US style
        elif year_col == 2:
            if month_col is not None and date_col is not None:
                fmt = sep.join((month_and_date(month_col, date_col, month_pattern), year_pattern))
            else:
                # default to non US style
                fmt = sep.join(('%d', '%s' % month_pattern, year_pattern))
        else:
            raise ValueError('year in the middle of date separators!')

        return fmt

    # This is an extremely fast approach to datetime parsing. Some dates are often repeated.
    # Rather than
    # re-parse these, we store all unique dates, parse them, and use a lookup to convert all dates.
    if isinstance(s, pd.DataFrame):
        out = s.copy(True)  # this is the bottleneck
        for columnName, column in out.iteritems():
            # loop through all the columns passed in
            if 'date' in columnName.lower():
                if column.dtype != '<M8[ns]' and \
                        ~column.isnull().all() and \
                        ~column.str.contains('^[a-zA-z]').all():
                    # if date is provided as a string then ignore and set to int
                    try:
                        col = column.astype(int)
                        out[columnName] = col
                    except:
                        # find the date columns(case in-sensitive), if pandas cant
                        # find the format, ignore error and maintain input
                        unique_dates = pd.to_datetime(column.unique(),
                                                      format=findFormat(column.unique()),
                                                      errors='ignore')
                        dates = dict(zip(column.unique(), unique_dates.tolist()))
                        out[columnName] = column.map(dates.get)

        return out

    else:
        if s.dtype == '<M8[ns]':
            return s
        unique_dates = pd.to_datetime(s.unique(), format=findFormat(s.unique()))
        dates = dict(zip(s.unique(), unique_dates.tolist()))

        return s.map(dates.get)


def excel_date_to_np(xl_date):
    """Excel date serial (as int) to numpy datetime"""
    return np.array(['1899-12-30'], dtype='datetime64[D]') + xl_date


def date_to_excel(pdate):
    """converts datetime to Excel date serial"""
    delta = pdate.to_datetime() - dt.datetime(1899, 12, 30)
    return (delta.days.astype(float) + delta.seconds.astype(float) / 86400).astype(int)


def time_delta_to_days(td):
    """Returns the day difference of a pandas series of timedelta64[ns]"""
    return (td.values / np.timedelta64(1, 'D')).astype(int)
