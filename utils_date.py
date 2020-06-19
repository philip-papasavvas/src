# Created 17 June 2020. Transplanted from utils_generic.py

import datetime as dt
from re import search, sub

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def date_to_str(d):
    return d.astype(str).replace("-", "")


def char_to_date(s):
    """
    Turning date from object to np.datetime64

    Args:
        s (pd.Dataframe, pd.Series): Pass in DataFrame if multi column process is needed

    Returns
        (pd.Series, pd.DataFrame): all columns containing "date" (case in-sensitive) will be amended

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
            monthPattern = '%m'
        except ValueError:
            monthPattern = '%b'

        yearCol, monthCol, dateCol = None, None, None
        for i in range(x.shape[-1]):
            if x[:, i].dtype != object:
                if all(x[:, i].astype(int) > 1000):
                    yearCol = i
                elif all(x[:, i].astype(int) <= 12):
                    monthCol = i
                elif all(x[:, i].astype(int) <= 31):
                    dateCol = i
            else:
                dateCol, monthCol, yearCol = 0, 1, 2  # only month can be string and must be in the middle
                break

        assert yearCol is not None, 'cannot find year in date string'
        try:
            yearPattern = '%Y' if (x[:, yearCol].astype(int) > 1000).all() else '%y'
        except (ValueError, TypeError, IndexError):
            return None  # last resort couldn't figure format out, let pandas do it

        monthAndDate = lambda m, d, monthPattern: sep.join(('%d', '%s' % monthPattern)) if m > d else sep.join(
            ('%s' % monthPattern, '%d'))

        if yearCol == 0:
            if monthCol is not None and dateCol is not None:
                fmt = sep.join((yearPattern, monthAndDate(monthCol, dateCol, monthPattern)))
            else:
                fmt = sep.join((yearPattern, '%s' % monthPattern, '%d'))  # default to non US style
        elif yearCol == 2:
            if monthCol is not None and dateCol is not None:
                fmt = sep.join((monthAndDate(monthCol, dateCol, monthPattern), yearPattern))
            else:
                fmt = sep.join(('%d', '%s' % monthPattern, yearPattern))  # default to non US style
        else:
            raise ValueError('year in the middle of date separators!')

        return fmt

    # This is an extremely fast approach to datetime parsing. Some dates are often repeated. Rather than
    # re-parse these, we store all unique dates, parse them, and use a lookup to convert all dates.
    if isinstance(s, pd.DataFrame):
        out = s.copy(True)  # this is the bottleneck
        for columnName, column in out.iteritems():
            # loop through all the columns passed in
            if 'date' in columnName.lower():
                if column.dtype != '<M8[ns]' and ~column.isnull().all() and ~column.str.contains('^[a-zA-z]').all():
                    # if date is provided as a string then ignore and set to int
                    try:
                        col = column.astype(int)
                        out[columnName] = col
                    except:
                        # find the date columns(case in-sensitive), if pandas cant find the format, ignore error and maintain input
                        uDates = pd.to_datetime(column.unique(), format=findFormat(column.unique()), errors='ignore')
                        dates = dict(zip(column.unique(), uDates.tolist()))
                        out[columnName] = column.map(dates.get)

        return out

    else:
        if s.dtype == '<M8[ns]':
            return s
        uDates = pd.to_datetime(s.unique(), format=findFormat(s.unique()))
        dates = dict(zip(s.unique(), uDates.tolist()))

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
