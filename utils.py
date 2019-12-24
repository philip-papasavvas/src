"""
Created on: 6 May 2019
Created by: Philip.P_adm

Utils module for generic useful functions, divided into classes
"""

# usual (built-in) imports
import pandas as pd
import numpy as np

import datetime as dt
import datetime
from dateutil.relativedelta import *

import os
from re import sub, search, escape
import re

# third party imports
import pyperclip
import xlrd
from math import ceil

# local imports (if any)

get_folder = lambda path: os.path.split(path)[0]
date_to_str = lambda d: d.astype(str).replace('-', '')


class Date():
    """Gathered all methods that deal with dates here"""
    @staticmethod
    def date_shift(date, d=0, w=0, m=0, y=0, reverse=True):
        """Method to take an input time and specify a difference offset

        Args:
            date (datetime)
            d, w, m , y (int, default 0): Days, weeks, months, years offset
            reverse (bool, default True): Reverse the data (or False to add the difference)

        Returns:
            new_date (datetime)

        Examples
            >>> Date.date_shift(date=datetime.datetime(year=2010, month=10, day=10), m=10) # datetime.datetime(2009, 12, 10, 0, 0)
            >>> Date.date_shift(date=datetime.datetime(year=2010, month=10, day=10), y=5, reverse=False) # datetime.datetime(2015, 10, 10, 0, 0)

        """
        orig_date = date
        time_offset = relativedelta(years=y, months=m, weeks=w, days=d)

        if not reverse:
            new_date = orig_date + time_offset
        else:
            new_date = orig_date - time_offset

        return new_date

    @staticmethod
    def datePlusTenorNew(date, pillar, reverse=False, expressInDays=False):
        '''
        Function to return the day count fraction(per year) between two dates based on convention
    
        Parameters
        ----------
            date : np.datetime64, np.array(np.datetime64[D])
            pillar : str, adds tenor to date, expressed as number letter pair
                    examples of number letter pairs: 1d, 4D, 3w, 12W, 2M, 5m, 1y, 5Y
                    other values allowed are 'spot', 'ON', 'O/N', 'SN', 'S/N'
            reverse: bool, default False. If False then the function will run to a time in the future,
                        if True then it can successfully subtract the pillar from the date
            expressInDay: bool, default False. If False a date is returned, if True an float is
                    returned representing the number of days between date and pillar date / 365
        Returns
        -------
            newDate: np.datetime64, np.array(np.datetime64[D])
                     or if expressInDays = True: float, np.array()
    
        Example:
            >>> currentDate = np.datetime64("2018-11-23")
            >>> dateDelta = Date.datePlusTenorNew(currentDate, "6M", reverse = True) #numpy.datetime64('2018-05-23')
            >>> dateDelta_2 = Date.datePlusTenorNew(currentDate, "9M") #numpy.datetime64('2019-08-23')
            >>> dateDelta_3 = Date.datePlusTenorNew(currentDate, "1M", reverse = True) #numpy.datetime64('2018-10-23')
        '''
        date = date.astype('datetime64[D]')
        if pillar == 'spot':
            newDate = date
        if search('y|Y', pillar):
            y = int(sub('\\D', '', pillar))
            yearly = date.astype('datetime64[Y]')
            monthly = date.astype('datetime64[M]')
            myDay = date.astype(int) - monthly.astype('datetime64[D]').astype(int) + 1
            myMonth = monthly.astype(int) - yearly.astype('datetime64[M]').astype(int) + 1
            if not reverse:
                newDate = yearly + y
            else:
                newDate = yearly - y
    
            newDate = newDate.astype('datetime64[M]') + myMonth - 1
            # Leap year case, make sure not rolling into next month
            outmonth = newDate.copy()
            newDate = newDate.astype('datetime64[D]') + myDay - 1
            if outmonth != newDate.astype('datetime64[M]'):
                newDate = newDate.astype('datetime64[M]').astype('datetime64[D]') - 1
        if search('m|M', pillar):
            m = int(sub('\\D', '', pillar))
            monthly = date.astype('datetime64[M]')
            myDay = date.astype(int) - monthly.astype('datetime64[D]').astype(int) + 1
            if not reverse:
                newDate = monthly + m
            else:
                newDate = monthly - m
            outmonth = newDate.copy()
            # add the days
            newDate = newDate.astype('datetime64[D]') + myDay - 1
            if outmonth != newDate.astype('datetime64[M]'):
                newDate = newDate.astype('datetime64[M]').astype('datetime64[D]') - 1
        if search('w|W', pillar):
            w = int(sub('\\D', '', pillar))
            if not reverse:
                newDate = date + np.timedelta64(w * 7, 'D')
            else:
                newDate = date - np.timedelta64(w * 7, 'D')
        if search('d|D', pillar):
            d = int(sub('\\D', '', pillar))
            if not reverse:
                newDate = date + np.timedelta64(d, 'D')
            else:
                newDate = date - np.timedelta64(d, 'D')
    
        if expressInDays:
                newDate = (newDate - date) / np.timedelta64(1, 'D')
    
        return newDate


    @staticmethod
    def previousDate(dataframe, date, timeDifference):
        """
        Function to return a date in the past according to the input date provided
        for the dataframe being analysed
    
        Params:
            dataframe:
                Contains datetime index
            date: np.datetime64
                Date at which you want to calculate the time difference from
            timeDifference:
                Expressed in terms of "M", "Y", "D", eg. "3M"
    
        Returns:
            dateDelta:
                Date in the past according to specified time difference
    
        Example:
            >>> dates = pd.date_range(start= "2010-01-01", end= "2020-01-01")
            >>> df = pd.DataFrame(data = [ i for i in range(len(dates))], index = dates)
            >>> Date.previousDate(dataframe = df, date= "2018-05-05", timeDifference = "6M")
    
        """
        df = dataframe
        date = np.datetime64(date)
    
        dateDelta = Date.datePlusTenorNew(date, timeDifference, reverse=True)
        if dateDelta not in df.index:
            if dateDelta - 1 not in df.index:
                dateDelta = dateDelta - 2
            else:
                dateDelta = dateDelta - 1
        else:
            dateDelta = dateDelta
        return dateDelta


    @staticmethod
    def char_to_date(s,dayfirst=False,format=None,infer_datetime_format=False):
        """
        Turning date from object to np.datetime64
    
        Parameters
        ----------
            s : pd.Dataframe, pd.Series
                Pass in dataframe if multi column process is needed
        Returns
        -------
            pd.Series
            pd.DataFrame
                all columns containing "date" (case in-sensitive) will be amended
        Note
        -----
            This method can handle EITHER "/" or "-" date separators but not a combination of both.
            Users should check that there are no mixtures of separators if s is an array
        """
    
        def findFormat(s):
            yearPattern = '%Y'
            try:
                sep = '/'
                if pd.Series(s).str.contains('-').all():
                    sep = '-'
                x = pd.Series(s).str.split('/|-',expand=True).values
                x = x.astype(int)
                monthPattern = '%m'
            except ValueError:
                monthPattern = '%b'
    
            yearCol, monthCol, dateCol = None, None, None
            for i in range(x.shape[-1]):
                if x[:,i].dtype != object:
                    if all(x[:,i].astype(int) > 1000):
                        yearCol=i
                    elif all(x[:,i].astype(int) <= 12):
                        monthCol = i
                    elif all(x[:,i].astype(int) <= 31):
                        dateCol = i
                else:
                    dateCol,monthCol,yearCol = 0,1,2 # only month can be string and must be in the middle
                    break
    
            assert yearCol is not None, 'cannot find year in date string'
            try:
                yearPattern = '%Y' if (x[:, yearCol].astype(int) > 1000).all() else '%y'
            except (ValueError, TypeError, IndexError):
                return None # last resort couldn't figure format out, let pandas do it
    
            monthAndDate = lambda m,d,monthPattern : sep.join(('%d','%s' % monthPattern)) if m>d else sep.join(('%s' % monthPattern,'%d'))
    
            if yearCol == 0:
                if monthCol is not None and dateCol is not None:
                    fmt = sep.join((yearPattern,monthAndDate(monthCol,dateCol,monthPattern)))
                else:
                    fmt = sep.join((yearPattern,'%s' % monthPattern,'%d')) # default to non US style
            elif yearCol == 2:
                if monthCol is not None and dateCol is not None:
                    fmt = sep.join((monthAndDate(monthCol,dateCol,monthPattern),yearPattern))
                else:
                    fmt = sep.join(('%d','%s' % monthPattern, yearPattern)) # default to non US style
            else:
                raise ValueError('year in the middle of date separators!')
    
            return fmt
    
        # This is an extremely fast approach to datetime parsing. Some dates are often repeated. Rather than
        # re-parse these, we store all unique dates, parse them, and use a lookup to convert all dates.
        if isinstance(s, pd.DataFrame):
            out = s.copy(True) #this is the bottleneck
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
                            dates = dict(zip(column.unique(),uDates.tolist()))
                            out[columnName] = column.map(dates.get)
    
            return out
    
        else:
            if s.dtype == '<M8[ns]':
                return s
            uDates = pd.to_datetime(s.unique(), format=findFormat(s.unique()))
            dates = dict(zip(s.unique(),uDates.tolist()))
    
            return s.map(dates.get)


    @staticmethod
    def x2pdate(xldate):
        """Excel date serial (as int) to numpy datetime"""
        return np.array(['1899-12-30'], dtype='datetime64[D]') + xldate


    @staticmethod
    def p2xdate(pdate):
        """converts datetime to Excel date serial"""
        delta = pdate.to_datetime() - dt.datetime(1899, 12, 30)
        return (delta.days.astype(float) + delta.seconds.astype(float) / 86400).astype(int)


    @staticmethod
    def daysToInt(days):
        '''Convert days to integer value whilst returning NaT as 0'''
        identifyNAT = np.array([np.nan]).astype(int)
        days = (days / np.timedelta64(1, 'D')).astype(int)
        ## wherever there are NaT, return 0 days, as integer arrays cannot hold nan
        days[days == identifyNAT] = 0
        return (days)


    @staticmethod
    def timeDelta_to_days(td):
        """Returns the day difference of a pandas series of timedelta64[ns]"""
        return (td.values / np.timedelta64(1, 'D')).astype(int)


class Utils():
    """Collated all Utils functions here as static methods"""

    @staticmethod
    def to_array(*args):
        """
        Turning x into np.ndarray

        Other Parameters
        ----------
        x : list, tuple, np.ndarray, pd.Series, np.datetime64, datetime.datetime

        Yields
        -------
        :class:'np.ndarray'

        Raises
        ------
        ValueError if x is not in the listed type

        Example
        -------
        >>> import numpy as np
        >>> from utils import Utils
        >>> x,y,z = Utils.to_array(2,["a","b"],None)
        >>> date_array, =  Utils.to_array(np.datetime64("2019-01-01"))
        """
        for x in args:

            if isinstance(x, dt.date):
                yield np.array([x.strftime('%Y-%m-%d')], dtype='datetime64[D]')
            elif isinstance(x, (list, tuple, np.ndarray)):
                yield np.array(x)
            elif isinstance(x, (pd.Series, pd.core.indexes.base.Index)):
                yield x.values
            elif isinstance(x, (int, np.int32, np.int64, float, str)):
                yield np.array([x], dtype=type(x))
            elif isinstance(x, np.datetime64):
                yield np.array([x], 'datetime64[D]')
            elif x is None:
                yield np.array([])
            else:
                raise ValueError('unable to convert to array')


    @staticmethod
    def average(*args):
        """
        Finds arithmetic mean of an array input
        >>> Utils.average(*[1,2,3]) #  2.0
        >>> Utils.average(1,2,5) # 2.6666666666666665
        """
        return sum(args, 0.0) / len(args)


    @staticmethod
    def difference(a, b):
        """Give difference between two iterables by keeping values in first
        >>> Utils.difference([3,10,9],[3,4,10]) #{9}
        """
        set_a, set_b = set(a), set(b)
        return set_a.difference(set_b)


    @staticmethod
    def match(cls,x,y,strict=True):
            '''
            Finds the index of x's elements in y. This is the same function as R implements.

            Parameters
            ----------
                x,y : list or np.ndarray or pd.Series
                strict : bool
                    Whether to raise error if some elements in x are not found in y

            Returns
            -------
                list or np.ndarray of int

            Raises
            ------
            AssertionError
                If any element of x is not in y

            '''
            # just be sure it handles pd.Series as well, but in reality one should not use this function for pd.Series
            x,y = cls.to_array(x,y)
            mask = x[:,None] == y

            rowMask = mask.any(axis=1)

            if strict:
                # this is 40x faster pd.core.algorithms.match(x,y)
                assert rowMask.all(), "%s not found, uniquely : %s " % ((~rowMask).sum(),np.array(x)[~rowMask])
                out = np.argmax(mask,axis=1) # returns the index of the first match

            else:
                # this is 26x faster than pd.core.algorithms.match(x,y,np.nan)
                # return floats where not found elements are returned as np.nan
                out = np.full(np.array(x).shape, np.nan)
                out[rowMask] = np.argmax(mask[rowMask],axis=1)

            return out


    @staticmethod
    def find(folderPath,pattern='.*',fullPath=False, expectOne=True):
        """
        To find path(s) of file(s), especially useful for searching the same file pattern in multiple folders

        Parameters
        ----------
            path : str, list, np.ndarray
                folder path(s). If multiple, it will be searched in its order
            pattern : str, list/tuple
                regex pattern. Use list/tuple if you need multiple conditions
            fullPath : bool, optional
                if the full path of the files are needed
            expectOne : bool, optional
                True will raise AssertionError if more than one file is found
        Returns
        -------
            str
                if one file is found
            list of str
                if multiple files are found
        Note
        ----
            This function can only handle same search pattern for every folderPath, and will return the first one it finds \
            if there are multiple folderPath. If it cannot find any, it will raise exceptions about the first folderPath

        Raises
        ------
            FileNotFoundError
                if folderPath(s) or pattern(s) does not match any findings
            FileExistsError
                if more than one file is found when expectOne = True
        """
        if isinstance(folderPath,str):
            folderPath, = Utils.to_array(folderPath)

        for i,path in enumerate(folderPath):

            try:
                listOfFiles = os.listdir(path = path)
            except (FileNotFoundError,OSError) as err:
                if i < len(folderPath) - 1:
                    print(err.args[-1] + ' for "%s",... trying next' % path)
                    continue  # go for next folderPath
                # out of luck
                err.args = (err.args[0],err.args[1] + ': %s' % path) # raise with first folderPath
                raise

            if isinstance(pattern,(list,tuple)):
                n = len(pattern)
                # multi condition pattern matching
                ipattern = '|'.join(pattern)
                # strict matching of all required patters
                files = [f for f in listOfFiles if np.unique(re.findall(ipattern,f,re.IGNORECASE)).size == n]
            else:
                files = [f for f in listOfFiles if re.findall(pattern, f,re.IGNORECASE)]

            try:
                if len(files) == 0:
                    # remove some special characters before raising error
                    if isinstance(pattern,str):
                        pattern = re.sub('[^A-Za-z0-9_.-]+','',pattern)
                    else:
                        pattern = [re.sub('[^A-Za-z0-9_.-]+', '', pat) for pat in pattern]
                    raise FileNotFoundError('%s exists but no file names with pattern: %s' % (path,pattern))

                elif len(files) > 1 and expectOne:
                    raise FileExistsError('%s exists but %s files found' % (path,len(files)))

            except (FileNotFoundError,FileExistsError) as err:
                if i < len(folderPath) - 1:
                    print(err.args[0] + ',... trying next')
                    continue  # go for next folderPath
                # out of luck, raise with first folderPath
                err.args = (re.sub(path.replace('\\','/'),folderPath[0],err.args[0]),) # re.sub doesn't like double backslashes
                raise

            break # stop loop when we got the files we wanted

        if fullPath:
            if path[-1] == '/':
                path = path[:-1]
            files = ['/'.join((path,f)) for f in files]

        if len(files) == 1 and expectOne:
            files = files[0]

        return files


    @staticmethod
    def concatColumns(sep='',*args):
            """Concatenate multiple columns of pd.DataFrame with sep"""
            df = pd.DataFrame()
            for arg in args:
                df = pd.concat([df,arg],axis=1,ignore_index=True)
            try:
                out = df.astype(str).add(sep).sum(axis=1).str.replace('%s+$' % escape(sep),'') # removes trailing sep
                # need to make any columns with nan to output NaN, which is the result when 'A' + '_' + 'NaN'
                mask = df.isnull().any(axis=1)
                out[mask] = np.nan
            except AttributeError:
                # incase of empty data frame
                out = pd.Series()
            return out


    @staticmethod
    def readExcel(file):
        '''
        Read excel files

        Parameters:
        -----------

        file: str,
            path to excel file to be read

        Returns
        -------
            dict with keys as sheet names and items as sheet contents as pd.dataframe
        '''

        xl_workbook = xlrd.open_workbook(file)
        sheet_names = xl_workbook.sheet_names()
        out = {}
        for sheet in sheet_names:
            fl = pd.read_excel(file, sheet_names[0])
            out.update({sheet: fl})

        return out


    @staticmethod
    def rolling_window(a, size):
        """
        Returns a rolling window of a n-dimensional array

        Parameters
        ----------
            a : np.ndarray
            size : int
                size of rolling window
        Returns
        -------
            n + 1 dimensional array
                the extra dimension added at the end

        Example
        -------
        >>> import numpy as np
        >>> from utils import Utils
        >>> Utils.rolling_window(np.random.randn(20),5).mean(axis=-1) # find the 5 element rolling mean of an array

        """
        a_ext = np.concatenate((np.full(a.shape[:-1] + (size-1,),np.nan) ,a),axis=-1)
        strides = a_ext.strides + (a_ext.strides[-1],)
        return np.lib.stride_tricks.as_strided(a_ext, shape=(a.shape + (size,)), strides=strides)


    @staticmethod
    def array_to_clipboard(array):
        """
        Copies an array into a string format acceptable by Excel.

        Note
        ----
            Columns separated by "\\t", rows separated by "\\n"
        """
        # Create string from array
        line_strings = []

        array = array.astype(str)

        if array.ndim > 2:
            raise ValueError('could not operate on array with ndim >2')
        elif array.ndim == 1:
            for line in array:
                line_strings.append(line.replace("\n", ""))
        else:
            for line in array:
                line_strings.append('\t'.join([l for l in line]).replace("\n", ""))

        array_string = "\r\n".join(line_strings)

        # Put string into clipboard
        pyperclip.copy(array_string)


    @staticmethod
    def formatCsvCommas(path):
        """ Reads in data and replaces ", " with "" before returning List of cleaned data"""

        data = []
        with open(path, newline='') as f:
            for lines in f:
                new_line = lines.replace(", ", "")
                data.append(new_line)

        # compile lines and remove special charaters
        data = pd.Series(data).str.split(',', expand=True).replace({'\\n': '', '\\r': ''}, regex=True)
        data.columns = data.iloc[0, :]
        data = data.iloc[1:].reset_index(drop=True)
        return data


    @staticmethod
    def prep_fund_data(df_path, date_col="Date"):
        """Prep fund data (csv) using char to date and setting 'Date' as the index

        Args:
            df_path (str): Path to dataframe

        Returns:
            df (dataframe): Dataframe with date columns converted to np.datetime64
        """
        df = pd.read_csv(df_path)
        df = Date.char_to_date(df)

        assert date_col in df.columns, "The date column: {} is not specified in the function".format(date_col)

        df.set_index('Date', inplace=True)
        return df


class Dict():
    """Compiling all the utils methods for dictionaries"""

    @staticmethod
    def flatten_dict(d):
        """Flatten dictionary d

        Example
            >>> Dict.flatten_dict(d={"a":{1}, "b":{"yes":{"more detail"}, "no": "level below" }})
            returns {'a': {1}, 'b.yes': {'more detail'}, 'b.no': 'level below'}
        """

        def items():
            for key, value in d.items():
                if isinstance(value, dict):
                    for subkey, subvalue in Dict.flatten_dict(value).items():
                        yield key + "." + subkey, subvalue
                else:
                    yield key, value

        return dict(items())


    @staticmethod
    def return_keys(dict):
        """
        Returns keys of a dict in a list
        >>> Dict.return_keys({'a':1, 'b':2, 'c':3}) #
        """
        return list(dict.keys())


    @staticmethod
    def return_values(dict):
        """
        Returns keys of a dict in a list
        >>> Dict.return_values({'a':1, 'b':2, 'c':3})
        """
        return list(dict.values())


class List():
    """Collection of list methods"""

    @staticmethod
    def comma_sep(lst):
        """Gets a list and returns one single string with elements separated by a comma
        >>> List.comma_sep(["hello","this","is","an", "example"]) # 'hello,this,is,an,example'
        """
        return ",".join(lst)


    @staticmethod
    def has_duplicates(lst):
        """
        Checks if a list has duplicate values
        >>> List.has_duplicates([1,2,4,5]) # False
        >>> List.has_duplicates([1,2,2,5]) # True
        """
        return len(lst) != len(set(lst))


    @staticmethod
    def all_unique(lst):
        """
        Check if a given list has duplicate elements
        >>> List.all_unique([1,2,3,4]) # True
        >>> List.all_unique([1,2,2,4]) # False
        """
        return len(lst) == len(set(lst))


    @staticmethod
    def chunk(lst, chunk_size):
        """
        Split a list into a list of smaller lists defined by chunk_size
        >>> List.chunk([1,2,4,5,6,6],5)
        >>> List.chunk([1,2,4,5,6,6],2)
        """
        return list(
            map(lambda x: lst[x * chunk_size: x * chunk_size + chunk_size],
                list(range(0, ceil(len(lst) / chunk_size))))
        )


    @staticmethod
    def count_occurences(lst, value):
        """
        Function to count occurrences of value in a list
        >>> List.count_occurences(lst=[1,2,3,4,4,4,4], value=4) # 4
        """
        return len([x for x in lst if x == value and type(x) == type(value)])


    @staticmethod
    def flatten(lst):
        """
        Flatten a list using recursion
        >>> List.flatten(lst=[1,2,[3,4,5,[6,7]]]) # [1, 2, 3, 4, 5, 6, 7]
        """
        res = []
        res.extend(List.flatten_list(list(map(lambda x: List.flatten(x) if type(x) == list else x, lst))))
        return res

    @staticmethod
    def flatten_list(arg):
        """Flatten lists
        Function used for flattening lists
        >>> List.flatten_list([2,3,5,[7,8]])
        """
        ls = []
        for i in arg:
            if isinstance(i, list):
                ls.extend(i)
            else:
                ls.append(i)
        return ls


class SecuritiesAnalysis():
    """Collate together all methods to do with securities analysis - returns/log returns/Sharpe/Sortino Ratio"""

    @staticmethod
    def log_daily_returns(data):
        """Give log daily returns"""
        log_daily_return = data.apply(lambda x: np.log(x) - np.log(x.shift(1)))[1:]
        return log_daily_return


    @staticmethod
    def daily_return(data):
        """Function to generate daily returns given input data (in dataframe, dtypes float, no time data)

        Example:
            >>> SecuritiesAnalysis.daily_return(data=pd.DataFrame([1,2,3,4]))
        """
        return data.pct_change(1).iloc[1:, ]


    @staticmethod
    def annual_return(data):
        """Annual return from securities data(frame)"""
        daily_rtn = data.pct_change(1).iloc[1:, ]
        ann_rtn = np.mean(daily_rtn) * 252
        return ann_rtn


    @staticmethod
    def annual_vol(data):
        """Annual return from securities data(frame)"""
        daily_rtn = data.pct_change(1).iloc[1:, ]
        ann_vol = np.std(daily_rtn) * np.sqrt(252)
        return ann_vol


    @staticmethod
    def info_ratio(data):
        """Annual return from securities data(frame)"""
        daily_rtn = data.pct_change(1).iloc[1:, ]
        annual_rtn = np.mean(daily_rtn) * 252
        ann_vol = np.std(daily_rtn) * np.sqrt(252)
        info_ratio = np.divide(annual_rtn, ann_vol)
        return info_ratio


    @staticmethod
    def sharpe_ratio(data, risk_free):
        """Function to give annualised Sharpe Ratio measure from input data, as well as risk free rate

        Args:
            data (dataframe)
            risk_free (float): Risk free rate, as a decimal, so RFR of 6% = 0.06
        """
        daily_rtn = data.pct_change(1).iloc[1:, ]
        annual_rtn = np.mean(daily_rtn) * 252
        ann_vol = np.std(daily_rtn) * np.sqrt(252)
        sharpe_ratio = np.divide(annual_rtn - risk_free, ann_vol)
        return sharpe_ratio


    @staticmethod
    def sortino_ratio(data, target_return, risk_free, rtn_period=1):
        """Method to calculate Sortino Ratio (gives a better measure of downside volatility, thus risk.
        Unlike the Sharpe Ratio it does not penalise upside volatility.

        Args:
            data (dataframe): Original dataframe of input data
            target_return (float): Target return (for the return period)
            risk_free (float): Risk free rate, usually annualised
            rtn_period (float, default 1): Specify the return period (number of days) for the ratio.

        Returns:
            sortino (ndarray)
        """

        prd_return = data.pct_change(rtn_period).iloc[1:, ]
        downside_return = np.array(prd_return.values - target_return)

        inner_bit = np.minimum(np.zeros(shape=downside_return.shape[1]), downside_return)

        tdd_sum = np.sum(np.square(inner_bit), axis=0)
        target_downside_dev = np.sqrt(tdd_sum / len(prd_return))

        sortino = (prd_return.mean() - risk_free) / target_downside_dev

        return sortino



class BloombergData():
    """Bring together all Bloomberg raw data methods to clean the data"""

    @staticmethod
    def return_clean_df(inputFile):
        """
        Params:
            inputFile: csv
                Data starts on the third row in format date | float.
                e.g.
                TICKER      | (empty)    | (empty) | ...
                "Date"      | "PX_LAST"  | (empty) | ...
                DD/MM/YYYY  | float      | (empty) | ...
        Read csv file with two columns from bloomberg
        one with the date, and the other with the price.

        Returns:
             data: dataframe
                Melted dataframe

        Example:
            >>> ex = BloombergData.return_clean_df("D:/Root_D/Philip/20181201 Practice/example-bbg-df.csv")

        """
        a = pd.read_csv(inputFile, header=None)
        a = a.copy()
        product = a.iloc[0,0]

        if a.iloc[1,1] == "PX_LAST":
            measure = "price"
        else:
            measure = "[to populate]"

        data = a.iloc[2:,:]
        data = data.copy()
        data['product'] = product
        data.columns = ['date', 'price', 'product']
        data = data[['product', 'date', 'price']]
        return data

    @staticmethod
    def return_melted_df(input_file):
        """
        Read csv file with Bloomberg data (in format below with or without blank columns) and create
        melted pivot format inputFile
        bb ticker | (empty)         | bb ticker | (empty)
        Date      | "PX_LAST"       | Date      | "PX_LAST"
        dd/mm/yyyy| float           | dd/mm/yyyy| float

        Returns:
        Contract    | Date      | Price
        xxxx        | dd/mm/yy  | ##.##

        Example:
            >>> sample_df = BloombergData.return_melted_df(input_file="D:/Root_D/Philip/20181201 Practice/example-bbg-df2.csv")
        """

        x = pd.read_csv(input_file, header=None, parse_dates=True)
        x.dropna(axis=1, how='all', inplace=True)

        if any(pd.DataFrame(x.iloc[1,:]).drop_duplicates() == ['Date', "PX_LAST"]):
            x = x.copy(True)
            if x.shape[1] % 2 == 0:
                df = pd.DataFrame()
                for i in range(0, x.shape[1], 2):
                    product = x.iloc[0,i] #extract name of product/security
                    data = x.iloc[2:, i:i+2]
                    data.dropna(inplace=True)
                    data.reset_index(drop=True,inplace=True)

                    data['product'] = product

                    data.columns = ['date', 'price', 'product']
                    #data['date'] = np.datetime64(data['date'])
                    dates = np.array(data['date'])

                    # create a mask to ensure that all entries have the correct date format,
                    # not just Excel serial numbers
                    res = []
                    for i in dates:
                        res.append(len(i))
                    res = np.array(res)
                    mask = (res != 10)
                    datesToCorrect = pd.to_datetime(Date.x2pdate(np.array(dates[mask], dtype='int32'))).strftime('%Y/%m/%d')
                    dates[mask] = datesToCorrect
                    data['date'] = dates

                    data = data[['product', 'date', 'price']]
                    df = df.append(data)
            else:
                print("Dataframe is not in the correct format")
        else:
            raise TypeError("The dataframe is not in the format expected with columns:"
                            "[Date, PX_LAST]")

        # df.set_index('product')
        return df


# merge dictionaries: {**a,**b}
# merge two lists: dict(zip(list_one,list_two))

# if a function returns multiple arguments, label as follows for the variable unpacking: a,*_, b = var1, ...., var2
# example
# def func(dict):
#     return list(dict.keys())[0], list(dict.keys())[1], list(dict.keys())[2],list(dict.keys())[3]
 # (a, *_, c) = func({'a':1, 'b':2, 'c':3, 'd':4}) --> a= 'a', c='d'