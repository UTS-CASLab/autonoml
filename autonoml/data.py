# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:58:33 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp

from enum import Enum

import pyarrow as pa
import pandas as pd

# class DataType(Enum):

#     def __new__(cls, *args, **kwds):
#         value = len(cls.__members__) + 1
#         obj = object.__new__(cls)
#         obj._value_ = value
#         return obj
    
#     def __init__(self, in_converter, in_string):
#         self.converter = in_converter
#         self.string = in_string

#     FLOAT = float, "float"
#     INTEGER = int, "int"
#     CATEGORICAL = str, "categorical"

#     def to_string(self):
#         return self.string

#     def convert(self, in_val):
#         return self.converter(in_val)


class DataFormatX(Enum):
    """
    Enumeration of ways that X can be formatted.
    X is a set of feature variables across a sequence of instances.
    Note: The 0th format is the 'standard' way a set of variables is kept in data storage.

    Example: Three instances with two features can be represented as follows.

    ARROW_TABLE...
    f_1: string
    f_2: int64
    ----
    f_1: [["a","b","c"]]
    f_2: [[1,2,3]]

    PANDAS_DATAFRAME...
      f_1  f_2
    0   a    1
    1   b    2
    2   c    3

    NUMPY_ARRAY...
    [['a' 1]
     ['b' 2]
     ['c' 3]]

    LIST_OF_DICTS...
    [{'f_1': 'a', 'f_2': 1}, {'f_1': 'b', 'f_2': 2}, {'f_1': 'c', 'f_2': 3}]

    DICT_OF_FEATURE_LISTS...
    {'f_1': ['a', 'b', 'c'], 'f_2': [1, 2, 3]}
    """
    ARROW_TABLE = 0
    PANDAS_DATAFRAME = 1
    NUMPY_ARRAY = 2
    LIST_OF_DICTS = 3
    DICT_OF_FEATURE_LISTS = 4

def reformat_x(in_data, in_format_old, in_format_new, in_keys_features):

    # print(in_format_old)
    # print(in_format_new)

    if in_format_old == in_format_new:
        return in_data

    # Convert back to a standard format.
    if in_format_old == DataFormatX.ARROW_TABLE:
        data_standard = in_data

    elif in_format_old == DataFormatX.PANDAS_DATAFRAME:
        data_standard = pa.Table.from_pandas(in_data)

    elif in_format_old == DataFormatX.NUMPY_ARRAY:
        data_standard = pa.Table.from_pandas(pd.DataFrame(in_data, columns=in_keys_features))

    elif in_format_old == DataFormatX.LIST_OF_DICTS:
        # Assumes feature keys are the same for each dict in the list.
        data_standard = pa.Table.from_pylist(in_data)

    elif in_format_old == DataFormatX.DICT_OF_FEATURE_LISTS:
        data_standard = pa.Table.from_pydict(in_data)

    else:
        raise NotImplementedError
    
    # Convert forward from a standard format.
    if in_format_new == DataFormatX.ARROW_TABLE:
        data_new = data_standard

    elif in_format_new == DataFormatX.PANDAS_DATAFRAME:
        data_new = data_standard.to_pandas()

    elif in_format_new == DataFormatX.NUMPY_ARRAY:
        data_new = data_standard.to_pandas().values

    elif in_format_new == DataFormatX.LIST_OF_DICTS:
        data_new = data_standard.to_pylist()

    elif in_format_new == DataFormatX.DICT_OF_FEATURE_LISTS:
        data_new = data_standard.to_pydict()

    else:
        raise NotImplementedError

    return data_new

class DataFormatY(Enum):
    """
    Enumeration of ways that Y can be formatted.
    Y is a target variable across a sequence of instances.
    Note: The 0th format is the 'standard' way a variable is kept in data storage.

    Example: Three instances with one target can be represented as follows.

    ARROW_ARRAY...
    <pyarrow.lib.BooleanArray object at ...>
    [
    true,
    false,
    true
    ]

    PANDAS_SERIES...
    0     True
    1    False
    2     True
    dtype: bool

    NUMPY_ARRAY...
    [ True False  True]

    LIST...
    [True, False, True]

    LIST_OF_LISTS_ACROSS_TARGETS...
    [[True], [False], [True]]
    """
    ARROW_ARRAY = 0
    PANDAS_SERIES = 1
    NUMPY_ARRAY = 2
    LIST = 3
    LIST_OF_LISTS_ACROSS_TARGETS = 4    # Some predictors allow for multiple targets.

def reformat_y(in_data, in_format_old, in_format_new):

    if in_format_old == in_format_new:
        return in_data

    # Convert back to a standard format.
    if in_format_old == DataFormatY.ARROW_ARRAY:
        data_standard = in_data

    elif in_format_old == DataFormatY.PANDAS_SERIES:
        data_standard = pa.Array.from_pandas(in_data)

    elif in_format_old == DataFormatY.NUMPY_ARRAY:
        data_standard = pa.array(in_data)

    elif in_format_old == DataFormatY.LIST:
        data_standard = pa.array(in_data)

    elif in_format_old == DataFormatY.LIST_OF_LISTS_ACROSS_TARGETS:
        data_temp = [list(vals_target) for vals_target in zip(*in_data)]
        if not len(data_temp) == 1:
            text_error = "Reformat issue: multiple targets are currently not supported."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
        else:
            data_standard = pa.array(data_temp[0])

    else:
        raise NotImplementedError
    
    # Convert forward from a standard format.
    if in_format_new == DataFormatY.ARROW_ARRAY:
        data_new = data_standard

    elif in_format_new == DataFormatY.PANDAS_SERIES:
        data_new = data_standard.to_pandas()

    elif in_format_new == DataFormatY.NUMPY_ARRAY:
        # Note: This is a non-writable array unless specified.
        data_new = data_standard.to_numpy(zero_copy_only = False)

    elif in_format_new == DataFormatY.LIST:
        data_new = data_standard.to_pylist()

    elif in_format_new == DataFormatY.LIST_OF_LISTS_ACROSS_TARGETS:
        data_temp = [data_standard.to_pylist()]
        data_new = [list(vals_instance) for vals_instance in zip(*data_temp)]

    else:
        raise NotImplementedError

    return data_new