# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:58:33 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp

from enum import Enum

import pandas as pd

class DataFormatX(Enum):
    """
    Enumeration of ways that X can be formatted.
    X is a set of feature variables across a sequence of instances.

    Example: Three instances with two features can be represented as follows.

    DICT_OF_FEATURE_LISTS...
    {'f_1': ['a', 'b', 'c'], 'f_2': [1, 2, 3]}

    LIST_OF_LISTS_ACROSS_FEATURES...
    [['a', 1], ['b', 2], ['c', 3]]

    LIST_OF_DICTS_ACROSS_FEATURES...
    [{'f_1': 'a', 'f_2': 1}, {'f_1': 'b', 'f_2': 2}, {'f_1': 'c', 'f_2': 3}]

    PANDAS_DATAFRAME...
      f_1  f_2
    0   a    1
    1   b    2
    2   c    3
    """
    DICT_OF_FEATURE_LISTS = 0
    LIST_OF_LISTS_ACROSS_FEATURES = 1
    LIST_OF_DICTS_ACROSS_FEATURES = 2
    PANDAS_DATAFRAME = 3

class DataFormatY(Enum):
    """
    Enumeration of ways that Y can be formatted.
    Y is a target variable across a sequence of instances.

    Example: Three instances with one target can be represented as follows.

    LIST...
    [True, False, True]

    LIST_OF_LISTS_ACROSS_TARGETS...
    [[True], [False], [True]]

    PANDAS_SERIES...
    0     True
    1    False
    2     True
    dtype: bool
    """
    LIST = 0
    LIST_OF_LISTS_ACROSS_TARGETS = 1    # Some predictors allow for multiple targets.
    PANDAS_SERIES = 2

def reformat_x(in_data, in_format_old, in_format_new, in_keys_features):

    if in_format_old == in_format_new:
        return in_data

    # Convert back to a standard format.
    if in_format_old == DataFormatX.DICT_OF_FEATURE_LISTS:
        data_standard = in_data

    elif in_format_old == DataFormatX.LIST_OF_LISTS_ACROSS_FEATURES:
        data_standard = {key:list(vals_feature) for key, vals_feature 
                         in zip(in_keys_features, zip(*in_data))}
        
    elif in_format_old == DataFormatX.LIST_OF_DICTS_ACROSS_FEATURES:
        # Assumes feature keys are the same for each dict in the list.
        data_standard = {key_feature: [instance[key_feature] for instance in in_data] 
                         for key_feature in in_data[0]}
        
    elif in_format_old == DataFormatX.PANDAS_DATAFRAME:
        data_standard = in_data.to_dict(orient = "list")

    else:
        raise NotImplementedError
    
    # Convert forward from a standard format.
    if in_format_new == DataFormatX.DICT_OF_FEATURE_LISTS:
        data_new = data_standard

    elif in_format_new == DataFormatX.LIST_OF_LISTS_ACROSS_FEATURES:
        data_temp = list()
        for key in in_keys_features:
            data_temp.append(data_standard[key])
        data_new = [list(vals_instance) for vals_instance in zip(*data_temp)]

    elif in_format_new == DataFormatX.LIST_OF_DICTS_ACROSS_FEATURES:
        data_new = [dict(zip(data_standard, vals_feature)) 
                    for vals_feature in zip(*data_standard.values())]

    elif in_format_new == DataFormatX.PANDAS_DATAFRAME:
        data_new = pd.DataFrame.from_dict(data_standard)

    else:
        raise NotImplementedError

    return data_new

def reformat_y(in_data, in_format_old, in_format_new):

    if in_format_old == in_format_new:
        return in_data

    # Convert back to a standard format.
    if in_format_old == DataFormatY.LIST:
        data_standard = in_data

    elif in_format_old == DataFormatY.LIST_OF_LISTS_ACROSS_TARGETS:
        data_temp = [list(vals_target) for vals_target in zip(*in_data)]
        if not len(data_temp) == 1:
            text_error = "Reformat issue: multiple targets are currently not supported."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
        else:
            data_standard = data_temp[0]

    elif in_format_old == DataFormatY.PANDAS_SERIES:
        data_standard = list(in_data)

    else:
        raise NotImplementedError
    
    # Convert forward from a standard format.
    if in_format_new == DataFormatY.LIST:
        data_new = data_standard

    elif in_format_new == DataFormatY.LIST_OF_LISTS_ACROSS_TARGETS:
        data_temp = [data_standard]
        data_new = [list(vals_instance) for vals_instance in zip(*data_temp)]

    elif in_format_new == DataFormatY.PANDAS_SERIES:
        data_new = pd.Series(data_standard)

    else:
        raise NotImplementedError

    return data_new