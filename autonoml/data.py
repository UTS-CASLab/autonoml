# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:58:33 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp

from enum import Enum
# from copy import deepcopy

import pandas as pd

class DataFormatX(Enum):
    """
    Enumeration of ways that X can be formatted.
    X covers a set of feature variables across a sequence of instances.
    """
    DICT_OF_FEATURE_LISTS = 0
    LIST_OF_LISTS_ACROSS_FEATURES = 1
    PANDAS_DATAFRAME = 2

class DataFormatY(Enum):
    """
    Enumeration of ways that Y can be formatted.
    Y covers a target variable across a sequence of instances.
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