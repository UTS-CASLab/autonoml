# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:30:05 2023

@author: David J. Kedziora
"""

from .utils import CustomBool
from .settings import SystemSettings as SS

from typing import List

import numpy as np

class Hyperparameter:
    """
    A class containing information for a hyperparameter of an MLComponent.
    This information guides hyperparameter optimisation (HPO).
    """
    def __init__(self, in_default, in_val = None, in_info: str = None):

        self.default = in_default

        if in_val is None:
            self.val = self.default
        else:
            self.val = in_val

        # An optional description of the hyperparameter to be displayed in strategy files.
        self.info = in_info

    def sample(self):
        raise NotImplementedError

    def to_dict_config(self, do_vary = False):
        dict_config = dict()
        if not self.info is None:
            dict_config["Info"] = self.info
        dict_config["Vary"] = CustomBool(do_vary)
        dict_config["Default"] = self.default
        return dict_config
    
    def from_dict_config(self, in_dict_config):
        self.default = self.validate_default(in_dict_config["Default"], None)
    
    def validate_default(self, in_default, in_default_if_none):
        raise NotImplementedError

    def validate_type(self, in_value, in_description: str = "value"):
        raise NotImplementedError
    


class HPCategorical(Hyperparameter):
    """
    A hyperparameter subclass that stores categories as options.
    If a default is not provided, the first option will be default.
    """
    def __init__(self, in_options: List, in_default = None, *args, **kwargs):

        self.options = self.validate_options(in_options)
        in_default = self.validate_default(in_default, in_options[0])
        super().__init__(in_default = in_default, *args, **kwargs)

    def sample(self):
        try:
            self.val = np.random.choice(self.options)
        except:
            pass

    def to_dict_config(self, *args, **kwargs):
        dict_config = super().to_dict_config(*args, **kwargs)
        if not self.options is None:
            dict_config["Options"] = {option: CustomBool(True) for option in self.options}
        return dict_config

    def from_dict_config(self, in_dict_config):
        options = list()
        for option, choice in in_dict_config["Options"].items():
            if CustomBool(choice):
                options.append(option)
        self.options = self.validate_options(options)
        super().from_dict_config(in_dict_config)
                
    def validate_options(self, in_options):
        if not (isinstance(in_options, list) and len(in_options) > 0):
            raise ValueError("A hyperparameter of categorical type has been given "
                             "options not formatted as a non-empty list.")
        else:
            return in_options

    def validate_default(self, in_default, in_default_if_none):
        if in_default_if_none is None:
            in_default_if_none = self.options[0]
        if in_default is None:
            in_default = in_default_if_none
        if not in_default in self.options:
            raise ValueError("A categorical hyperparameter has been given a default value "
                             "that does not exist in its set of options.")
        return in_default

    def validate_type(self, in_value, in_description: str = "value"):
        raise NotImplementedError



class HPNumerical(Hyperparameter):
    def __init__(self, in_min: int = None, in_max: int = None, is_log_scale: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min = in_min
        self.max = in_max
        self.is_log_scale = is_log_scale

    def sample(self):
        try:
            if self.is_log_scale:
                val = np.exp(np.random.uniform(np.log(self.min), np.log(self.max)))
            else:
                val = np.random.uniform(self.min, self.max)
            self.val = val
        except:
            pass

    def to_dict_config(self, *args, **kwargs):
        dict_config = super().to_dict_config(*args, **kwargs)
        dict_config["Min"] = self.min
        dict_config["Max"] = self.max
        return dict_config

    def from_dict_config(self, in_dict_config):
        self.min = self.validate_min(in_dict_config["Min"])
        self.max = self.validate_max(in_dict_config["Max"])
        super().from_dict_config(in_dict_config)

    def validate_min(self, in_min):
        raise NotImplementedError
    
    def validate_max(self, in_max):
        raise NotImplementedError



class HPInt(HPNumerical):
    def __init__(self, in_default: int = None, in_min: int = None, in_max: int = None,
                 *args, **kwargs):

        in_min = self.validate_min(in_min)
        in_max = self.validate_max(in_max)
        in_default = self.validate_default(in_default, (in_min + in_max)/2)

        if in_max < in_min:
            raise ValueError("A hyperparameter of integer type has been given a maximum bound below its minimum bound.")

        super().__init__(in_default = in_default, in_min = in_min, in_max = in_max,
                         *args, **kwargs)
        
    def validate_min(self, in_min):
        if in_min is None:
            in_min = SS.INT_MIN
        in_min = self.validate_type(in_min, "minimum bound")
        return in_min

    def validate_max(self, in_max):
        if in_max is None:
            in_max = SS.INT_MAX
        in_max = self.validate_type(in_max, "maximum bound")
        return in_max
    
    def validate_default(self, in_default, in_default_if_none):
        if in_default_if_none is None:
            in_default_if_none = (self.min + self.max)/2
        if in_default is None:
            in_default = int(in_default_if_none)
        in_default = self.validate_type(in_default, "default value")
        return in_default

    def validate_type(self, in_value, in_description: str = "value"):

        if not isinstance(in_value, int):
            try:
                old_value = in_value
                in_value = int(old_value)
                if in_value != old_value:
                    raise ValueError
            except:
                raise ValueError("A hyperparameter of integer type has been given "
                                 "a non-integer %s." % in_description)
        return in_value



class HPFloat(HPNumerical):
    def __init__(self, in_default: float = None, in_min: float = None, in_max: float = None,
                 *args, **kwargs):

        in_min = self.validate_min(in_min)
        in_max = self.validate_max(in_max)
        in_default = self.validate_default(in_default, (in_min + in_max)/2)

        if in_max < in_min:
            raise ValueError("A hyperparameter of float type has been given a maximum bound below its minimum bound.")

        super().__init__(in_default = in_default, in_min = in_min, in_max = in_max,
                         *args, **kwargs)
        
    def validate_min(self, in_min):
        if in_min is None:
            in_min = SS.FLOAT_MIN
        in_min = self.validate_type(in_min, "minimum bound")
        return in_min

    def validate_max(self, in_max):
        if in_max is None:
            in_max = SS.FLOAT_MAX
        in_max = self.validate_type(in_max, "maximum bound")
        return in_max
    
    def validate_default(self, in_default, in_default_if_none):
        if in_default_if_none is None:
            in_default_if_none = (self.min + self.max)/2
        if in_default is None:
            in_default = float(in_default_if_none)
        in_default = self.validate_type(in_default, "default value")
        return in_default
        
    def validate_type(self, in_value, in_description: str = "value"):

        if not isinstance(in_value, float):
            try:
                old_value = in_value
                in_value = float(old_value)
                if in_value != old_value:
                    raise ValueError
            except:
                raise ValueError("A hyperparameter of float type has been given "
                                 "a non-float %s." % in_description)
        return in_value