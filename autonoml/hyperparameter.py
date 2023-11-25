# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:30:05 2023

@author: David J. Kedziora
"""

from .utils import CustomBool
from .settings import SystemSettings as SS

import numpy as np

class Hyperparameter:
    """
    A class containing information for a hyperparameter of an MLComponent.
    This information guides hyperparameter optimisation (HPO).
    """
    def __init__(self, in_val = None,
                 in_default = None, in_min = None, in_max = None, is_log_scale: bool = False,
                 in_info: str = None):
        self.default = in_default
        self.min = in_min
        self.max = in_max
        self.is_log_scale = is_log_scale

        if in_val is None:
            self.val = self.default
        else:
            self.val = in_val

        # An optional description of the hyperparameter to be displayed in strategy files.
        self.info = in_info

    def sample(self):
        try:
            if self.is_log_scale:
                val = np.exp(np.random.uniform(np.log(self.min), np.log(self.max)))
            else:
                val = np.random.uniform(self.min, self.max)
            self.val = val
        except:
            pass

    def to_dict_config(self, do_vary = False):
        dict_config = dict()
        if not self.info is None:
            dict_config["Info"] = self.info
        dict_config["Vary"] = CustomBool(do_vary)
        dict_config["Default"] = self.default
        dict_config["Min"] = self.min
        dict_config["Max"] = self.max
        return dict_config
    
    def from_dict_config(self, in_dict_config):
        self.min = self.validate_min(in_dict_config["Min"])
        self.max = self.validate_max(in_dict_config["Max"])
        self.default = self.validate_default(in_dict_config["Default"], (self.min + self.max)/2)

    def validate_min(self, in_min):
        raise NotImplementedError
    
    def validate_max(self, in_max):
        raise NotImplementedError
    
    def validate_default(self, in_default, in_default_if_none):
        raise NotImplementedError

    def validate_type(self, in_value, in_description: str = None):
        raise NotImplementedError


class HPInt(Hyperparameter):
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
        if in_default is None:
            in_default = int(in_default_if_none)
        in_default = self.validate_type(in_default, "default value")
        return in_default

    def validate_type(self, in_value, in_description: str = None):

        if in_description is None:
            in_description = "value"

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


class HPFloat(Hyperparameter):
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
        if in_default is None:
            in_default = float(in_default_if_none)
        in_default = self.validate_type(in_default, "default value")
        return in_default
        
    def validate_type(self, in_value, in_description: str = None):
        
        if in_description is None:
            in_description = "value"

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