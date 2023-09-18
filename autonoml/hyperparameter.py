# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:30:05 2023

@author: David J. Kedziora
"""

from .utils import CustomBool
import numpy as np

class Hyperparameter:
    """
    A class containing information for a hyperparameter of an MLComponent.
    This information guides hyperparameter optimisation (HPO).
    """

    def __init__(self, in_val = None,
                 in_default = None, in_min = None, in_max = None, is_log_scale = False):
        self.default = in_default
        self.min = in_min
        self.max = in_max
        self.is_log_scale = is_log_scale

        if in_val is None:
            self.val = self.default
        else:
            self.val = in_val

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
        dict_config = {"Vary": CustomBool(do_vary), 
                       "Default": self.default, "Min": self.min, "Max": self.max}
        return dict_config
    
    def from_dict_config(self, in_dict_config):
        self.default = in_dict_config["Default"]
        self.min = in_dict_config["Min"]
        self.max = in_dict_config["Max"]

class HPInt(Hyperparameter):
    pass

class HPFloat(Hyperparameter):
    pass