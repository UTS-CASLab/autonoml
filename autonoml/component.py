# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:16:42 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .data import reformat_x, reformat_y

import numpy as np

class Hyperparameter:
    """
    A class containing information for a hyperparameter of an MLComponent.
    This information guides hyperparameter optimisation (HPO).
    """

    def __init__(self, in_val = None,
                 in_default = None, in_min = None, in_max = None, is_exponential = False):
        self.default = in_default
        self.min = in_min
        self.max = in_max
        self.is_exponential = is_exponential

        if in_val is None:
            self.val = self.default
        else:
            self.val = in_val

    def randomise(self):
        try:
            if self.is_exponential:
                val = np.exp(np.random.uniform(np.log(self.min), np.log(self.max)))
            else:
                val = np.random.uniform(self.min, self.max)
            self.val = val
        except:
            pass


class MLComponent:
    """
    A base class for components of a learning pipeline.
    """
    
    count = 0
    
    def __init__(self, in_hpars = None, do_random_hpars = False, *args, **kwargs):
        self.model = None
        self.name = str(MLComponent.count)
        MLComponent.count += 1

        # Every component has a default dictionary of hyperparameters.
        # Their values will be the defaults unless randomised.
        self.hpars = self.new_hpars()
        if do_random_hpars:
            for key in self.hpars:
                self.hpars[key].randomise()

        # Overwrite the defaults with custom specifications provided as a dict.
        if not in_hpars is None:
            for key in in_hpars:
                if key in self.hpars:
                    # If the dict value is a hyperparameter, overwrite the old hyperparameter.
                    if isinstance(in_hpars[key], Hyperparameter):
                        self.hpars[key] = in_hpars[key]
                    # If the dict value is anything else, overwrite the old hyperparameter value.
                    # TODO: Consider error-checking the values.
                    else:
                        self.hpars[key].val = in_hpars[key]
                else:
                    text_warning = "Component '%s' does not possess hyperparameter '%s'."
                    log.warning("%s - %s" % (Timestamp(), text_warning))

        # Every component operates on input data organised in a certain format.
        # This data represents a set of features.
        self.keys_features = None
        self.format_x = None

    @staticmethod
    def new_hpars():
        return dict()
    
    def hpars_as_string(self):
        return ", ".join(key + ": " + str(self.hpars[key].val) for key in self.hpars)

    def learn(self, x, y):
        raise NotImplementedError
    
    def reformat_x(self, x, in_format_old):
        x = reformat_x(in_data = x, 
                       in_format_old = in_format_old, 
                       in_format_new = self.format_x,
                       in_keys_features = self.keys_features)
        return x, self.format_x
    
    def reformat_y(self, y, in_format_old):
        return y, in_format_old

class MLPreprocessor(MLComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, x):
        raise NotImplementedError

class MLPredictor(MLComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Predictors work with output data organised in a certain format.
        # This data represents a target variable.
        self.key_target = None
        self.format_y = None

    def score(self, x, y):
        raise NotImplementedError
    
    def query(self, x):
        raise NotImplementedError
    
    def reformat_y(self, y, in_format_old):
        y = reformat_y(in_data = y, 
                       in_format_old = in_format_old, 
                       in_format_new = self.format_y)
        return y, self.format_y

    def get_feature_importance(self):
        return None