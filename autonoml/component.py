# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:16:42 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .data import reformat_x, reformat_y
from .hyperparameter import Hyperparameter

from typing import List

class MLComponent:
    """
    A base class for components of a learning pipeline.
    """
    count = 0
    
    def __init__(self, in_hpars = None, do_random_hpars: bool = False, *args, **kwargs):
        self.model = None
        self.name = str(MLComponent.count)
        MLComponent.count += 1

        # Every component has a default dictionary of hyperparameters.
        # Their values will be the defaults unless randomised.
        self.hpars = self.new_hpars()
        if do_random_hpars:
            for key in self.hpars:
                self.hpars[key].sample()

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

        # Store a flag that determines whether component updates involve learning or adapting.
        # There is an initial learning, while subsequent observations can be adapted to.
        self.is_untouched = True

        # Every component operates on input data organised in a certain format.
        # This data represents a set of features.
        self.keys_features = None
        self.format_x = None

    #%% Component-specific methods that should be overwritten by custom subclasses.

    @staticmethod
    def new_hpars():
        return dict()

    def learn(self, x, y):
        raise NotImplementedError
    
    def adapt(self, x, y):
        pass

    #%% Utility methods that should not be overwritten by custom subclasses.

    def update(self, x, y):
        if self.is_untouched:
            self.learn(x, y)
            self.is_untouched = False
        else:
            self.adapt(x, y)

    def hpars_as_string(self):
        return ", ".join(key + ": " + str(self.hpars[key].val) for key in self.hpars)
    
    def reformat_x(self, x, in_format_old):
        x = reformat_x(in_data = x, 
                       in_format_old = in_format_old, 
                       in_format_new = self.format_x,
                       in_keys_features = self.keys_features)
        return x, self.format_x
    
    def reformat_y(self, y, in_format_old):
        return y, in_format_old
    
    def set_keys_features(self, in_keys_features: List[str]):
        self.keys_features = in_keys_features



#%% MLPreprocessor subclass.

class MLPreprocessor(MLComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    #%% Component-specific methods that must be overwritten by custom subclasses.

    def transform(self, x):
        """
        Should return a transformed feature space in the format of: self.format_x
        """ 
        raise NotImplementedError
    


#%% MLPredictor subclass.

class MLPredictor(MLComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Predictors work with output data organised in a certain format.
        # This data represents a target variable.
        self.key_target = None
        self.format_y = None
    
    #%% Component-specific methods that must be overwritten by custom subclasses.

    def query(self, x):
        """
        Should return a target space in the format of: self.format_y
        """ 
        raise NotImplementedError
    
    #%% Utility methods that can be overwritten by custom subclasses.
    
    def get_feature_importance(self):
        """
        Should return an array of weights of equal size to the vector of features.
        """
        return None
    
    #%% Utility methods that should not be overwritten by custom subclasses.

    def reformat_y(self, y, in_format_old):
        y = reformat_y(in_data = y, 
                       in_format_old = in_format_old, 
                       in_format_new = self.format_y)
        return y, self.format_y
    
    def set_key_target(self, in_key_target: str):
        self.key_target = in_key_target



#%% Other categories.

class MLOnlineLearner(MLComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_Online"

    def adapt(self, x, y):
        self.learn(x, y)

class MLDummy(MLComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_Dummy"

class MLImputer(MLPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_Imputer"

class MLScaler(MLPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_Scaler"

class MLClassifier(MLPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_Classifier"

class MLRegressor(MLPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_Regressor"