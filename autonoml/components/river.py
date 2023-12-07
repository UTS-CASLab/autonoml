# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:49:12 2023

@author: David J. Kedziora
"""

from ..hyperparameter import Hyperparameter, HPInt, HPFloat
from ..component import MLPreprocessor, MLPredictor, MLOnlineLearner, MLScaler, MLRegressor
from ..data import DataFormatX, DataFormatY

import numpy as np

from river import preprocessing, linear_model, metrics



class RiverPreprocessor(MLOnlineLearner, MLPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_River"

class OnlineStandardScaler(MLScaler, RiverPreprocessor):
    def __new__(cls, in_hpars = None, *args, **kwargs):

        cls_this = OnlineStandardScaler

        # When a new object of this class is created, create the right subclass.
        if cls == cls_this:
            key = "batch_size"

            # Check default hyperparameters.
            hpars = cls.new_hpars()
            batch_size = hpars[key].val

            # Check supplied hyperparameters, if any.
            if not in_hpars is None:
                if key in in_hpars:
                    if isinstance(in_hpars[key], Hyperparameter):
                        batch_size = in_hpars[key].val
                    else:
                        batch_size = in_hpars[key]

            # Create the subclass, but do not initialise.
            if batch_size > 1:
                return super(cls_this, cls).__new__(OnlineStandardScalerBatch)
            else:
                return super(cls_this, cls).__new__(OnlineStandardScalerIncremental)
            
        # If this method was called with any other class, propagate upwards.
        else:
            return super(cls_this, cls).__new__(cls)
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = preprocessing.StandardScaler()
        self.name += "_Standard"

    @staticmethod
    def new_hpars():
        hpars = dict()
        hpars["batch_size"] = HPInt(in_default = 1, in_min = 1, in_max = 1000,
                                    is_log_scale = True)
        return hpars
    
class OnlineStandardScalerBatch(OnlineStandardScaler):

    is_unselectable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "Batch"

        self.format_x = DataFormatX.PANDAS_DATAFRAME
        self.format_y = DataFormatY.PANDAS_SERIES

    def learn(self, x, y = None):
        batch_size = self.hpars["batch_size"].val
        assignments = np.arange(len(x))//batch_size
        for x_batch in x.groupby(assignments):
            self.model.learn_many(X=x_batch[-1])

    def transform(self, x):
        return self.model.transform_many(X=x)

class OnlineStandardScalerIncremental(OnlineStandardScaler):

    is_unselectable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "Increment"

        self.format_x = DataFormatX.LIST_OF_DICTS
        self.format_y = DataFormatY.LIST

    def learn(self, x, y = None):
        for x_increment in x:
            self.model.learn_one(x=x_increment)

    def transform(self, x):
        x_new = list()
        for x_increment in x:
            x_new.append(self.model.transform_one(x=x_increment))
        return x_new



class RiverPredictor(MLOnlineLearner, MLPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_River"

class OnlineLinearRegressor(MLRegressor, RiverPredictor):
    def __new__(cls, in_hpars = None, *args, **kwargs):

        cls_this = OnlineLinearRegressor

        # When a new object of this class is created, create the right subclass.
        if cls == cls_this:
            key = "batch_size"

            # Check default hyperparameters.
            hpars = cls.new_hpars()
            batch_size = hpars[key].val

            # Check supplied hyperparameters, if any.
            if not in_hpars is None:
                if key in in_hpars:
                    if isinstance(in_hpars[key], Hyperparameter):
                        batch_size = in_hpars[key].val
                    else:
                        batch_size = in_hpars[key]

            # Create the subclass, but do not initialise.
            if batch_size > 1:
                return super(cls_this, cls).__new__(OnlineLinearRegressorBatch)
            else:
                return super(cls_this, cls).__new__(OnlineLinearRegressorIncremental)
            
        # If this method was called with any other class, propagate upwards.
        else:
            return super(cls_this, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = linear_model.LinearRegression()
        self.name += "_LinearRegressor"

        # Update the SGD weight optimiser and intercept learning rate.
        self.model.intercept_lr.learning_rate = float(self.hpars["learning_rate"].val)
        self.model.optimizer.lr.learning_rate = float(self.hpars["learning_rate"].val)

    @staticmethod
    def new_hpars():
        hpars = dict()
        hpars["batch_size"] = HPInt(in_default = 1, in_min = 1, in_max = 1000,
                                    is_log_scale = True)
        hpars["learning_rate"] = HPFloat(in_default = 0.001, in_min = 0.000001, in_max = 1,
                                         is_log_scale = True)
        return hpars

    # def score(self, x, y):
    #     # TODO: Generalise the metrics.
    #     metric = metrics.R2()
    #     for y_true, y_pred in zip(y, self.query(x)):
    #         metric.update(y_true, y_pred)
    #     value = metric.get()
    #     return value
    
class OnlineLinearRegressorBatch(OnlineLinearRegressor):

    is_unselectable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "Batch"

        self.format_x = DataFormatX.PANDAS_DATAFRAME
        self.format_y = DataFormatY.PANDAS_SERIES

    def learn(self, x, y):
        batch_size = self.hpars["batch_size"].val
        assignments = np.arange(len(x))//batch_size
        for x_batch, y_batch in zip(x.groupby(assignments), y.groupby(assignments)):
            self.model.learn_many(X=x_batch[-1], y=y_batch[-1])

    def query(self, x):
        return self.model.predict_many(X=x)

class OnlineLinearRegressorIncremental(OnlineLinearRegressor):

    is_unselectable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "Increment"

        self.format_x = DataFormatX.LIST_OF_DICTS
        self.format_y = DataFormatY.LIST

    def learn(self, x, y):
        for x_increment, y_increment in zip(x, y):
            self.model.learn_one(x=x_increment, y=y_increment)

    def query(self, x):
        response = list()
        for x_increment in x:
            response.append(self.model.predict_one(x=x_increment))
        return response