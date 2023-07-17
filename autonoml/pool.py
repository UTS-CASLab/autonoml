# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:16:42 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .plot import plot_feature_importance, plot_performance
from .data_storage import DataFormat

from river import linear_model
from river import metrics

from sklearn.cross_decomposition import PLSRegression

import functools

def deco_learn(learn):
    @functools.wraps(learn)
    def wrapper_decorator(self, x, y):
        if self.model:
            try:
                learn(self, x, y)
            except Exception as e:
                raise(e)
        else:
            text_error = "Cannot fit an empty MLComponent to data."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
    return wrapper_decorator

def deco_score(score):
    @functools.wraps(score)
    def wrapper_decorator(self, x, y, do_remember = False, for_training = False):
        if self.model:
            if do_remember:
                if for_training:
                    self.training_y_true.extend(y)
                    self.training_y_response.extend(self.query(x))
                else:
                    self.testing_y_true.extend(y)
                    self.testing_y_response.extend(self.query(x))
            try:
                value = score(self, x, y)
            except Exception as e:
                raise(e)
            return value
        else:
            text_error = "Cannot score an empty MLComponent on test data."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
    return wrapper_decorator

def deco_query(query):
    @functools.wraps(query)
    def wrapper_decorator(self, x):
        if self.model:
            try:
                response = query(self, x)
            except Exception as e:
                raise(e)
            return response
        else:
            text_error = "Cannot query an empty MLComponent on its response to data."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
    return wrapper_decorator

class MLComponent:
    """
    A base class for components of a learning pipeline.
    """
    
    count = 0
    
    def __init__(self):
        self.model = None
        self.name = str(MLComponent.count)
        self.data_format = None
        MLComponent.count += 1
        
        self.training_y_true = list()
        self.training_y_response = list()
        self.testing_y_true = list()
        self.testing_y_response = list()
    
    # def learn(self, x, y):
    #     if self.model:
    #         self.model.fit(x=x, y=y)
    #     else:
    #         text_error = "Cannot fit an empty MLComponent to data."
    #         log.error("%s - %s" % (Timestamp(), text_error))
    #         raise Exception(text_error)
            
    # def score(self, x, y, do_remember = False, for_training = False):
    #     if self.model:
    #         if do_remember:
    #             if for_training:
    #                 self.training_y_true.extend(y)
    #                 self.training_y_response.extend([k[0] for k in self.query(x)])
    #             else:
    #                 self.testing_y_true.extend(y)
    #                 self.testing_y_response.extend([k[0] for k in  self.query(x)])
    #         return self.model.score(x=x, y=y)
    #     else:
    #         text_error = "Cannot score an empty MLComponent on test data."
    #         log.error("%s - %s" % (Timestamp(), text_error))
    #         raise Exception(text_error)
            
    # def query(self, x):
    #     if self.model:
    #         return self.model.predict(x=x)
    #     else:
    #         text_error = "Cannot query an empty MLComponent on its response to data."
    #         log.error("%s - %s" % (Timestamp(), text_error))
    #         raise Exception(text_error)

    def get_feature_importance(self):
        return None

    def inspect_structure(self, in_keys_features):
        if self.model:
            importance = self.get_feature_importance()
            if not importance is None:
                plot_feature_importance(in_keys_features = in_keys_features,
                                        in_importance = importance,
                                        in_title = "Feature Importance: " + self.name)
        else:
            text_error = "Cannot inspect the structure of an empty MLComponent."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
            
    def inspect_performance(self, for_training = False):
        if self.model:
            if for_training:
                text_type = "Training"
                vals_response = self.training_y_response
                vals_true = self.training_y_true
            else:
                text_type = "Testing"
                vals_response = self.testing_y_response
                vals_true = self.testing_y_true

            if self.data_format == DataFormat.LIST_OF_LISTS:
                # Flatten into lists that can be plotted.
                vals_response = [list(row) for row in zip(*vals_response)][0]
                vals_true = [list(row) for row in zip(*vals_true)][0]

            plot_performance(in_vals_response = vals_response,
                             in_vals_true = vals_true,
                             in_title = "Performance (" + text_type + "): " + self.name)
                
        else:
            text_error = "Cannot inspect the performance of an empty MLComponent."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
            
            
    
class PartialLeastSquaresRegressor(MLComponent):
    def __init__(self):
        super().__init__()
        self.model = PLSRegression(n_components=1)
        self.name += "_PLSR"
        self.data_format = DataFormat.LIST_OF_LISTS

    @deco_learn
    def learn(self, x, y):
        self.model.fit(X=x, Y=y)

    @deco_score
    def score(self, x, y):
        return self.model.score(X=x, y=y)

    @deco_query
    def query(self, x):
        return self.model.predict(X=x)

    def get_feature_importance(self):
        return self.model.coef_.T[0]

        
class OnlineLinearRegressor(MLComponent):
    def __init__(self):
        super().__init__()
        self.model = linear_model.LinearRegression()
        self.name += "_LinearRegressor"
        self.data_format = DataFormat.DATAFRAME

    @deco_learn
    def learn(self, x, y):
        self.model.learn_many(X=x, y=y)

    @deco_score
    def score(self, x, y):
        metric = metrics.R2()
        for y_true, y_pred in zip(y, self.query(x)):
            metric.update(y_true, y_pred)
        value = metric.get()
        print(value)
        return value

    @deco_query
    def query(self, x):
        return self.model.predict_many(X=x)