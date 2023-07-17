# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:16:42 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .plot import plot_feature_importance, plot_performance

# from river import linear_model
# from river import metrics

from sklearn.cross_decomposition import PLSRegression

class MLComponent:
    """
    A base class for components of a learning pipeline.
    """
    
    count = 0
    
    def __init__(self):
        self.model = None
        self.name = str(MLComponent.count)
        MLComponent.count += 1
        
        self.training_y_true = list()
        self.training_y_response = list()
        self.testing_y_true = list()
        self.testing_y_response = list()
    
    def learn(self, x, y):
        if self.model:
            self.model.fit(x, y)
        else:
            text_error = "Cannot fit an empty MLComponent to data."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
            
    def score(self, x, y, do_remember = False, for_training = False):
        if self.model:
            if do_remember:
                if for_training:
                    self.training_y_true.extend(y)
                    self.training_y_response.extend([k[0] for k in self.query(x)])
                else:
                    self.testing_y_true.extend(y)
                    self.testing_y_response.extend([k[0] for k in  self.query(x)])
            return self.model.score(x, y)
        else:
            text_error = "Cannot score an empty MLComponent on test data."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
            
    def query(self, x):
        if self.model:
            return self.model.predict(x)
        else:
            text_error = "Cannot query an empty MLComponent on its response to data."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
            
    def inspect_structure(self, in_keys_features):
        if self.model:
            plot_feature_importance(in_keys_features = in_keys_features,
                                    in_importance = self.model.coef_.T[0])
        else:
            text_error = "Cannot inspect the structure of an empty MLComponent."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
            
    def inspect_performance(self, for_training = False):
        if self.model:
            if for_training:
                plot_performance(in_vals_response = self.training_y_response,
                                 in_vals_true = self.training_y_true,
                                 in_title = "Performance (Training): " + self.name)
            else:
                plot_performance(in_vals_response = self.testing_y_response,
                                 in_vals_true = self.testing_y_true,
                                 in_title = "Performance (Testing): " + self.name)
                
        else:
            text_error = "Cannot inspect the performance of an empty MLComponent."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
            
            
    
class PartialLeastSquaresRegressor(MLComponent):
    def __init__(self):
        super().__init__()
        self.model = PLSRegression(n_components=1)
        self.name += "_PLSR"
        
# class OnlineLogisticRegressor(MLComponent):
#     def __init__(self):
#         super().__init__()
#         self.model = PLSRegression(n_components=1)