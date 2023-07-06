# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:16:42 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp

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
    
    def learn(self, x, y):
        if self.model:
            self.model.fit(x, y)
        else:
            text_error = "Cannot fit an empty MLComponent to data."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
            
    def score(self, x, y):
        if self.model:
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
            
            
    
class PartialLeastSquaresRegressor(MLComponent):
    def __init__(self):
        super().__init__()
        self.model = PLSRegression(n_components=1)
        self.name += "_PLSR"
        
# class OnlineLogisticRegressor(MLComponent):
#     def __init__(self):
#         super().__init__()
#         self.model = PLSRegression(n_components=1)