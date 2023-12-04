# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:49:01 2023

@author: David J. Kedziora
"""

from ..component import MLPreprocessor, MLPredictor
from ..data import DataFormatX, DataFormatY

from sklearn import (preprocessing, linear_model, cross_decomposition,
                     dummy, svm)



class SKLearnPreprocessor(MLPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_SKLearnPrep"
        self.format_x = DataFormatX.NUMPY_ARRAY

class StandardScaler(SKLearnPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = preprocessing.StandardScaler()
        self.name += "_StandardScaler"

    def learn(self, x, y):
        self.model.fit(X=x, y=y)

    def adapt(self, x, y):
        self.model.partial_fit(X=x, y=y)

    def transform(self, x):
        return self.model.transform(X=x)



class SKLearnPredictor(MLPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_SKLearnPred"
        self.format_x = DataFormatX.NUMPY_ARRAY
        self.format_y = DataFormatY.NUMPY_ARRAY

    def learn(self, x, y):
        self.model.fit(X=x, y=y)

    def query(self, x):
        return self.model.predict(X=x)
    
class DummyRegressor(SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = dummy.DummyRegressor()
        self.name += "_Dummy"

class LinearRegressor(SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = linear_model.LinearRegression()
        self.name += "_LinearRegressor"

    def get_feature_importance(self):
        return self.model.coef_
    
class SGDRegressor(SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = linear_model.SGDRegressor()
        self.name += "_SGDRegressor"
    
    def adapt(self, x, y):
        self.model.partial_fit(X=x, y=y)

    def get_feature_importance(self):
        return self.model.coef_
    
class LinearSupportVectorRegressor(SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = svm.LinearSVR()
        self.name += "_LinearSVR"

    def get_feature_importance(self):
        return self.model.coef_

class PartialLeastSquaresRegressor(SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = cross_decomposition.PLSRegression(n_components=1)
        self.name += "_PLSR"
        self.format_y = DataFormatY.LIST_OF_LISTS_ACROSS_TARGETS

    def learn(self, x, y):
        self.model.fit(X=x, Y=y)

    def get_feature_importance(self):
        return self.model.coef_.T[0]