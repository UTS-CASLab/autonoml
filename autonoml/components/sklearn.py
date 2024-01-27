# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:49:01 2023

@author: David J. Kedziora
"""

from ..hyperparameter import HPCategorical, HPInt, HPFloat
from ..component import (MLPreprocessor, MLPredictor, 
                         MLOnlineLearner, MLDummy,
                         MLImputer, MLScaler, 
                         MLClassifier, MLRegressor)
from ..data import DataFormatX, DataFormatY

from sklearn import (impute, preprocessing, 
                     linear_model, cross_decomposition,
                     dummy, svm,
                     ensemble)



class SKLearnPreprocessor(MLPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_SKLearn"
        self.format_x = DataFormatX.NUMPY_ARRAY_2D

class SimpleImputer(MLImputer, SKLearnPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = impute.SimpleImputer()
        self.name += "_Simple"

    def learn(self, x, y):
        self.model.fit(X=x, y=y)

    def transform(self, x):
        return self.model.transform(X=x)

class StandardScaler(MLScaler, MLOnlineLearner, SKLearnPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = preprocessing.StandardScaler()
        self.name += "_Standard"

    def learn(self, x, y):
        self.model.fit(X=x, y=y)

    def adapt(self, x, y):
        self.model.partial_fit(X=x, y=y)

    def transform(self, x):
        return self.model.transform(X=x)



class SKLearnPredictor(MLPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_SKLearn"
        self.format_x = DataFormatX.NUMPY_ARRAY_2D
        self.format_y = DataFormatY.NUMPY_ARRAY_1D

    def learn(self, x, y):
        self.model.fit(X=x, y=y)

    def query(self, x):
        return self.model.predict(X=x)
    

    
class DummyClassifier(MLClassifier, MLDummy, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = dummy.DummyClassifier()
        self.name += ""

class LogisticRegressor(MLClassifier, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = linear_model.LogisticRegression()
        self.name += "_Logistic"
    
class LinearSVC(MLClassifier, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = svm.LinearSVC()
        self.name += "_LinearSVC"

        

class Perceptron(MLClassifier, MLOnlineLearner, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = linear_model.Perceptron()
        self.name += "_Perceptron"

    def adapt(self, x, y):
        self.model.partial_fit(X=x, y=y)

class SGDClassifier(MLClassifier, MLOnlineLearner, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = linear_model.SGDClassifier()
        self.name += "_SGD"
    
    def adapt(self, x, y):
        self.model.partial_fit(X=x, y=y)


    

    
class DummyRegressor(MLRegressor, MLDummy, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = dummy.DummyRegressor()
        self.name += ""

class LinearRegressor(MLRegressor, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = linear_model.LinearRegression()
        self.name += "_Linear"

    def get_feature_importance(self):
        return self.model.coef_
    


class LinearSVR(MLRegressor, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_LinearSVR"

        loss = self.hpars["loss"].val
        if loss == "L1":
            loss_translated = "epsilon_insensitive"
        elif loss == "L2":
            loss_translated = "squared_epsilon_insensitive"

        self.model = svm.LinearSVR(epsilon = self.hpars["epsilon"].val,
                                   C = self.hpars["C"].val,
                                   loss = loss_translated,
                                   dual = "auto")

    @staticmethod
    def new_hpars():
        hpars = dict()
        info = ("The acceptable error, defining the width the 'SVR tube'.")
        hpars["epsilon"] = HPFloat(in_default = 0.0, in_min = 0.0, in_max = 1.0,
                                   in_info = info)
        info = ("Regularisation parameter.")
        hpars["C"] = HPFloat(in_default = 1.0, in_min = 0.01, in_max = 100.0,
                             is_log_scale = True, in_info = info)
        info = ("Loss function.")
        losses = ["L1", "L2"]
        hpars["loss"] = HPCategorical(in_options = losses, in_info = info)
        return hpars

    def get_feature_importance(self):
        return self.model.coef_



class PLSRegressor(MLRegressor, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_PLSR"
        self.format_y = DataFormatY.LIST_OF_LISTS_ACROSS_TARGETS

        self.model = cross_decomposition.PLSRegression(n_components=1)

    def learn(self, x, y):
        self.model.fit(X=x, Y=y)

    def get_feature_importance(self):
        return self.model.coef_.T[0]
    


class RandomForestRegressor(MLRegressor, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_RandomForest"

        max_depth = self.hpars["max_depth"].val
        if max_depth == 0:
            max_depth = None

        self.model = ensemble.RandomForestRegressor(n_estimators = self.hpars["n_estimators"].val,
                                                    max_depth = max_depth,
                                                    min_samples_leaf = self.hpars["min_samples_leaf"].val,
                                                    min_samples_split = self.hpars["min_samples_split"].val)

    @staticmethod
    def new_hpars():
        hpars = dict()
        hpars["n_estimators"] = HPInt(in_default = 32, in_min = 2, in_max = 512,
                                      is_log_scale = True)
        hpars["max_depth"] = HPInt(in_default = 0, in_min = 0, in_max = 100, 
                                   in_info = "Max tree depth. Zero is unrestricted.")
        hpars["min_samples_leaf"] = HPInt(in_default = 1, in_min = 1, in_max = 8, is_log_scale = True)
        hpars["min_samples_split"] = HPInt(in_default = 2, in_min = 2, in_max = 16, is_log_scale = True)
        return hpars

    # TODO: Verify.
    def get_feature_importance(self):
        return self.model.feature_importances_
    


class GradientBoostingRegressor(MLRegressor, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_GradientBoosting"

        max_depth = self.hpars["max_depth"].val
        if max_depth == 0:
            max_depth = None

        self.model = ensemble.GradientBoostingRegressor(learning_rate = self.hpars["learning_rate"].val,
                                                        n_estimators = self.hpars["n_estimators"].val,
                                                        max_depth = max_depth,
                                                        min_samples_leaf = self.hpars["min_samples_leaf"].val,
                                                        min_samples_split = self.hpars["min_samples_split"].val)

    @staticmethod
    def new_hpars():
        hpars = dict()
        hpars["learning_rate"] = HPFloat(in_default = 0.1, in_min = 0.01, in_max = 1.0,
                                         is_log_scale = True)
        hpars["n_estimators"] = HPInt(in_default = 16, in_min = 1, in_max = 256,
                                      is_log_scale = True)
        hpars["max_depth"] = HPInt(in_default = 4, in_min = 0, in_max = 16, 
                                   in_info = "Max tree depth. Zero is unrestricted.")
        hpars["min_samples_leaf"] = HPInt(in_default = 1, in_min = 1, in_max = 8, is_log_scale = True)
        hpars["min_samples_split"] = HPInt(in_default = 2, in_min = 2, in_max = 16, is_log_scale = True)
        return hpars

    # TODO: Verify.
    def get_feature_importance(self):
        return self.model.feature_importances_
    

    
class SGDRegressor(MLRegressor, MLOnlineLearner, SKLearnPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_SGD"

        self.model = linear_model.SGDRegressor(alpha = self.hpars["alpha"].val,
                                               eta0 = self.hpars["eta_zero"].val,
                                               learning_rate = self.hpars["learning_method"].val)

    @staticmethod
    def new_hpars():
        hpars = dict()

        methods = ["invscaling", "constant", "optimal", "adaptive"]
        hpars["learning_method"] = HPCategorical(in_options = methods)
        hpars["eta_zero"] = HPFloat(in_default = 0.01, in_min = 0.0001, in_max = 1.0,
                                    is_log_scale = True, in_info = "Initial learning rate.")
        hpars["alpha"] = HPFloat(in_default = 0.0001, in_min = 10**-7, in_max = 1.0,
                                 is_log_scale = True, in_info = "Regularisation factor.")
        return hpars
    
    def adapt(self, x, y):
        self.model.partial_fit(X=x, y=y)

    def get_feature_importance(self):
        return self.model.coef_