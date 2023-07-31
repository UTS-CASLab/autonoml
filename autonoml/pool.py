# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:16:42 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .plot import plot_feature_importance, plot_performance
from .data import (DataFormatX, DataFormatY,
                   reformat_x, reformat_y)

from copy import deepcopy
import numpy as np

from sklearn import (preprocessing as skprep,
                     linear_model as sklin,
                     cross_decomposition,
                     dummy, svm)
                     
from river import (preprocessing as rivprep, 
                   linear_model as rivlin, 
                   metrics)

class Hyperparameter:
    """
    A class containing information for a hyperparameter of an MLComponent.
    This information guides hyperparameter optimisation (HPO).
    """

    def __init__(self, in_val = None, in_default = None):
        self.min = None
        self.max = None
        self.default = in_default



        if in_val is None:
            self.val = self.default
        else:
            self.val = in_val


class MLComponent:
    """
    A base class for components of a learning pipeline.
    """
    
    count = 0
    
    def __init__(self, in_hpars = None, *args, **kwargs):
        self.model = None
        self.name = str(MLComponent.count)
        MLComponent.count += 1

        # Every component has a default dictionary of hyperparameters.
        self.hpars = self.new_hpars()

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



#%% Scikit-learn components.

class SKLearnPreprocessor(MLPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_SKLearnPrep"
        self.format_x = DataFormatX.LIST_OF_LISTS_ACROSS_FEATURES

class StandardScaler(SKLearnPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = skprep.StandardScaler()
        self.name += "_StandardScaler"

    def learn(self, x, y):
        self.model.fit(X=x, y=y)

    def transform(self, x):
        return self.model.transform(X=x)



class SKLearnPredictor(MLPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_SKLearnPred"
        self.format_x = DataFormatX.LIST_OF_LISTS_ACROSS_FEATURES
        self.format_y = DataFormatY.LIST

    def learn(self, x, y):
        self.model.fit(X=x, y=y)

    def score(self, x, y):
        return self.model.score(X=x, y=y)

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
        self.model = sklin.LinearRegression()
        self.name += "_LinearRegressor"

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



#%% River components.

class RiverPreprocessor(MLPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_RiverPrep"

class OnlineStandardScaler(RiverPreprocessor):
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
            return super(cls_this, cls).__new__(cls, in_hpars, *args, **kwargs)
        
    @staticmethod
    def new_hpars():
        hpars = dict()
        hpars["batch_size"] = Hyperparameter(in_default = 1)
        return hpars

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = rivprep.StandardScaler()
        self.name += "_StandardScaler"
    
class OnlineStandardScalerBatch(OnlineStandardScaler):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "Increment"

        self.format_x = DataFormatX.LIST_OF_DICTS_ACROSS_FEATURES
        self.format_y = DataFormatY.LIST

    def learn(self, x, y = None):
        for x_increment in x:
            self.model.learn_one(x=x_increment)

    def transform(self, x):
        x_new = list()
        for x_increment in x:
            x_new.append(self.model.transform_one(x=x_increment))
        return x_new



class RiverPredictor(MLPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "_RiverPred"

class OnlineLinearRegressor(RiverPredictor):
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
            return super(cls_this, cls).__new__(cls, in_hpars, *args, **kwargs)
        
    @staticmethod
    def new_hpars():
        hpars = dict()
        hpars["batch_size"] = Hyperparameter(in_default = 1)
        return hpars

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = rivlin.LinearRegression()
        self.name += "_LinearRegressor"

    def score(self, x, y):
        # TODO: Generalise the metrics.
        metric = metrics.R2()
        for y_true, y_pred in zip(y, self.query(x)):
            metric.update(y_true, y_pred)
        value = metric.get()
        return value
    
class OnlineLinearRegressorBatch(OnlineLinearRegressor):
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += "Increment"

        self.format_x = DataFormatX.LIST_OF_DICTS_ACROSS_FEATURES
        self.format_y = DataFormatY.LIST

    def learn(self, x, y):
        for x_increment, y_increment in zip(x, y):
            self.model.learn_one(x=x_increment, y=y_increment)

    def query(self, x):
        response = list()
        for x_increment in x:
            response.append(self.model.predict_one(x=x_increment))
        return response
    


#%% Pipeline code.

# TODO: Upgrade to DAGs.
class MLPipeline:
    """
    A class for a sequential learning pipeline.
    The pipeline contains a list of learning components.
    The last must be a predictor.
    """

    count = 0

    def __init__(self, in_keys_features, in_key_target, in_components = None):
        self.name = "Pipe_" + str(MLPipeline.count)
        MLPipeline.count += 1

        log.info("%s - Constructing MLPipeline '%s'." % (Timestamp(), self.name))

        self.components = list()

        if in_components is None:
            # TODO: Implement and use DummyClassifier depending on the problem.
            in_components = [DummyRegressor()]

        for component in in_components:

            # Pass on the relevant variable keys to individual components.
            # TODO: Consider feature selection/generation down pipeline.
            component.keys_features = deepcopy(in_keys_features)
            if isinstance(component, MLPredictor):
                component.key_target = in_key_target

            self.components.append(component)
            log.info("%s   Attached MLComponent '%s'." 
                        % (Timestamp(None), component.name))

        if not isinstance(self.components[-1], MLPredictor):
            text_error = "The last component in an MLPipeline must be an MLPredictor."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)

        self.training_y_true = list()
        self.training_y_response = list()
        self.testing_y_true = list()
        self.testing_y_response = list()

    def components_as_string(self):
        return " -> ".join([component.name for component in self.components])

    def process(self, x, y, in_format_x = None, in_format_y = None, 
                do_learn = True, do_query = True, do_score = True,
                do_remember = False, for_training = False):

        # Assume data is provided from storage in default format.
        # It will be reformatted as required by components.
        format_x = in_format_x
        if format_x is None:
            format_x = DataFormatX(0)
        format_y = in_format_y
        if format_y is None:
            format_y = DataFormatY(0)

        response = None
        metric = None

        num_components = len(self.components)
        for idx_component in range(num_components):

            # Every component learns from the incoming features and possibly the target.
            component = self.components[idx_component]
            x, format_x = component.reformat_x(x = x, in_format_old = format_x)
            y, format_y = component.reformat_y(y = y, in_format_old = format_y)
            if do_learn:
                component.learn(x=x, y=y)

            if isinstance(component, MLPreprocessor):
                # Preprocessing components modify the propagated feature set.
                x = component.transform(x=x)
            elif isinstance(component, MLPredictor):
                # TODO: Pipe predictions to the next component as a feature.
                pass
                # Any predictors in the pipeline are queried and scored.
                if do_query:
                    response = component.query(x=x)
                if do_score:
                    metric = component.score(x=x, y=y)

        # If there is a response, reformat it to the standard data format.
        if do_query:
            response = reformat_y(in_data = response,
                                  in_format_old = format_y,
                                  in_format_new = DataFormatY(0))

            # Memorise the true target and the final response.
            if do_remember:
                y = reformat_y(in_data = y,
                               in_format_old = format_y,
                               in_format_new = DataFormatY(0))

                # TODO: Decide if do_learn and for_training are identical.
                if for_training:
                    self.training_y_true.extend(y)
                    self.training_y_response.extend(response)
                else:
                    self.testing_y_true.extend(y)
                    self.testing_y_response.extend(response)

        # The final predictor in the pipeline has its response/score returned.
        return response, metric

    def inspect_structure(self):
        for component in self.components:
            if isinstance(component, MLPredictor):
                importance = component.get_feature_importance()
                if not importance is None:
                    plot_feature_importance(in_keys_features = component.keys_features,
                                            in_importance = importance,
                                            in_title = "Feature Importance: " + component.name)
            
    def inspect_performance(self, for_training = False):
        if for_training:
            text_type = "Training"
            vals_response = self.training_y_response
            vals_true = self.training_y_true
        else:
            text_type = "Testing"
            vals_response = self.testing_y_response
            vals_true = self.testing_y_true

        title = "Performance (%s): %s\n%s" % (text_type, self.name, self.components_as_string())
        plot_performance(in_vals_response = vals_response,
                         in_vals_true = vals_true,
                         in_title = title)